"""Executor-plan timing for FastWAM; no actuator commands or progress polling."""

import math

import numpy as np


class ExecutionRTC:
    interval_ns = 33_333_333

    def __init__(self, horizon, *, margin_ms=10, ramp_steps=6, ramp_rate=5,
                 max_lateness_ms=50):
        self.horizon = horizon
        self.margin_ns = int(margin_ms * 1e6)
        self.ramp_steps, self.ramp_rate = ramp_steps, ramp_rate
        self.max_lateness_ns = int(max_lateness_ms * 1e6)
        self.delay_ns = 0.0
        self.reset()

    def reset(self):
        self.plan = self.previous_plan = None
        self.last_sample = None
        self.next_latency_probe_ns = 0
        self.probe_due = False

    @staticmethod
    def sample(plan, timestamps):
        points = np.asarray(plan["positions"], dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != 16 or not len(points)
                or not np.isfinite(points).all() or plan["interval_ns"] <= 0):
            raise ValueError("RTC plan requires finite [H,16] absolute targets and a positive interval")
        times = np.arange(len(points), dtype=np.int64) * plan["interval_ns"]
        relative = timestamps - int(plan["start_timestamp_ns"])
        return np.stack([np.interp(relative, times, column) for column in points.T], -1).astype(np.float32)

    def prepare(self, observation, started_ns):
        metadata = observation.metadata
        self.probe_due = False
        if metadata.get("execution_restart"):
            self.reset()
            plan = None
        else:
            plan = observation.execution_plan
        if plan and plan.get("sample_chunk_id") != self.last_sample:
            started = plan.get("inference_started_timestamp_ns", 0)
            elapsed = plan.get("received_timestamp_ns", 0) - started
            if started and elapsed >= 0:
                self.delay_ns = 0.9 * self.delay_ns + 0.1 * elapsed if self.delay_ns else float(elapsed)
                self.last_sample = plan.get("sample_chunk_id")

        active = plan is not None and len(plan.get("positions", [])) > 0
        origin = int(observation.timestamp)
        prior = weights = None
        frozen = 0
        if active:
            if self.plan is None or self.plan["chunk_id"] != plan["chunk_id"]:
                self.previous_plan, self.plan = self.plan, plan
            frozen = max(0, int(metadata.get("rtc_frozen_steps", 0)), math.ceil(
                (max(0, started_ns - origin) + self.delay_ns + self.margin_ns) / self.interval_ns
            ))
            if frozen >= self.horizon:
                # No published chunk means no new receipt sample. Occasionally
                # remeasure inference without emitting actions to recover from a spike.
                fresh = max(0, started_ns - origin) + self.margin_ns < self.horizon * self.interval_ns
                if (fresh and int(metadata.get("rtc_frozen_steps", 0)) < self.horizon
                        and started_ns >= self.next_latency_probe_ns):
                    self.probe_due = True
                    self.next_latency_probe_ns = started_ns + 5_000_000_000
                return None
            timestamps = origin + np.arange(self.horizon, dtype=np.int64) * self.interval_ns
            prior = self.sample(plan, timestamps)
            before = timestamps < plan["start_timestamp_ns"]
            if before.any():
                previous = self.previous_plan
                if previous is None or origin < previous["start_timestamp_ns"]:
                    return None
                prior[before] = self.sample(previous, timestamps[before])
            ramp_steps = int(metadata.get("rtc_ramp_steps", self.ramp_steps))
            rate = float(metadata.get("rtc_ramp_rate", self.ramp_rate))
            if ramp_steps < 0 or not math.isfinite(rate) or rate < 0:
                raise ValueError("RTC ramp steps and rate must be nonnegative and finite")
            count = min(ramp_steps, self.horizon - frozen)
            weights = np.ones(self.horizon, dtype=np.float32)
            weights[:frozen] = 0
            if count:
                ramp = np.arange(count, dtype=np.float32) / count
                # Zero at takeover; the first free point after the ramp has weight one.
                weights[frozen:frozen + count] = ramp if rate == 0 else (
                    -np.expm1(-rate * ramp) / -np.expm1(-rate)
                )

        execution = dict(
            action_window_start=frozen,
            action_origin_timestamp_ns=origin if active else 0,
            takeover_timestamp_ns=origin + frozen * self.interval_ns if active else 0,
            based_on_chunk_id=plan["chunk_id"] if active else "",
            inference_started_timestamp_ns=started_ns,
            max_lateness_ns=self.max_lateness_ns,
        )
        return prior, weights, execution
