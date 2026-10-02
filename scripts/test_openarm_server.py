"""Exercise the deployed action-only interface without controlling a robot."""

import argparse
import json
from pathlib import Path
import urllib.request

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18010")
    parser.add_argument("--observation", type=Path, default=Path("artifacts/openarm_checks/observation.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/openarm_checks/server_checks.json"))
    args = parser.parse_args()
    with urllib.request.urlopen(args.url + "/health", timeout=10) as response:
        health = json.load(response)
    results = []
    for _ in range(2):
        request = urllib.request.Request(args.url + "/infer", data=args.observation.read_bytes(),
                                         headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
        actions = np.asarray(result["actions"])
        assert actions.shape == (32, 16) and np.isfinite(actions).all()
        results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"health": health, "results": results}, indent=2) + "\n")
    print(json.dumps({"checkpoint_step": health["checkpoint_step"], "shape": list(actions.shape),
                      "latency_ms": [r["latency_ms"] for r in results], "finite": True}, indent=2))


if __name__ == "__main__":
    main()
