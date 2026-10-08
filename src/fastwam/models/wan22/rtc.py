"""Action-prior inpainting for the original FastWAM flow sampler."""

import torch


class ActionPriorInpainting:
    """Anchor to the same initial noise: x_sigma = (1-sigma)*prior + sigma*noise.

    Update weights are per horizon position: 0 freezes, 1 leaves the native
    update unchanged, and intermediate values softly constrain each step.
    """

    def __init__(self, prior, update_weights, noise):
        self.prior = torch.as_tensor(prior, device=noise.device, dtype=torch.float32)
        if self.prior.ndim == 2:
            self.prior = self.prior.unsqueeze(0)
        if self.prior.shape != noise.shape or not torch.isfinite(self.prior).all():
            raise ValueError(f"action_prior must be finite [H,D] or [1,H,D], expected {tuple(noise.shape)}")
        horizon = noise.shape[1]
        weights = torch.zeros(horizon, device=noise.device) if update_weights is None else torch.as_tensor(
            update_weights, device=noise.device, dtype=torch.float32
        )
        if weights.shape != (horizon,) or not ((weights >= 0) & (weights <= 1)).all():
            raise ValueError(f"action_update_weights must be finite [{horizon}] values in [0, 1]")
        self.weights = weights.reshape(1, horizon, 1)
        self.noise = noise.float().clone()

    def constrain(self, sample, sigma):
        # Tensor-only inside the sampling loop; no additional RNG or host sync.
        anchor = torch.lerp(self.prior, self.noise, sigma.float())
        mixed = torch.lerp(anchor, sample.float(), self.weights).to(sample.dtype)
        mixed = torch.where(self.weights == 0, anchor.to(sample.dtype), mixed)
        return torch.where(self.weights == 1, sample, mixed)
