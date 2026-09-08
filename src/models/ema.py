"""Exponential moving average of generator weights.

Sampling from an EMA copy rather than the live generator is worth several FID
points for essentially no cost, and it smooths over the step-to-step thrash that
a batch size of 8 produces. Always evaluate and export the EMA copy.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 1000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _decay_at(self, step: int) -> float:
        """Ramp the decay in over `warmup_steps`.

        At step 0 the weights are random, so a flat 0.999 decay would keep
        dragging that noise along for thousands of steps. Ramping means the EMA
        tracks the live model closely at first and only slows down once there is
        something worth averaging.
        """
        if self.warmup_steps <= 0:
            return self.decay
        # Reaches `decay` at `warmup_steps`, and is ~0.1 at step 0.
        progress = min(1.0, (step + 1) / self.warmup_steps)
        return min(self.decay, self.decay * progress + 0.1 * (1 - progress))

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = self._decay_at(step)
        ema_params = dict(self.ema.named_parameters())
        for name, param in model.named_parameters():
            ema_params[name].mul_(d).add_(param.detach(), alpha=1 - d)
        # Buffers (BatchNorm running stats, spectral-norm vectors) are copied,
        # not averaged -- they are statistics, not learned parameters.
        ema_buffers = dict(self.ema.named_buffers())
        for name, buf in model.named_buffers():
            ema_buffers[name].copy_(buf.detach())

    def state_dict(self) -> dict:
        return self.ema.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.ema.load_state_dict(state)

    def __call__(self, *args, **kwargs):
        return self.ema(*args, **kwargs)
