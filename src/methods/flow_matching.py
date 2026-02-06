"""
Flow Matching (FM)
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class FlowMatching(BaseMethod):
    """
    Simple flow matching implementation with linear interpolation between
    data x0 and noise x1.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ):
        super().__init__(model, device)
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device) * (self.t_max - self.t_min) + self.t_min

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Linear interpolation between data x0 and noise x1.
        Returns x_t and the target velocity v = x1 - x0.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        t_view = t.view(-1, 1, 1, 1)
        x_t = (1.0 - t_view) * x_0 + t_view * noise
        v_target = noise - x_0
        return x_t, v_target, noise

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = x_0.shape[0]
        t = self._sample_t(batch_size, x_0.device)
        x_t, v_target, _ = self.forward_process(x_0, t)

        v_pred = self.model(x_t, t)
        loss = F.mse_loss(v_pred, v_target)

        metrics = {
            "loss": loss.detach(),
            "mse": loss.detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 100,
        **kwargs
    ) -> torch.Tensor:
        """
        ODE sampling with Euler steps from t=1 -> t=0.
        """
        self.eval_mode()
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        x = torch.randn(batch_size, *image_shape, device=self.device)
        dt = (self.t_max - self.t_min) / num_steps

        for i in range(num_steps):
            t = self.t_max - i * dt
            t_batch = torch.full((batch_size,), t, device=self.device)
            v = self.model(x, t_batch)
            x = x - v * dt

        return x

    def to(self, device: torch.device) -> "FlowMatching":
        nn.Module.to(self, device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["t_min"] = self.t_min
        state["t_max"] = self.t_max
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "FlowMatching":
        fm_config = config.get("flow_matching", config)
        instance = cls(
            model=model,
            device=device,
            t_min=fm_config.get("t_min", 0.0),
            t_max=fm_config.get("t_max", 1.0),
        )
        return instance.to(device)
