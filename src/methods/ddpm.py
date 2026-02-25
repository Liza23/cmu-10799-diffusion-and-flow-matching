"""
Denoising Diffusion Probabilistic Models (DDPM)

Option A: Attribute disentanglement regularization via L_attr = BCE(g(pred_x0), c).
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: Literal["linear"] = "linear",
        attr_classifier: Optional[nn.Module] = None,
        attr_loss_weight: float = 0.0,
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.attr_classifier = attr_classifier
        self.attr_loss_weight = attr_loss_weight
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

        if beta_schedule != "linear":
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

        # Create linear beta schedule
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Precompute useful terms and register as buffers
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars),
        )

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================
    
    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that
    
    # =========================================================================
    # Forward process
    # =========================================================================

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract per-timestep values and reshape to [B, 1, 1, 1]."""
        out = a.gather(0, t)
        return out.view(-1, *((1,) * (len(x_shape) - 1)))

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement the forward (noise adding) process of DDPM
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bars, t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bars, t, x_0.shape)

        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    # =========================================================================
    # Training loss
    # =========================================================================

    def _apply_cond_drop(self, cond: Optional[torch.Tensor], cond_drop_prob: float) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """With probability cond_drop_prob replace condition with zeros. Returns (cond_dropped, keep_mask) with keep_mask (B,) 1.0 where cond was kept."""
        if cond is None:
            return None, None
        if cond_drop_prob <= 0.0:
            return cond, torch.ones(cond.shape[0], device=cond.device, dtype=torch.float32)
        keep = (torch.rand(cond.shape[0], device=cond.device) >= cond_drop_prob).float()
        mask = keep.unsqueeze(1).expand_as(cond)
        return cond * mask, keep

    def compute_loss(self, x_0: torch.Tensor, cond: Optional[torch.Tensor] = None, cond_drop_prob: float = 0.0, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        DDPM loss. Optionally conditional with classifier-free dropout.

        Args:
            x_0: Clean data (batch_size, channels, height, width)
            cond: Optional attributes (batch_size, num_attributes). Dropped with cond_drop_prob.
            cond_drop_prob: Probability of replacing cond with zeros (unconditional).
        """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        x_t, noise = self.forward_process(x_0, t)

        cond_dropped, keep_mask = self._apply_cond_drop(cond, cond_drop_prob)
        pred_noise = self.model(x_t, t, cond=cond_dropped)
        # Prevent NaN/Inf from UNet (e.g. mixed precision overflow) from poisoning loss and downstream
        pred_noise = torch.clamp(pred_noise, -20.0, 20.0)
        pred_noise = torch.nan_to_num(pred_noise, nan=0.0, posinf=20.0, neginf=-20.0)
        loss = F.mse_loss(pred_noise, noise)

        metrics = {
            "loss": loss.detach(),
            "mse": loss.detach(),
        }
        # Option A: Attribute consistency loss BCE(g(pred_x0), c) only on samples where cond was kept
        if (
            self.attr_classifier is not None
            and self.attr_loss_weight > 0
            and cond is not None
            and keep_mask is not None
        ):
            low_t = t < (self.num_timesteps // 2)
            sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, t, x_t.shape)
            sqrt_one_minus = self._extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
            pred_x0 = (x_t - sqrt_one_minus * pred_noise) / sqrt_alpha_bar
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            pred_x0 = torch.nan_to_num(pred_x0, nan=0.0, posinf=1.0, neginf=-1.0)
            with torch.amp.autocast(device_type=x_t.device.type, enabled=False):
                pred_x0_f32 = pred_x0.float()
                logits = self.attr_classifier(pred_x0_f32)
            logits = torch.clamp(logits, -10.0, 10.0)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            loss_attr_per = F.binary_cross_entropy_with_logits(logits, cond, reduction="none").mean(dim=1)
            loss_attr_per = torch.nan_to_num(loss_attr_per, nan=0.0, posinf=2.0, neginf=0.0)
            mask = keep_mask * (low_t.float().to(keep_mask.device))
            n = mask.sum().clamp(min=1.0)
            loss_attr = (loss_attr_per * mask).sum() / n
            loss_attr = torch.nan_to_num(loss_attr, nan=0.0, posinf=2.0, neginf=0.0)
            loss_attr = torch.clamp(loss_attr, 0.0, 2.0)
            loss = loss + self.attr_loss_weight * loss_attr
            metrics["loss_attr"] = loss_attr.detach()

        # Final guard: never return non-finite loss (fallback to MSE only so backward still works)
        if not torch.isfinite(loss):
            loss = F.mse_loss(pred_noise, noise)
        return loss, metrics

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    @torch.no_grad()
    def reverse_process(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        One step of the DDPM reverse process. Supports classifier-free guidance.
        """
        if guidance_scale <= 1.0 or cond is None:
            pred_noise = self.model(x_t, t, cond=cond)
        else:
            pred_noise_uncond = self.model(x_t, t, cond=None)
            pred_noise_cond = self.model(x_t, t, cond=cond)
            pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bars, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
        x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t

        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = coef1 * x_0_pred + coef2 * x_t

        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *((1,) * (len(x_t.shape) - 1)))
        x_prev = posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise
        return x_prev

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
        cond: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """One DDIM step from t -> t_prev. Supports classifier-free guidance."""
        if guidance_scale <= 1.0 or cond is None:
            pred_noise = self.model(x_t, t, cond=cond)
        else:
            pred_noise_uncond = self.model(x_t, t, cond=None)
            pred_noise_cond = self.model(x_t, t, cond=cond)
            pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

        alpha_bar_t = self._extract(self.alpha_bars, t, x_t.shape)
        alpha_bar_prev = self._extract(self.alpha_bars, t_prev, x_t.shape)

        x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

        sigma = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * torch.sqrt(
            1.0 - alpha_bar_t / alpha_bar_prev
        )
        noise = torch.randn_like(x_t)
        dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma ** 2) * pred_noise
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        sampler: Literal["ddpm", "ddim"] = "ddpm",
        eta: float = 0.0,
        cond: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample from the model. Optional cond and classifier-free guidance_scale.
        """
        self.eval_mode()
        num_steps = num_steps or self.num_timesteps
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        x_t = torch.randn(batch_size, *image_shape, device=self.device)

        timesteps = torch.linspace(
            self.num_timesteps - 1,
            0,
            num_steps,
            device=self.device,
        ).round().long()
        if sampler == "ddpm":
            for t in timesteps:
                t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                x_t = self.reverse_process(x_t, t_batch, cond=cond, guidance_scale=guidance_scale)
        elif sampler == "ddim":
            for i in range(len(timesteps)):
                t = timesteps[i]
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=self.device)
                t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                t_prev_batch = torch.full((batch_size,), t_prev, device=self.device, dtype=torch.long)
                x_t = self.ddim_step(x_t, t_batch, t_prev_batch, eta=eta, cond=cond, guidance_scale=guidance_scale)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        return x_t

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        nn.Module.to(self, device)
        self.device = device
        if self.attr_classifier is not None:
            self.attr_classifier = self.attr_classifier.to(device)
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        # TODO: add other things you want to save
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        attr_classifier = None
        attr_loss_weight = ddpm_config.get("attr_loss_weight", 0.0)
        ckpt = ddpm_config.get("attr_classifier_checkpoint")
        if ckpt and attr_loss_weight > 0:
            from src.models.attr_classifier import load_attr_classifier
            num_attrs = config.get("model", {}).get("num_attributes") or config.get("data", {}).get("num_attributes", 40)
            img_size = config.get("data", {}).get("image_size", 64)
            attr_classifier = load_attr_classifier(ckpt, device, num_attributes=num_attrs, image_size=img_size)
        instance = cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            beta_schedule=ddpm_config.get("beta_schedule", "linear"),
            attr_classifier=attr_classifier,
            attr_loss_weight=attr_loss_weight,
        )
        return instance.to(device)
