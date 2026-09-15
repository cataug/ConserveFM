from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import CONSTRAINT_TYPES, CORRUPTION_TYPES, MethodConfig


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8 if width >= 8 else 1, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(8 if width >= 8 else 1, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h


@dataclass
class RepairOutput:
    state_norm: torch.Tensor
    delta_norm: torch.Tensor
    localization_logits: torch.Tensor | None
    type_logits: torch.Tensor | None
    router_weights: torch.Tensor


class ConserveRepair(nn.Module):
    def __init__(
        self,
        channels: int,
        cfg: MethodConfig,
        width: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.cfg = cfg
        self.constraint_names = list(cfg.enabled_constraints)

        self.lead_embed = nn.Sequential(
            nn.Linear(1, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.in_proj = nn.Conv2d(channels * 2, width, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.out_proj = nn.Conv2d(width, channels, 3, padding=1)

        self.localization_head = (
            nn.Conv2d(width, 1, 1) if cfg.localization_head else None
        )
        self.type_head = (
            nn.Linear(width, len(CORRUPTION_TYPES)) if cfg.constraint_type_head else None
        )

        n_constraints = max(1, len(self.constraint_names))
        if cfg.router_mode == "adaptive":
            self.router_head = nn.Linear(width + 1, n_constraints)
            self.static_router_logits = None
        elif cfg.router_mode == "learned_static":
            self.router_head = None
            self.static_router_logits = nn.Parameter(torch.zeros(n_constraints))
        else:
            self.router_head = None
            self.static_router_logits = None

    def _router(self, pooled: torch.Tensor, lead_hours: torch.Tensor) -> torch.Tensor:
        n = len(self.constraint_names)
        if n == 0 or self.cfg.router_mode == "off":
            return torch.zeros((pooled.shape[0], 0), device=pooled.device, dtype=pooled.dtype)

        if self.cfg.router_mode == "adaptive":
            lead = torch.log1p(lead_hours.float()).to(pooled.dtype)[:, None] / 5.0
            logits = self.router_head(
                torch.cat([pooled, lead], dim=1)
            ).float()
            return torch.softmax(logits, dim=1)

        if self.cfg.router_mode == "learned_static":
            return torch.softmax(self.static_router_logits, dim=0)[None].expand(pooled.shape[0], -1)

        if self.cfg.router_mode == "manual":
            vals = torch.tensor(
                [self.cfg.manual_constraint_weights[name] for name in self.constraint_names],
                device=pooled.device,
                dtype=pooled.dtype,
            )
            vals = vals / vals.sum().clamp_min(1e-8)
            return vals[None].expand(pooled.shape[0], -1)

        # equal
        return torch.full(
            (pooled.shape[0], n),
            1.0 / n,
            device=pooled.device,
            dtype=pooled.dtype,
        )

    def forward(
        self,
        raw_norm: torch.Tensor,
        previous_norm: torch.Tensor,
        lead_hours: torch.Tensor,
    ) -> RepairOutput:
        x = torch.cat([raw_norm, previous_norm], dim=1)
        h = self.in_proj(x)
        lead = self.lead_embed(torch.log1p(lead_hours.float())[:, None] / 5.0).to(h.dtype)
        h = h + lead[:, :, None, None]
        h = self.blocks(h)
        pred = self.out_proj(F.gelu(h))

        if self.cfg.prediction_mode == "residual":
            delta = pred
            state = raw_norm + delta
        else:
            state = pred
            delta = state - raw_norm

        loc = self.localization_head(h) if self.localization_head is not None else None
        pooled = h.mean(dim=(-2, -1))
        type_logits = self.type_head(pooled) if self.type_head is not None else None
        router = self._router(pooled, lead_hours)
        return RepairOutput(state, delta, loc, type_logits, router)
