from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .channels import ChannelLayout
from .registry import CORRUPTION_TYPES


@dataclass
class CorruptionResult:
    state: torch.Tensor
    mask: torch.Tensor
    type_index: torch.Tensor
    type_name: str
    severity: float


def _box_mask(batch: int, h: int, w: int, device, dtype, rng: random.Random):
    mask = torch.zeros((batch, 1, h, w), device=device, dtype=dtype)
    for b in range(batch):
        bh = max(4, int(h * rng.uniform(0.15, 0.45)))
        bw = max(4, int(w * rng.uniform(0.15, 0.45)))
        y0 = rng.randint(0, max(0, h - bh))
        x0 = rng.randint(0, max(0, w - bw))
        mask[b, :, y0:y0 + bh, x0:x0 + bw] = 1.0
    return mask


def _local_scale(x: torch.Tensor) -> torch.Tensor:
    # robust-enough per-channel scale without depending on global dataset stats
    s = x.flatten(2).std(dim=-1, keepdim=True).clamp_min(1e-6)
    return s[..., None]


class PhysicsCorruptor:
    def __init__(self, layout: ChannelLayout, enabled: list[str], seed: int = 42):
        self.layout = layout
        self.enabled = list(enabled)
        self.rng = random.Random(seed)
        for name in self.enabled:
            if name not in CORRUPTION_TYPES:
                raise ValueError(name)

    def corrupt(
        self,
        state_phys: torch.Tensor,
        severity: float | None = None,
        force_type: str | None = None,
    ) -> CorruptionResult:
        if state_phys.ndim != 4:
            raise ValueError("state_phys must be [B,C,H,W]")
        if not self.enabled:
            raise RuntimeError("No enabled corruptions")

        name = force_type or self.rng.choice(self.enabled)
        if name not in self.enabled:
            raise ValueError(f"Corruption {name} is disabled")
        severity = float(severity if severity is not None else self.rng.choice([0.25, 0.5, 1.0, 2.0]))

        x = state_phys.clone()
        b, _, h, w = x.shape
        mask = _box_mask(b, h, w, x.device, x.dtype, self.rng)
        scale = _local_scale(x)

        if name == "moisture":
            ids = self.layout.indices("specific_humidity")
            tp = self.layout.one("total_precipitation_6hr")
            sign = -1.0 if self.rng.random() < 0.5 else 1.0
            factor = 1.0 + sign * 0.18 * severity
            x[:, ids] = x[:, ids] * (1.0 + mask * (factor - 1.0))
            x[:, tp:tp + 1] = x[:, tp:tp + 1] * (1.0 - mask * sign * 0.12 * severity)

        elif name == "hydrostatic":
            g_ids = self.layout.indices("geopotential")
            # level-dependent offset breaks dPhi/dlnp while preserving smooth spatial structure
            level_ramp = torch.linspace(-1.0, 1.0, len(g_ids), device=x.device, dtype=x.dtype)[None, :, None, None]
            amp = scale[:, g_ids].mean(dim=1, keepdim=True) * 0.35 * severity
            x[:, g_ids] = x[:, g_ids] + mask * amp * level_ramp

        elif name == "advection":
            affected = (
                self.layout.indices("temperature")
                + self.layout.indices("specific_humidity")
                + [self.layout.one("2m_temperature")]
            )
            shift_lon = max(1, int(round(severity * 2)))
            shift_lat = max(1, int(round(severity)))
            rolled = torch.roll(x[:, affected], shifts=(shift_lon, shift_lat), dims=(-2, -1))
            x[:, affected] = x[:, affected] * (1.0 - mask) + rolled * mask

        elif name == "spectral":
            yy = torch.arange(h, device=x.device, dtype=x.dtype)[:, None]
            xx = torch.arange(w, device=x.device, dtype=x.dtype)[None, :]
            checker = torch.sin(yy * math.pi) * torch.cos(xx * math.pi)
            checker = checker[None, None]
            amp = scale * 0.12 * severity
            x = x + mask * amp * checker

        elif name == "range_extreme":
            # sparse local spikes, including potentially invalid q/precip ranges
            noise = torch.randn_like(x)
            spike = (noise.abs() > 2.2).to(x.dtype) * noise.sign()
            x = x + mask * spike * scale * (1.5 * severity)

        else:
            raise KeyError(name)

        type_index = torch.full(
            (b,), CORRUPTION_TYPES.index(name), device=x.device, dtype=torch.long
        )
        return CorruptionResult(x, mask, type_index, name, severity)
