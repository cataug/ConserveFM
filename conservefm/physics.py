from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F

from .channels import ChannelLayout

RD = 287.05
G = 9.80665
R_EARTH = 6_371_000.0


def _periodic_diff_lon(x: torch.Tensor, dlon_rad: float) -> torch.Tensor:
    return (torch.roll(x, shifts=-1, dims=-2) - torch.roll(x, shifts=1, dims=-2)) / (2.0 * dlon_rad)


def _diff_lat(x: torch.Tensor, dlat_rad: float) -> torch.Tensor:
    # replicate padding is sufficient for the polar endpoints for this training residual
    xp = F.pad(x, (1, 1, 0, 0), mode="replicate")
    return (xp[..., 2:] - xp[..., :-2]) / (2.0 * dlat_rad)


def hydrostatic_residual(state: torch.Tensor, layout: ChannelLayout) -> torch.Tensor:
    gid = layout.indices("geopotential")
    tid = layout.indices("temperature")
    phi = state[:, gid]
    temp = state[:, tid]
    p = torch.as_tensor(layout.levels_hpa, device=state.device, dtype=state.dtype).clamp_min(1.0)
    ln_p = torch.log(p)
    dln = (ln_p[1:] - ln_p[:-1])[None, :, None, None]
    dphi = phi[:, 1:] - phi[:, :-1]
    tmid = 0.5 * (temp[:, 1:] + temp[:, :-1])
    # Hydrostatic: dPhi/dln(p) ~= -R_d T
    rel = (dphi / dln + RD * tmid) / (RD * tmid.abs().clamp_min(150.0))
    return rel.square().mean()


def moisture_mass_residual(
    state: torch.Tensor,
    previous: torch.Tensor,
    layout: ChannelLayout,
) -> torch.Tensor:
    qid = layout.indices("specific_humidity")
    ps = layout.one("surface_pressure")
    tp = layout.one("total_precipitation_6hr")

    p = torch.as_tensor(layout.levels_hpa, device=state.device, dtype=state.dtype)
    # positive pressure-thickness proxy at each level
    p_sorted, order = torch.sort(p)
    q = state[:, qid][:, order]
    q_prev = previous[:, qid][:, order]
    edges = torch.empty((len(p_sorted) + 1,), device=state.device, dtype=state.dtype)
    edges[1:-1] = 0.5 * (p_sorted[:-1] + p_sorted[1:])
    edges[0] = torch.clamp(p_sorted[0] - (edges[1] - p_sorted[0]), min=0.0)
    edges[-1] = p_sorted[-1] + (p_sorted[-1] - edges[-2])
    dp_pa = (edges[1:] - edges[:-1]).clamp_min(0.0) * 100.0
    weights = dp_pa[None, :, None, None] / G

    col_q = (q.clamp_min(0.0) * weights).sum(dim=1)
    col_prev = (q_prev.clamp_min(0.0) * weights).sum(dim=1)
    drift = (col_q.mean(dim=(-2, -1)) - col_prev.mean(dim=(-2, -1)))
    drift = drift / col_prev.mean(dim=(-2, -1)).abs().clamp_min(1.0)

    # Surface-pressure global mass drift, dimensionless.
    ps_now = state[:, ps].mean(dim=(-2, -1))
    ps_prev = previous[:, ps].mean(dim=(-2, -1))
    ps_drift = (ps_now - ps_prev) / ps_prev.abs().clamp_min(5e4)

    q_negative = F.relu(-state[:, qid]).mean()
    tp_negative = F.relu(-state[:, tp:tp + 1]).mean()
    return drift.square().mean() + ps_drift.square().mean() + 10.0 * q_negative + 10.0 * tp_negative


def advection_residual(
    state: torch.Tensor,
    previous: torch.Tensor,
    layout: ChannelLayout,
    lead_hours: torch.Tensor,
    longitude_deg: torch.Tensor | None = None,
    latitude_deg: torch.Tensor | None = None,
) -> torch.Tensor:
    t2 = layout.one("2m_temperature")
    u10 = layout.one("10m_u_component_of_wind")
    v10 = layout.one("10m_v_component_of_wind")
    t = state[:, t2]
    t0 = previous[:, t2]
    u = state[:, u10]
    v = state[:, v10]

    h, w = t.shape[-2:]
    dlon_rad = math.radians(360.0 / max(h, 1))
    dlat_rad = math.radians(180.0 / max(w - 1, 1))

    dtdlon = _periodic_diff_lon(t, dlon_rad)
    dtdlat = _diff_lat(t, dlat_rad)

    if latitude_deg is None:
        lat = torch.linspace(-90.0, 90.0, w, device=state.device, dtype=state.dtype)
    else:
        lat = latitude_deg.to(device=state.device, dtype=state.dtype)
    coslat = torch.cos(torch.deg2rad(lat)).abs().clamp_min(0.05)[None, None, :]
    dx = R_EARTH * coslat
    dy = R_EARTH
    dtdx = dtdlon / dx
    dtdy = dtdlat / dy

    dt = lead_hours.to(device=state.device, dtype=state.dtype).clamp_min(1.0) * 3600.0
    tendency = (t - t0) / dt[:, None, None]
    adv = u * dtdx + v * dtdy
    residual = tendency + adv
    scale = tendency.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (residual / scale).square().mean()


def _laplacian(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.roll(x, 1, -2) + torch.roll(x, -1, -2)
        + F.pad(x, (1, 1, 0, 0), mode="replicate")[..., :-2]
        + F.pad(x, (1, 1, 0, 0), mode="replicate")[..., 2:]
        - 4.0 * x
    )


def spectral_residual(state: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    e = _laplacian(state).square().mean(dim=(-2, -1))
    e0 = _laplacian(previous).square().mean(dim=(-2, -1)).detach()
    return torch.square(torch.log1p(e) - torch.log1p(e0)).mean()


def compute_physics_losses(
    state: torch.Tensor,
    previous: torch.Tensor,
    layout: ChannelLayout,
    lead_hours: torch.Tensor,
    enabled: list[str],
    latitude_deg: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    if "moisture" in enabled:
        losses["moisture"] = moisture_mass_residual(state, previous, layout)
    if "hydrostatic" in enabled:
        losses["hydrostatic"] = hydrostatic_residual(state, layout)
    if "advection" in enabled:
        losses["advection"] = advection_residual(
            state, previous, layout, lead_hours, latitude_deg=latitude_deg
        )
    if "spectral" in enabled:
        losses["spectral"] = spectral_residual(state, previous)
    return losses


def hard_project(state: torch.Tensor, layout: ChannelLayout) -> torch.Tensor:
    """Simple deterministic projection baseline; operates in physical units."""
    x = state.clone()
    qid = layout.indices("specific_humidity")
    tp = layout.one("total_precipitation_6hr")
    x[:, qid] = x[:, qid].clamp_min(0.0)
    x[:, tp:tp + 1] = x[:, tp:tp + 1].clamp_min(0.0)

    # Hydrostatic vertical projection anchored at the highest-pressure level.
    gid = layout.indices("geopotential")
    tid = layout.indices("temperature")
    p = torch.as_tensor(layout.levels_hpa, device=x.device, dtype=x.dtype)
    order = torch.argsort(p, descending=True)  # surface -> upper atmosphere
    phi = x[:, gid][:, order].clone()
    temp = x[:, tid][:, order]
    psort = p[order]
    for k in range(1, len(order)):
        dln = torch.log(psort[k] / psort[k - 1])
        tmid = 0.5 * (temp[:, k] + temp[:, k - 1])
        phi[:, k] = phi[:, k - 1] - RD * tmid * dln
    inv = torch.argsort(order)
    x[:, gid] = phi[:, inv]
    return x
