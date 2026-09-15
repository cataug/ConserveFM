from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .channels import PRESSURE_VARIABLES, SURFACE_VARIABLES, ChannelLayout, build_layout


class ERA5Reader:
    """Read the locally copied WeatherBench2 ERA5 Zarr into [C, lon, lat]."""

    def __init__(self, zarr_path: str | Path):
        import xarray as xr

        self.path = Path(zarr_path)
        self.ds = xr.open_zarr(str(self.path), consolidated=False, chunks=None)
        self.levels = np.asarray(self.ds["level"].values, dtype=np.float32)
        self.layout = build_layout(self.levels)
        self.times = np.asarray(self.ds["time"].values)
        self.longitude = np.asarray(self.ds["longitude"].values, dtype=np.float32)
        self.latitude = np.asarray(self.ds["latitude"].values, dtype=np.float32)

    def read_state(self, time_index: int) -> np.ndarray:
        pieces = []
        for var in PRESSURE_VARIABLES:
            a = np.asarray(self.ds[var].isel(time=time_index).values, dtype=np.float32)
            if a.ndim != 3:
                raise RuntimeError(f"Unexpected {var} shape: {a.shape}")
            pieces.append(a)
        for var in SURFACE_VARIABLES:
            a = np.asarray(self.ds[var].isel(time=time_index).values, dtype=np.float32)
            if a.ndim != 2:
                raise RuntimeError(f"Unexpected {var} shape: {a.shape}")
            pieces.append(a[None])
        state = np.concatenate(pieces, axis=0)
        if state.shape[0] != self.layout.n_channels:
            raise RuntimeError(
                f"Channel mismatch: got {state.shape[0]} vs layout {self.layout.n_channels}"
            )
        return state


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise ValueError("mean/std must be 1D [C]")

    def normalize_np(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean[:, None, None]) / self.std[:, None, None]

    def denormalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)[None, :, None, None]
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)[None, :, None, None]
        return x * std + mean

    def normalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)[None, :, None, None]
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)[None, :, None, None]
        return (x - mean) / std

    def save(self, path: str | Path, layout: ChannelLayout) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean,
            std=self.std,
            levels=np.asarray(layout.levels_hpa, dtype=np.float32),
            channel_names=np.asarray(layout.names, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path):
        d = np.load(path, allow_pickle=True)
        return cls(d["mean"], d["std"])


def build_or_load_stats(
    reader: ERA5Reader,
    stats_path: str | Path,
    candidate_indices: Iterable[int],
    samples: int = 128,
    seed: int = 42,
) -> Normalizer:
    stats_path = Path(stats_path)
    if stats_path.exists():
        return Normalizer.load(stats_path)

    indices = list(dict.fromkeys(int(i) for i in candidate_indices))
    rng = random.Random(seed)
    if len(indices) > samples:
        indices = rng.sample(indices, samples)

    print(f"[STATS] computing channel stats from {len(indices)} ERA5 states", flush=True)
    sum_c = np.zeros(reader.layout.n_channels, dtype=np.float64)
    sq_c = np.zeros(reader.layout.n_channels, dtype=np.float64)
    n_per_channel = 0

    for j, idx in enumerate(indices, 1):
        x = reader.read_state(idx).astype(np.float64)
        sum_c += x.sum(axis=(1, 2))
        sq_c += np.square(x).sum(axis=(1, 2))
        n_per_channel += x.shape[1] * x.shape[2]
        print(f"[STATS] {j}/{len(indices)} time_index={idx}", flush=True)

    mean = sum_c / max(n_per_channel, 1)
    var = sq_c / max(n_per_channel, 1) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    norm = Normalizer(mean.astype(np.float32), std.astype(np.float32))
    norm.save(stats_path, reader.layout)
    return norm


def read_examples(path: str | Path, split: str | None = None) -> list[dict]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split and row["split"] != split:
                continue
            rows.append(row)
    return rows


class RawForecastProvider:
    """Contract for a frozen FM forecast source."""

    def get(self, context_end_idx: int, lead_hours: int) -> np.ndarray:
        raise NotImplementedError


class PersistenceProvider(RawForecastProvider):
    def __init__(self, reader: ERA5Reader):
        self.reader = reader

    def get(self, context_end_idx: int, lead_hours: int) -> np.ndarray:
        return self.reader.read_state(context_end_idx)


class ZarrForecastProvider(RawForecastProvider):
    """
    Forecast cache contract:
      forecasts/<backbone>.zarr/
        lead_6   [time, channel, longitude, latitude]
        lead_24  [time, channel, longitude, latitude]
        lead_72  [time, channel, longitude, latitude]

    Index `time` is the ERA5 context_end_idx. Values are in physical units and use
    the exact ConserveFM 71-channel ordering recorded in attrs['channel_names'].
    """

    def __init__(self, store: str | Path, layout: ChannelLayout):
        import zarr

        self.store_path = Path(store)
        if not self.store_path.exists():
            raise FileNotFoundError(self.store_path)
        self.root = zarr.open_group(str(self.store_path), mode="r")
        expected = list(layout.names)
        actual = list(self.root.attrs.get("channel_names", []))
        if actual and actual != expected:
            raise RuntimeError(
                "Forecast-store channel_names do not match ERA5/ConserveFM layout"
            )

    def get(self, context_end_idx: int, lead_hours: int) -> np.ndarray:
        key = f"lead_{int(lead_hours)}"
        if key not in self.root:
            raise KeyError(f"Forecast cache missing array {key}")
        out = np.asarray(self.root[key][int(context_end_idx)], dtype=np.float32)
        return out


def make_forecast_provider(
    backbone: str,
    reader: ERA5Reader,
    forecast_root: str | Path,
    smoke: bool = False,
):
    store = Path(forecast_root) / f"{backbone}.zarr"
    if store.exists():
        print(f"[FORECAST] using frozen forecast cache: {store}", flush=True)
        return ZarrForecastProvider(store, reader.layout), "zarr_frozen_fm"
    if smoke:
        print(
            f"[FORECAST] WARNING: {store} missing; smoke mode uses persistence only",
            flush=True,
        )
        return PersistenceProvider(reader), "persistence_smoke_only"
    raise FileNotFoundError(
        f"Final run requires frozen raw forecasts for backbone={backbone}: {store}. "
        "No silent persistence fallback is allowed outside --smoke."
    )


class RepairDataset(Dataset):
    def __init__(
        self,
        reader: ERA5Reader,
        normalizer: Normalizer,
        examples: list[dict],
        provider: RawForecastProvider,
        max_examples: Optional[int] = None,
    ):
        self.reader = reader
        self.normalizer = normalizer
        self.examples = examples[:max_examples] if max_examples else examples
        self.provider = provider

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        row = self.examples[i]
        context_idx = int(row["context_end_idx"])
        target_idx = int(row["target_idx"])
        lead = int(row["lead_hours"])

        previous = self.reader.read_state(context_idx)
        target = self.reader.read_state(target_idx)
        raw = self.provider.get(context_idx, lead)

        return {
            "sample_id": row["sample_id"],
            "context_idx": context_idx,
            "target_idx": target_idx,
            "lead_hours": lead,
            "previous_phys": torch.from_numpy(previous),
            "raw_phys": torch.from_numpy(raw),
            "target_phys": torch.from_numpy(target),
            "previous": torch.from_numpy(self.normalizer.normalize_np(previous)),
            "raw": torch.from_numpy(self.normalizer.normalize_np(raw)),
            "target": torch.from_numpy(self.normalizer.normalize_np(target)),
        }
