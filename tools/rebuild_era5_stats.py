#!/usr/bin/env python3

from pathlib import Path
import os
import random
import numpy as np

from conservefm.data import ERA5Reader, Normalizer, read_examples


ROOT = Path.home() / "ConserveFM"

ERA5 = ROOT / "data/ERA5_WeatherBench2_1979_2022"
MANIFEST = ROOT / "manifests/research_v1/era5_examples.csv"

TARGET = ROOT / "manifests/research_v1/era5_channel_stats.npz"
OLD_ONE = ROOT / "manifests/research_v1/era5_channel_stats.ONE_SAMPLE_SMOKE.npz"

N = 256
SEED = 42


def choose_key(keys, candidates):
    for k in candidates:
        if k in keys:
            return k
    return None


print("=" * 76)
print("ERA5 CHANNEL STATISTICS REBUILD")
print("=" * 76)

reader = ERA5Reader(str(ERA5))

names = list(reader.layout.names)
C = len(names)

print("channels :", C)
print("samples  :", N)
print("grid     :", len(reader.longitude), "x", len(reader.latitude))

rows = read_examples(
    str(MANIFEST),
    "train",
)

unique_indices = sorted({
    int(r["context_end_idx"])
    for r in rows
})

print("unique train states available:", len(unique_indices))

if len(unique_indices) < N:
    raise RuntimeError(
        f"Only {len(unique_indices)} unique train states, need {N}"
    )

rng = random.Random(SEED)
indices = rng.sample(unique_indices, N)

sum_x = np.zeros(C, dtype=np.float64)
sum_x2 = np.zeros(C, dtype=np.float64)
count = np.zeros(C, dtype=np.int64)

for j, idx in enumerate(indices, 1):
    x = np.asarray(
        reader.read_state(idx),
        dtype=np.float32,
    )

    if x.shape[0] != C:
        raise RuntimeError(
            f"Bad channel count at index {idx}: {x.shape}"
        )

    flat = x.reshape(C, -1)

    finite = np.isfinite(flat)

    values = np.where(
        finite,
        flat,
        0.0,
    ).astype(np.float64)

    sum_x += values.sum(axis=1)
    sum_x2 += (values * values).sum(axis=1)
    count += finite.sum(axis=1)

    if j == 1 or j % 16 == 0 or j == N:
        print(
            f"[STATS] {j:3d}/{N} "
            f"time_index={idx}",
            flush=True,
        )

mean = sum_x / count

variance = (
    sum_x2 / count
    - mean * mean
)

variance = np.maximum(
    variance,
    1e-12,
)

std = np.sqrt(variance)

if not np.all(np.isfinite(mean)):
    raise RuntimeError("Non-finite means")

if not np.all(np.isfinite(std)):
    raise RuntimeError("Non-finite stds")

if np.any(std <= 0):
    raise RuntimeError("Zero std detected")


#
# Preserve exact NPZ schema used by existing Normalizer.
#
template_path = TARGET if TARGET.exists() else OLD_ONE

if not template_path.exists():
    raise RuntimeError(
        "No previous stats file available as schema template"
    )

with np.load(
    template_path,
    allow_pickle=True,
) as old:
    payload = {
        k: old[k]
        for k in old.files
    }

print("existing NPZ keys:", sorted(payload))

mean_key = choose_key(
    payload,
    [
        "mean",
        "means",
        "channel_mean",
        "channel_means",
        "mu",
    ],
)

std_key = choose_key(
    payload,
    [
        "std",
        "stds",
        "channel_std",
        "channel_stds",
        "sigma",
    ],
)

if mean_key is None or std_key is None:
    raise RuntimeError(
        f"Cannot identify mean/std keys in {sorted(payload)}"
    )

print("mean key:", mean_key)
print("std key :", std_key)

payload[mean_key] = mean.astype(np.float32)
payload[std_key] = std.astype(np.float32)

#
# Preserve/update channel-name metadata if present.
#
for key in [
    "names",
    "channel_names",
    "variables",
]:
    if key in payload:
        payload[key] = np.asarray(names)

tmp = TARGET.with_name(
    TARGET.stem + ".256states.tmp.npz"
)

np.savez_compressed(
    tmp,
    **payload,
)

#
# Critical validation using the actual ConserveFM loader.
#
test = Normalizer.load(str(tmp))

print("[VALIDATE] Normalizer.load: OK")

backup = TARGET.with_name(
    "era5_channel_stats.FOUR_STATE_SMOKE.npz"
)

if TARGET.exists():
    if not backup.exists():
        os.replace(
            TARGET,
            backup,
        )
    else:
        TARGET.unlink()

os.replace(
    tmp,
    TARGET,
)

print()
print("=" * 76)
print("STATS COMPLETE")
print("=" * 76)
print("saved :", TARGET)
print("backup:", backup)
print("states:", N)
print("cells/channel:", int(count.min()), "..", int(count.max()))
print("mean finite:", bool(np.all(np.isfinite(mean))))
print("std finite :", bool(np.all(np.isfinite(std))))
print("std range  :", float(std.min()), "..", float(std.max()))
print("STATUS     : OK")
