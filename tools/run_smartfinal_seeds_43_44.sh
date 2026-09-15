#!/usr/bin/env bash

cd "$HOME/ConserveFM"

PY="$HOME/Malashin_Projects/.venv_a100/bin/python"
DATA_PY="$HOME/ConserveFM/.venv_data/bin/python"
DATA_SITE=$("$DATA_PY" -c 'import site; print(site.getsitepackages()[0])')

# ------------------------------------------------------------
# PRECHECK + initialize each Smart-Final from its SAME-SEED
# fully trained No-Pretrain checkpoint.
# ------------------------------------------------------------
"$PY" - <<'PY'
from pathlib import Path
import shutil
import torch
import pandas as pd

ROOT = Path.home() / "ConserveFM"

for seed in (43, 44):
    srcdir = ROOT / f"runs/train_climax_no_corruption_pretrain_s{seed}"
    src = srcdir / "checkpoint_best.pt"
    hist = srcdir / "train_history.csv"

    if not src.exists():
        raise SystemExit(
            f"\nSTOP: missing {src}\n"
            f"Need the SAME-SEED No_Pretrain source for seed {seed}."
        )

    if not hist.exists():
        raise SystemExit(
            f"\nSTOP: missing {hist}\n"
            f"Cannot verify No_Pretrain seed {seed} completion."
        )

    h = pd.read_csv(hist)

    hf = h[h["stage"] == "finetune"]

    print(f"\n===== No_Pretrain s{seed} =====")
    print(
        hf[["epoch","global_step","val_repair_mse"]]
        .tail(10)
        .to_string(index=False)
    )

    # No-pretrain protocol = 6 finetune epochs: epoch 0..5.
    completed = sorted(
        set(hf["epoch"].astype(int).tolist())
    )

    if not completed or max(completed) < 5:
        raise SystemExit(
            f"\nSTOP: No_Pretrain s{seed} is NOT fully trained.\n"
            f"Completed finetune epochs: {completed}\n"
            f"Need epoch 5 before creating an independent Smart-Final seed."
        )

    dst = ROOT / f"runs/train_climax_conservefm_smartfinal_s{seed}"

    if dst.exists():
        shutil.rmtree(dst)

    dst.mkdir(parents=True)

    ck = torch.load(
        src,
        map_location="cpu",
        weights_only=False,
    )

    # Source weights are initialization only.
    # FINETUNE_EPOCHS=2 + epoch=0 complete=True => exactly one new epoch.
    ck["stage"] = "finetune"
    ck["epoch"] = 0
    ck["epoch_complete"] = True
    ck["next_batch_in_epoch"] = 0
    ck["global_step"] = 0
    ck["best_val"] = float("inf")
    ck["checkpoint_format"] = 2

    torch.save(
        ck,
        dst / "checkpoint_last.pt",
    )

    print(
        f"READY Smart-Final s{seed}: "
        f"{dst / 'checkpoint_last.pt'}"
    )

print("\nPRECHECK PASSED FOR BOTH SEEDS")
PY

if [ $? -ne 0 ]; then
    echo "PRECHECK FAILED — nothing was trained."
    exit 1
fi

# ------------------------------------------------------------
# Train s43 then s44.
# Same protocol as Smart-Final s42.
# ------------------------------------------------------------
for SEED in 43 44
do
    echo
    echo "============================================================"
    echo "SMART-FINAL SEED $SEED"
    echo "============================================================"

    CONSERVEFM_PRETRAIN_EPOCHS=0 \
    CONSERVEFM_FINETUNE_EPOCHS=2 \
    CONSERVEFM_RESET_OPTIMIZER_ON_RESUME=1 \
    CONSERVEFM_AUX_EVERY=4 \
    CONSERVEFM_AUX_WEIGHT=0.5 \
    CONSERVEFM_MAX_BATCHES_PER_EPOCH=0 \
    CONSERVEFM_NUM_WORKERS=8 \
    CONSERVEFM_BATCH=2 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$HOME/ConserveFM:$DATA_SITE" \
    "$PY" -X faulthandler -u \
      tools/train_conservefm.py \
      --run-id "train_climax_conservefm_smartfinal_s${SEED}" \
      --backbone climax \
      --family main \
      --method conservefm_full \
      --seed "$SEED" \
      --research-manifest "$HOME/ConserveFM/manifests/research_smart12k_v1" \
      --era5-zarr "$HOME/ConserveFM/data/ERA5_WeatherBench2_1979_2022" \
      --forecast-root "$HOME/ConserveFM/forecasts" \
      --stats-path "$HOME/ConserveFM/manifests/research_v1/era5_channel_stats.npz" \
      --output-dir "$HOME/ConserveFM/runs/train_climax_conservefm_smartfinal_s${SEED}" \
      2>&1 | tee "logs/conservefm_smartfinal_s${SEED}.log"

    RC=${PIPESTATUS[0]}

    echo "SEED $SEED EXIT CODE: $RC"

    if [ "$RC" -ne 0 ]; then
        echo "Stopping: seed $SEED failed."
        exit "$RC"
    fi
done

echo
echo "============================================================"
echo "SMART-FINAL s43 + s44 COMPLETE"
echo "============================================================"
