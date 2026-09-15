#!/usr/bin/env bash

cd "$HOME/ConserveFM"

PY="$HOME/Malashin_Projects/.venv_a100/bin/python"

DATA_PY="$HOME/ConserveFM/.venv_data/bin/python"
DATA_SITE=$("$DATA_PY" -c 'import site; print(site.getsitepackages()[0])')

JOBS="$HOME/ConserveFM/reports/prelim20_salvage/jobs.tsv"

TRAINLOG="$HOME/ConserveFM/logs/prelim20_train"
EVALLOG="$HOME/ConserveFM/logs/prelim20_eval"
EVALROOT="$HOME/ConserveFM/runs/prelim20_eval"

mkdir -p \
  "$TRAINLOG" \
  "$EVALLOG" \
  "$EVALROOT"

# A100 40 GB.
# Models are small, but ERA5/Zarr I/O becomes the limiting factor.
MAX_TRAIN_PAR=6
MAX_EVAL_PAR=10


wait_slot () {
    local max="$1"

    while [ "$(jobs -rp | wc -l)" -ge "$max" ]
    do
        wait -n || true
    done
}


manifest_train () {
    local backbone="$1"

    if [ "$backbone" = "prithvi_wxc" ]; then
        echo "$HOME/ConserveFM/manifests/research_prithvi_transfer_v1"
    else
        echo "$HOME/ConserveFM/manifests/research_smart12k_v1"
    fi
}


manifest_eval () {
    local backbone="$1"

    if [ "$backbone" = "prithvi_wxc" ]; then
        echo "$HOME/ConserveFM/manifests/research_prithvi_transfer_v1"
    else
        echo "$HOME/ConserveFM/manifests/research_v1"
    fi
}


run_budget_train () {

    local RUN_ID="$1"
    local BACKBONE="$2"
    local FAMILY="$3"
    local METHOD="$4"
    local SEED="$5"
    local CKDIR="$6"

    local MANIFEST
    MANIFEST="$(manifest_train "$BACKBONE")"

    echo "[START TRAIN] $RUN_ID"

    # 6000 examples:
    # batch=2 -> 3000 batches/stage.
    #
    # Most corruption-pretrain variants:
    #   1 pretrain + 1 finetune ≈ 6000 batches.
    #
    # No-pretrain:
    #   only finetune ≈ 3000 batches.
    #
    # On the observed server rate this remains around the intended
    # ~10–20 min low-budget regime.
    CONSERVEFM_PRETRAIN_EPOCHS=1 \
    CONSERVEFM_FINETUNE_EPOCHS=1 \
    CONSERVEFM_RESET_OPTIMIZER_ON_RESUME=1 \
    CONSERVEFM_AUX_EVERY=4 \
    CONSERVEFM_AUX_WEIGHT=0.5 \
    CONSERVEFM_MAX_BATCHES_PER_EPOCH=0 \
    CONSERVEFM_NUM_WORKERS=2 \
    CONSERVEFM_BATCH=2 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$HOME/ConserveFM:$DATA_SITE" \
    "$PY" -X faulthandler -u \
      tools/train_conservefm.py \
      --run-id "prelim20_${RUN_ID}" \
      --backbone "$BACKBONE" \
      --family "$FAMILY" \
      --method "$METHOD" \
      --seed "$SEED" \
      --research-manifest "$MANIFEST" \
      --era5-zarr \
        "$HOME/ConserveFM/data/ERA5_WeatherBench2_1979_2022" \
      --forecast-root \
        "$HOME/ConserveFM/forecasts" \
      --stats-path \
        "$HOME/ConserveFM/manifests/research_v1/era5_channel_stats.npz" \
      --output-dir "$CKDIR" \
      --max-train-examples 6000 \
      --max-val-examples 128 \
      2>&1 \
      | sed -u "s/^/[TRAIN $RUN_ID] /" \
      | tee "$TRAINLOG/${RUN_ID}.log"

    RC=${PIPESTATUS[0]}

    echo "[DONE TRAIN] $RUN_ID exit=$RC"
}


run_screen_eval () {

    local RUN_ID="$1"
    local BACKBONE="$2"
    local FAMILY="$3"
    local METHOD="$4"
    local SEED="$5"
    local CKDIR="$6"

    local MANIFEST
    MANIFEST="$(manifest_eval "$BACKBONE")"

    local OUT="$EVALROOT/$RUN_ID"

    mkdir -p "$OUT"

    echo "[START EVAL] $RUN_ID"

    CONSERVEFM_EVAL_BATCH_SIZE=32 \
    CONSERVEFM_STRESS_EXAMPLES=16 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$HOME/ConserveFM:$DATA_SITE" \
    "$PY" -X faulthandler -u \
      tools/eval_conservefm.py \
      --run-id "prelim20_eval_${RUN_ID}" \
      --backbone "$BACKBONE" \
      --family "$FAMILY" \
      --method "$METHOD" \
      --seed "$SEED" \
      --research-manifest "$MANIFEST" \
      --era5-zarr \
        "$HOME/ConserveFM/data/ERA5_WeatherBench2_1979_2022" \
      --forecast-root \
        "$HOME/ConserveFM/forecasts" \
      --stats-path \
        "$HOME/ConserveFM/manifests/research_v1/era5_channel_stats.npz" \
      --output-dir "$OUT" \
      --checkpoint-dir "$CKDIR" \
      --lead-hours 6,24,72 \
      --stress-types \
        moisture,hydrostatic,advection,spectral,range_extreme \
      --severities \
        0.25,0.5,1.0,2.0 \
      --eval-stress \
      2>&1 \
      | sed -u "s/^/[EVAL $RUN_ID] /" \
      | tee "$EVALLOG/${RUN_ID}.log"

    RC=${PIPESTATUS[0]}

    echo "[DONE EVAL] $RUN_ID exit=$RC"
}


# ============================================================
# PHASE A — TRAIN ONLY CONFIGURATIONS WITH NO USABLE CHECKPOINT
# ============================================================

echo
echo "======================================================================"
echo "PHASE A — LOW-BUDGET TRAINING OF MISSING CHECKPOINTS"
echo "======================================================================"

tail -n +2 "$JOBS" |
while IFS=$'\t' read -r \
    RUN_ID BACKBONE FAMILY METHOD SEED CKDIR NEEDS_TRAIN \
    SOURCE_KIND SOURCE_CHECKPOINT PRIOR_STEP PRIOR_VAL PRIOR_ROWS
do

    [ "$NEEDS_TRAIN" = "1" ] || continue

    wait_slot "$MAX_TRAIN_PAR"

    run_budget_train \
        "$RUN_ID" \
        "$BACKBONE" \
        "$FAMILY" \
        "$METHOD" \
        "$SEED" \
        "$CKDIR" &
done

wait || true


# ============================================================
# PHASE B — SAME CHEAP STRESS SCREEN FOR ALL 66 LEARNED RUNS
# ============================================================

echo
echo "======================================================================"
echo "PHASE B — UNIFORM STRESS SCREEN OF ALL LEARNED RUNS"
echo "======================================================================"

tail -n +2 "$JOBS" |
while IFS=$'\t' read -r \
    RUN_ID BACKBONE FAMILY METHOD SEED CKDIR NEEDS_TRAIN \
    SOURCE_KIND SOURCE_CHECKPOINT PRIOR_STEP PRIOR_VAL PRIOR_ROWS
do

    if [ ! -f "$CKDIR/checkpoint_best.pt" ]; then

        if [ -f "$CKDIR/checkpoint_last.pt" ]; then
            ln -sf \
              "$CKDIR/checkpoint_last.pt" \
              "$CKDIR/checkpoint_best.pt"
        else
            echo "[SKIP EVAL] no checkpoint: $RUN_ID"
            continue
        fi
    fi

    wait_slot "$MAX_EVAL_PAR"

    run_screen_eval \
        "$RUN_ID" \
        "$BACKBONE" \
        "$FAMILY" \
        "$METHOD" \
        "$SEED" \
        "$CKDIR" &
done

wait || true


# ============================================================
# PHASE C — RAW FM + HARD PROJECTION FOR EACH BACKBONE
# Same 16-example stress subset.
# ============================================================

echo
echo "======================================================================"
echo "PHASE C — DETERMINISTIC BASELINES"
echo "======================================================================"

for BACKBONE in climax fourcastnet prithvi_wxc
do
    MANIFEST="$(manifest_eval "$BACKBONE")"

    for METHOD in raw_fm hard_projection
    do
        RUN_ID="det_${BACKBONE}_${METHOD}"

        OUT="$EVALROOT/$RUN_ID"

        mkdir -p "$OUT"

        wait_slot "$MAX_EVAL_PAR"

        (
            echo "[START DET] $RUN_ID"

            CONSERVEFM_EVAL_BATCH_SIZE=32 \
            CONSERVEFM_STRESS_EXAMPLES=16 \
            CUDA_VISIBLE_DEVICES=0 \
            PYTHONPATH="$HOME/ConserveFM:$DATA_SITE" \
            "$PY" -X faulthandler -u \
              tools/eval_conservefm.py \
              --run-id "prelim20_eval_${RUN_ID}" \
              --backbone "$BACKBONE" \
              --family deterministic \
              --method "$METHOD" \
              --seed 42 \
              --research-manifest "$MANIFEST" \
              --era5-zarr \
                "$HOME/ConserveFM/data/ERA5_WeatherBench2_1979_2022" \
              --forecast-root \
                "$HOME/ConserveFM/forecasts" \
              --stats-path \
                "$HOME/ConserveFM/manifests/research_v1/era5_channel_stats.npz" \
              --output-dir "$OUT" \
              --lead-hours 6,24,72 \
              --stress-types \
                moisture,hydrostatic,advection,spectral,range_extreme \
              --severities \
                0.25,0.5,1.0,2.0 \
              --eval-stress \
              2>&1 \
              | sed -u "s/^/[DET $RUN_ID] /" \
              | tee "$EVALLOG/${RUN_ID}.log"

            RC=${PIPESTATUS[0]}

            echo "[DONE DET] $RUN_ID exit=$RC"
        ) &
    done
done

wait || true


echo
echo "======================================================================"
echo "PRELIMINARY SALVAGE CAMPAIGN FINISHED"
echo "======================================================================"
echo "Evaluations:"
echo "$EVALROOT"
