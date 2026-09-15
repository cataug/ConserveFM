#!/usr/bin/env bash

cd "$HOME/ConserveFM"

PY="$HOME/Malashin_Projects/.venv_a100/bin/python"
DATA_PY="$HOME/ConserveFM/.venv_data/bin/python"
DATA_SITE=$("$DATA_PY" -c 'import site; print(site.getsitepackages()[0])')

STAMP="$(date +%Y%m%d_%H%M%S)"

OUTROOT="$HOME/ConserveFM/runs/iclr_extra_${STAMP}"
LOGROOT="$HOME/ConserveFM/logs/iclr_extra_${STAMP}"
REPORTROOT="$HOME/ConserveFM/reports/iclr_extra_${STAMP}"

mkdir -p "$OUTROOT" "$LOGROOT" "$REPORTROOT"

echo "$OUTROOT" > "$HOME/ConserveFM/reports/iclr_extra_latest_path.txt"
echo "$LOGROOT" > "$HOME/ConserveFM/reports/iclr_extra_latest_logs.txt"
echo "$REPORTROOT" > "$HOME/ConserveFM/reports/iclr_extra_latest_reports.txt"

# ------------------------------------------------------------------
# Existing seed-42 learned repair checkpoints.
# FourCastNet will use THESE SAME checkpoints zero-shot.
# ------------------------------------------------------------------

CK_DIRECT="$HOME/ConserveFM/runs/train_climax_direct_state_prediction_s42"
CK_ACC="$HOME/ConserveFM/runs/train_climax_no_corruption_pretrain_s42"
CK_PHYS="$HOME/ConserveFM/runs/train_climax_conservefm_smartfinal_s42"

for CK in "$CK_DIRECT" "$CK_ACC" "$CK_PHYS"
do
    if [ ! -f "$CK/checkpoint_best.pt" ]; then
        echo "MISSING CHECKPOINT: $CK/checkpoint_best.pt"
        exit 1
    fi
done

echo "============================================================"
echo "ICLR EXTRA EVALUATION"
echo "============================================================"
echo "OUTROOT : $OUTROOT"
echo "LOGROOT : $LOGROOT"
echo
echo "FourCastNet transfer is ZERO-SHOT:"
echo "repair weights remain ClimaX-trained."
echo "============================================================"

MAX_PAR=6


wait_for_slot () {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]
    do
        wait -n || true
    done
}


run_job () {
    LABEL="$1"
    BACKBONE="$2"
    METHOD="$3"
    CKDIR="$4"
    MODE="$5"

    OUTDIR="$OUTROOT/$LABEL"
    LOGFILE="$LOGROOT/${LABEL}.log"
    RCFILE="$REPORTROOT/${LABEL}.rc"

    mkdir -p "$OUTDIR"

    ARGS=(
        "$PY" -X faulthandler -u
        tools/eval_conservefm.py

        --run-id "$LABEL"
        --backbone "$BACKBONE"
        --family main
        --method "$METHOD"
        --seed 42

        --research-manifest
        "$HOME/ConserveFM/manifests/research_v1"

        --era5-zarr
        "$HOME/ConserveFM/data/ERA5_WeatherBench2_1979_2022"

        --forecast-root
        "$HOME/ConserveFM/forecasts"

        --stats-path
        "$HOME/ConserveFM/manifests/research_v1/era5_channel_stats.npz"

        --output-dir "$OUTDIR"

        --lead-hours 6,24,72

        --stress-types
        moisture,hydrostatic,advection,spectral,range_extreme

        --severities
        0.25,0.5,1.0,2.0
    )

    if [ -n "$CKDIR" ]; then
        ARGS+=(
            --checkpoint-dir "$CKDIR"
        )
    fi

    if [ "$MODE" = "transfer" ]; then
        ARGS+=(
            --eval-clean
            --eval-stress
        )

    elif [ "$MODE" = "transfer_ood" ]; then
        ARGS+=(
            --eval-clean
            --eval-stress
            --eval-ghcn
        )

    elif [ "$MODE" = "ood" ]; then
        ARGS+=(
            --eval-ghcn
        )

    else
        echo "UNKNOWN MODE: $MODE"
        echo 99 > "$RCFILE"
        return
    fi

    echo
    echo "============================================================"
    echo "START $LABEL"
    echo "backbone=$BACKBONE method=$METHOD mode=$MODE"
    echo "============================================================"

    CONSERVEFM_EVAL_BATCH_SIZE=32 \
    CONSERVEFM_STRESS_EXAMPLES=64 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$HOME/ConserveFM:$DATA_SITE" \
    stdbuf -oL -eL "${ARGS[@]}" \
      2>&1 \
      | sed -u "s/^/[$LABEL] /" \
      | tee "$LOGFILE"

    RC=${PIPESTATUS[0]}

    echo "$RC" > "$RCFILE"

    echo "[$LABEL] EXIT CODE: $RC"
}


launch () {
    wait_for_slot
    run_job "$@" &
}


# ==================================================================
# A. FOURCASTNET ZERO-SHOT TRANSFER
#
# These run:
#   clean ERA5 test
#   synthetic stress
#   GHCN OOD
# ==================================================================

launch \
    "fourcastnet_raw_s42" \
    "fourcastnet" \
    "raw_fm" \
    "" \
    "transfer_ood"

launch \
    "fourcastnet_hard_projection_s42" \
    "fourcastnet" \
    "hard_projection" \
    "" \
    "transfer_ood"

launch \
    "fourcastnet_direct_state_s42" \
    "fourcastnet" \
    "direct_state_prediction" \
    "$CK_DIRECT" \
    "transfer_ood"

launch \
    "fourcastnet_conservefm_acc_s42" \
    "fourcastnet" \
    "no_corruption_pretrain" \
    "$CK_ACC" \
    "transfer_ood"

launch \
    "fourcastnet_conservefm_phys_s42" \
    "fourcastnet" \
    "conservefm_full" \
    "$CK_PHYS" \
    "transfer_ood"


# ==================================================================
# B. CLIMAX EXTERNAL/OOD GHCN
#
# Same methods, same seed, GHCN only.
# ==================================================================

launch \
    "climax_raw_ghcn_s42" \
    "climax" \
    "raw_fm" \
    "" \
    "ood"

launch \
    "climax_hard_projection_ghcn_s42" \
    "climax" \
    "hard_projection" \
    "" \
    "ood"

launch \
    "climax_direct_state_ghcn_s42" \
    "climax" \
    "direct_state_prediction" \
    "$CK_DIRECT" \
    "ood"

launch \
    "climax_conservefm_acc_ghcn_s42" \
    "climax" \
    "no_corruption_pretrain" \
    "$CK_ACC" \
    "ood"

launch \
    "climax_conservefm_phys_ghcn_s42" \
    "climax" \
    "conservefm_full" \
    "$CK_PHYS" \
    "ood"


# ==================================================================
# WAIT FOR EVERYTHING
# ==================================================================

wait || true

echo
echo "============================================================"
echo "ALL ICLR EXTRA EVAL PROCESSES FINISHED"
echo "============================================================"

FAILED=0

for RCFILE in "$REPORTROOT"/*.rc
do
    [ -e "$RCFILE" ] || continue

    RC="$(cat "$RCFILE")"
    NAME="$(basename "$RCFILE" .rc)"

    printf "%-46s  exit=%s\n" "$NAME" "$RC"

    if [ "$RC" != "0" ]; then
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "failed jobs: $FAILED"
echo "results    : $OUTROOT"
echo "logs       : $LOGROOT"
echo "status     : $REPORTROOT"

if [ "$FAILED" -ne 0 ]; then
    echo
    echo "Some evaluations failed; successful ones were kept."
fi
