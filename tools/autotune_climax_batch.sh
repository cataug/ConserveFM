#!/usr/bin/env bash

set +e

cd "$HOME/ConserveFM" || exit 1

PY="$HOME/Malashin_Projects/.venv_a100/bin/python"
DATA_PY="$HOME/ConserveFM/.venv_data/bin/python"

DATA_SITE=$("$DATA_PY" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$HOME/ConserveFM:$HOME/ConserveFM/external/ClimaX/src:$DATA_SITE"

mkdir -p logs/batch_tune models/batch_tune

BEST=0

# достаточно плотная лестница около ожидаемого предела
for B in 8 12 16 20 24 28 32 40 48 56 64; do

    echo
    echo "============================================================"
    echo "TEST BATCH = $B"
    echo "============================================================"

    OUT="models/batch_tune/b${B}"
    LOG="logs/batch_tune/b${B}.log"

    rm -rf "$OUT"

    MAX_TRAIN=$((B * 4))

    "$PY" -X faulthandler -u \
      tools/train_climax_forecaster.py \
      --output-dir "$OUT" \
      --epochs 1 \
      --max-train "$MAX_TRAIN" \
      --max-val "$B" \
      --batch-size "$B" \
      --accum 1 \
      --num-workers 8 \
      --prefetch-factor 1 \
      --lr 5e-7 \
      --log-every 1 \
      2>&1 | tee "$LOG"

    CODE=${PIPESTATUS[0]}

    if [ "$CODE" -eq 0 ]; then
        BEST=$B

        PEAK=$(grep -oE 'GPU_peak=[0-9.]+GiB' "$LOG" \
          | sed 's/GPU_peak=//;s/GiB//' \
          | sort -nr \
          | head -1)

        echo
        echo "[PASS] batch=$B peak=${PEAK:-unknown} GiB"

        # Если уже почти заполнили A100, дальше незачем рисковать
        if [ -n "$PEAK" ]; then
            STOP=$(awk -v x="$PEAK" 'BEGIN { print (x >= 34.0) ? 1 : 0 }')
            if [ "$STOP" -eq 1 ]; then
                echo "[STOP] reached >=34 GiB peak"
                break
            fi
        fi

    else
        echo
        echo "[FAIL] batch=$B exit=$CODE"

        if grep -qiE \
          'out of memory|CUDA error.*memory|CUDA out of memory' \
          "$LOG"; then
            echo "[OOM] stopping batch search"
            break
        fi

        echo "[ERROR] not an OOM, inspect $LOG"
        exit "$CODE"
    fi

done

echo
echo "============================================================"
echo "AUTOTUNE COMPLETE"
echo "BEST BATCH: $BEST"
echo "============================================================"

echo "$BEST" > models/batch_tune/BEST_BATCH.txt
