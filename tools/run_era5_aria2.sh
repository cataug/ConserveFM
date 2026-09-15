#!/usr/bin/env bash

ROOT="$HOME/ConserveFM"
PY="$ROOT/.venv_data/bin/python"

DATASET="$ROOT/data/ERA5_WeatherBench2_1979_2022"
MANIFEST="$ROOT/manifests/era5_remaining_aria2.txt"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/era5_aria2_${STAMP}.log"
SESSION="$ROOT/manifests/era5_aria2_session.txt"

CONNECTIONS="${ERA5_CONNECTIONS:-32}"

mkdir -p "$ROOT/logs"
mkdir -p "$ROOT/manifests"

echo "============================================================"
echo "ERA5 ARIA2 DOWNLOADER"
echo "Dataset:     $DATASET"
echo "Connections: $CONNECTIONS"
echo "Log:         $LOG"
echo "============================================================"
echo

echo "Rebuilding manifest and skipping completed chunks..."

"$PY" "$ROOT/tools/build_era5_aria2_manifest.py"
BUILD_CODE=$?

if [ "$BUILD_CODE" -ne 0 ]; then
    echo
    echo "[ERROR] Manifest creation failed with code $BUILD_CODE"
else
    REMAINING="$(grep -c '^https://' "$MANIFEST" 2>/dev/null)"

    echo
    echo "Objects remaining: $REMAINING"
    echo

    if [ "$REMAINING" -eq 0 ]; then
        echo "Nothing remains to download."
    else
        aria2c \
          --input-file="$MANIFEST" \
          --max-concurrent-downloads="$CONNECTIONS" \
          --max-connection-per-server=1 \
          --split=1 \
          --continue=true \
          --auto-file-renaming=false \
          --allow-overwrite=true \
          --file-allocation=none \
          --connect-timeout=20 \
          --timeout=120 \
          --lowest-speed-limit=0 \
          --max-tries=0 \
          --retry-wait=3 \
          --summary-interval=30 \
          --console-log-level=notice \
          --show-console-readout=true \
          --download-result=full \
          --save-session="$SESSION" \
          --save-session-interval=60 \
          --force-save=true \
          > >(tee -a "$LOG") 2>&1 &

        ARIA_PID=$!

        echo "aria2 PID: $ARIA_PID"
        echo "Ctrl+C stops aria2 but keeps the SSH shell open."
        echo

        STOP_REQUESTED=0

        trap '
            STOP_REQUESTED=1
            echo
            echo "Stopping aria2 PID '"$ARIA_PID"'..."
            kill -TERM '"$ARIA_PID"' 2>/dev/null
        ' INT TERM

        PREVIOUS_SIZE=0

        while kill -0 "$ARIA_PID" 2>/dev/null; do
            CURRENT_BYTES="$(
                du -sb "$DATASET" 2>/dev/null \
                | awk "{print \$1}"
            )"

            CURRENT_BYTES="${CURRENT_BYTES:-0}"

            if [ "$PREVIOUS_SIZE" -gt 0 ]; then
                ADDED=$((CURRENT_BYTES - PREVIOUS_SIZE))
            else
                ADDED=0
            fi

            COMPLETE_FILES="$(
                find "$DATASET" \
                  -type f \
                  ! -name '*.aria2' \
                  2>/dev/null \
                | wc -l
            )"

            ACTIVE_FILES="$(
                find "$DATASET" \
                  -type f \
                  -name '*.aria2' \
                  2>/dev/null \
                | wc -l
            )"

            echo
            echo "------------------------------------------------------------"
            date
            echo "Local size:"
            du -sh "$DATASET" 2>/dev/null

            echo "Added during previous minute:"
            numfmt \
              --to=iec-i \
              --suffix=B \
              "$ADDED" \
              2>/dev/null

            echo "Completed local files: $COMPLETE_FILES"
            echo "Active partial downloads: $ACTIVE_FILES"

            echo "Free disk:"
            df -h "$DATASET" | tail -n 1

            echo "aria2 process:"
            ps -p "$ARIA_PID" \
              -o pid,etime,stat,%cpu,%mem,cmd

            echo "Recent aria2 output:"
            tail -n 10 "$LOG" 2>/dev/null

            PREVIOUS_SIZE="$CURRENT_BYTES"

            sleep 60
        done

        wait "$ARIA_PID"
        ARIA_CODE=$?

        echo
        echo "============================================================"
        echo "aria2 finished with code: $ARIA_CODE"
        echo "Local size:"
        du -sh "$DATASET" 2>/dev/null
        echo "Log: $LOG"
        echo "============================================================"

        if [ "$STOP_REQUESTED" -eq 1 ]; then
            echo "Stopped by user. Restarting the same script will resume."
        else
            echo
            echo "Checking remaining objects..."

            "$PY" "$ROOT/tools/build_era5_aria2_manifest.py"

            REMAINING_AFTER="$(
                grep -c '^https://' "$MANIFEST" 2>/dev/null
            )"

            echo "Remaining after run: $REMAINING_AFTER"
        fi
    fi
fi
