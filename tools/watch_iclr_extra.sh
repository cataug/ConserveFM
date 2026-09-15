#!/usr/bin/env bash

cd "$HOME/ConserveFM"

LOGROOT="$(cat reports/iclr_extra_latest_logs.txt)"
REPORTROOT="$(cat reports/iclr_extra_latest_reports.txt)"

JOBS=(
  fourcastnet_raw_s42
  fourcastnet_hard_projection_s42
  fourcastnet_direct_state_s42
  fourcastnet_conservefm_acc_s42
  fourcastnet_conservefm_phys_s42
  climax_raw_ghcn_s42
  climax_hard_projection_ghcn_s42
  climax_direct_state_ghcn_s42
  climax_conservefm_acc_ghcn_s42
  climax_conservefm_phys_ghcn_s42
)

bar () {
    local pct="$1"
    local width=24

    local fill=$(( pct * width / 100 ))
    local empty=$(( width - fill ))

    printf "["
    printf "%${fill}s" "" | tr ' ' '█'
    printf "%${empty}s" "" | tr ' ' '░'
    printf "]"
}

spinner () {
    local n="$1"
    local width=24
    local pos=$(( n % width ))

    printf "["

    for ((i=0; i<width; i++)); do
        if [ "$i" -eq "$pos" ]; then
            printf "▓"
        else
            printf "░"
        fi
    done

    printf "]"
}

find_pid () {
    local job="$1"

    pgrep -af "tools/eval_conservefm.py" \
      | grep -- "--run-id $job" \
      | awk 'NR==1 {print $1}'
}

progress_from_log () {
    local logfile="$1"

    [ -f "$logfile" ] || return 1

    # ----------------------------------------------------------
    # 1. Explicit percentage: 63%, 63.5%, progress=63%, etc.
    # ----------------------------------------------------------
    local pct
    pct="$(
      tail -n 500 "$logfile" \
      | grep -Eo '([0-9]{1,3}([.][0-9]+)?)%' \
      | tail -n 1 \
      | tr -d '%'
    )"

    if [ -n "$pct" ]; then
        awk -v x="$pct" 'BEGIN {
            if (x < 0) x=0;
            if (x > 99) x=99;
            printf "%d", x
        }'
        return 0
    fi

    # ----------------------------------------------------------
    # 2. Generic counters such as 1234/9182.
    # Take the LAST plausible counter.
    # ----------------------------------------------------------
    local pair
    pair="$(
      tail -n 800 "$logfile" \
      | grep -Eo '[0-9]+/[0-9]+' \
      | tail -n 1
    )"

    if [ -n "$pair" ]; then
        local cur="${pair%/*}"
        local tot="${pair#*/}"

        if [ "$tot" -gt 0 ] 2>/dev/null; then
            awk -v a="$cur" -v b="$tot" 'BEGIN {
                x = 100*a/b;
                if (x < 0) x=0;
                if (x > 99) x=99;
                printf "%d", x
            }'
            return 0
        fi
    fi

    return 1
}

last_phase () {
    local logfile="$1"

    [ -f "$logfile" ] || {
        echo "waiting"
        return
    }

    local line
    line="$(
      tail -n 300 "$logfile" \
      | grep -E '\[(GHCN|CLEAN|STRESS|EVAL|FORECAST)\]' \
      | tail -n 1
    )"

    if [ -z "$line" ]; then
        echo "running"
        return
    fi

    # Strip our [job-name] prefix.
    line="$(echo "$line" | sed -E 's/^\[[^]]+\][[:space:]]*//')"

    # Keep dashboard compact.
    echo "$line" | cut -c1-58
}

tick=0

while true; do

    printf "\033[2J\033[H"

    echo "ICLR EXTRA EVAL — LIVE PROGRESS"
    echo "=============================================================================================="
    printf "%-38s %-26s %-8s %-8s %s\n" \
      "JOB" "PROGRESS" "STATE" "PID" "PHASE"
    echo "----------------------------------------------------------------------------------------------"

    finished=0
    failed=0
    running=0

    for job in "${JOBS[@]}"; do

        rcfile="$REPORTROOT/$job.rc"
        logfile="$LOGROOT/$job.log"

        if [ -f "$rcfile" ]; then

            rc="$(cat "$rcfile")"

            if [ "$rc" = "0" ]; then
                finished=$((finished + 1))

                printf "%-38s " "$job"
                bar 100
                printf "  %-8s %-8s %s\n" \
                  "DONE" "-" "complete"
            else
                failed=$((failed + 1))

                printf "%-38s " "$job"
                bar 100
                printf "  %-8s %-8s %s\n" \
                  "FAILED" "-" "exit=$rc"
            fi

            continue
        fi

        pid="$(find_pid "$job")"

        if [ -n "$pid" ]; then
            running=$((running + 1))

            pct="$(progress_from_log "$logfile")"

            printf "%-38s " "$job"

            if [ -n "$pct" ]; then
                bar "$pct"
                printf " %3s%% " "$pct"
            else
                spinner "$tick"
                printf "  ... "
            fi

            printf "%-8s %-8s %s\n" \
              "RUNNING" \
              "$pid" \
              "$(last_phase "$logfile")"

        else
            printf "%-38s " "$job"
            bar 0
            printf "  %-8s %-8s %s\n" \
              "WAIT" "-" "queued / starting"
        fi
    done

    echo "----------------------------------------------------------------------------------------------"
    printf "DONE %d/10   RUNNING %d   FAILED %d\n" \
      "$finished" "$running" "$failed"

    if [ "$finished" -eq 10 ]; then
        echo
        echo "ALL JOBS COMPLETE."
        break
    fi

    if [ $((finished + failed)) -eq 10 ]; then
        echo
        echo "ALL JOBS FINISHED; failures: $failed"
        break
    fi

    tick=$((tick + 1))
    sleep 3
done
