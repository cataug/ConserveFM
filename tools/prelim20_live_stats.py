from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd

ROOT = Path.home() / "ConserveFM"

META_P = ROOT / "reports/prelim20_salvage/jobs.csv"
TRAINROOT = ROOT / "runs/prelim20_train"
EVALROOT = ROOT / "runs/prelim20_eval"
OUT = ROOT / "reports/prelim20_salvage"

OUT.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(META_P)

# ============================================================
# PROCESS SNAPSHOT
# ============================================================

try:
    ps = subprocess.check_output(
        ["pgrep", "-af", "train_conservefm.py|eval_conservefm.py"],
        text=True,
    )
except subprocess.CalledProcessError:
    ps = ""

running_train_ids = set()
running_eval_ids = set()

for line in ps.splitlines():
    for run_id in meta["run_id"].astype(str):
        if run_id in line:
            if "train_conservefm.py" in line:
                running_train_ids.add(run_id)
            if "eval_conservefm.py" in line:
                running_eval_ids.add(run_id)

# ============================================================
# TRAINING / CHECKPOINT STATUS
# ============================================================

train_rows = []

for _, r in meta.iterrows():

    run_id = str(r["run_id"])
    ckdir = Path(r["checkpoint_dir"])

    hist = ckdir / "train_history.csv"
    best = ckdir / "checkpoint_best.pt"
    last = ckdir / "checkpoint_last.pt"

    status = "pending"

    if int(r["needs_train"]) == 0:
        status = "reused"
    elif run_id in running_train_ids:
        status = "running"
    elif best.exists() or last.exists():
        status = "completed"

    best_val = np.nan
    last_val = np.nan
    global_step = np.nan
    elapsed = np.nan
    epochs = 0

    if hist.exists():
        try:
            h = pd.read_csv(hist)

            if "val_repair_mse" in h.columns:
                vals = pd.to_numeric(
                    h["val_repair_mse"],
                    errors="coerce",
                )
                best_val = vals.min()
                last_val = vals.iloc[-1]

            if "global_step" in h.columns:
                global_step = pd.to_numeric(
                    h["global_step"],
                    errors="coerce",
                ).max()

            if "elapsed_seconds" in h.columns:
                elapsed = pd.to_numeric(
                    h["elapsed_seconds"],
                    errors="coerce",
                ).max()

            epochs = len(h)

        except Exception:
            pass

    # fallback to original-run metadata for reused jobs
    if status == "reused":
        if pd.isna(global_step):
            global_step = r.get(
                "prior_global_step",
                np.nan,
            )

        if pd.isna(best_val):
            best_val = r.get(
                "prior_best_val",
                np.nan,
            )

    train_rows.append({
        "run_id": run_id,
        "backbone": r["backbone"],
        "method": r["method"],
        "seed": r["seed"],
        "source_kind": r["source_kind"],
        "status": status,
        "global_step": global_step,
        "best_val": best_val,
        "last_val": last_val,
        "elapsed_seconds": elapsed,
        "history_rows": epochs,
        "checkpoint_best": best.exists(),
        "checkpoint_last": last.exists(),
    })

train = pd.DataFrame(train_rows)

train.to_csv(
    OUT / "live_train_status.csv",
    index=False,
)

# ============================================================
# EVAL STATUS + METRICS
# ============================================================

def mean_metric(d, key):
    vals = []

    for v in d.values():
        if isinstance(v, dict) and key in v:
            try:
                vals.append(float(v[key]))
            except Exception:
                pass

    return float(np.mean(vals)) if vals else np.nan


eval_rows = []

for _, r in meta.iterrows():

    run_id = str(r["run_id"])
    d = EVALROOT / run_id
    p = d / "metrics_stress.json"

    if p.exists():
        status = "completed"

        try:
            m = json.loads(p.read_text())

            rmse = mean_metric(m, "rmse_phys")
            mae = mean_metric(m, "mae_phys")
            loc = mean_metric(m, "localization_iou")
            typ = mean_metric(m, "type_accuracy")
            repair = mean_metric(m, "repair_l1_norm")

            phys_m = mean_metric(m, "physics_moisture")
            phys_h = mean_metric(m, "physics_hydrostatic")
            phys_a = mean_metric(m, "physics_advection")
            phys_s = mean_metric(m, "physics_spectral")

        except Exception:
            status = "broken"
            rmse = mae = loc = typ = repair = np.nan
            phys_m = phys_h = phys_a = phys_s = np.nan

    elif run_id in running_eval_ids:
        status = "running"
        rmse = mae = loc = typ = repair = np.nan
        phys_m = phys_h = phys_a = phys_s = np.nan

    else:
        status = "pending"
        rmse = mae = loc = typ = repair = np.nan
        phys_m = phys_h = phys_a = phys_s = np.nan

    eval_rows.append({
        "run_id": run_id,
        "backbone": r["backbone"],
        "method": r["method"],
        "seed": r["seed"],
        "status": status,
        "stress_rmse_phys_mixed": rmse,
        "stress_mae_phys_mixed": mae,
        "loc_iou": loc,
        "type_acc": typ,
        "repair_l1": repair,
        "physics_moisture": phys_m,
        "physics_hydrostatic": phys_h,
        "physics_advection": phys_a,
        "physics_spectral": phys_s,
    })

ev = pd.DataFrame(eval_rows)

ev.to_csv(
    OUT / "live_eval_status.csv",
    index=False,
)

# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 100)
print("PRELIM20 — LIVE CAMPAIGN STATUS")
print("=" * 100)

print("\nTRAIN/CHECKPOINT STATUS")
print(
    train["status"]
    .value_counts()
    .reindex(
        ["reused", "completed", "running", "pending"],
        fill_value=0,
    )
    .to_string()
)

print("\nBY BACKBONE")
print(
    pd.crosstab(
        train["backbone"],
        train["status"],
    ).to_string()
)

budget_done = train[
    train["status"] == "completed"
].copy()

if not budget_done.empty:

    sec = pd.to_numeric(
        budget_done["elapsed_seconds"],
        errors="coerce",
    ).dropna()

    print("\nBUDGET-TRAIN RUNTIME")

    if len(sec):
        print(
            f"completed jobs : {len(sec)}\n"
            f"mean          : {sec.mean()/60:.1f} min\n"
            f"median        : {sec.median()/60:.1f} min\n"
            f"min           : {sec.min()/60:.1f} min\n"
            f"max           : {sec.max()/60:.1f} min\n"
            f"total GPU-wall: {sec.sum()/3600:.2f} h"
        )

    print("\nLATEST COMPLETED BUDGET RUNS")

    cols = [
        "run_id",
        "backbone",
        "method",
        "seed",
        "global_step",
        "best_val",
        "elapsed_seconds",
    ]

    x = budget_done[cols].copy()

    x["elapsed_min"] = (
        x["elapsed_seconds"] / 60
    )

    print(
        x.drop(columns="elapsed_seconds")
        .sort_values(
            "elapsed_min",
            ascending=False,
        )
        .head(20)
        .to_string(
            index=False,
            float_format=lambda z: f"{z:.4f}",
        )
    )

print("\nEVAL STATUS")
print(
    ev["status"]
    .value_counts()
    .reindex(
        ["completed", "running", "pending", "broken"],
        fill_value=0,
    )
    .to_string()
)

done_eval = ev[
    ev["status"] == "completed"
].copy()

if not done_eval.empty:

    print("\nPRELIMINARY RANKING — AVAILABLE EVALS")
    print(
        done_eval[
            [
                "backbone",
                "method",
                "seed",
                "stress_rmse_phys_mixed",
                "stress_mae_phys_mixed",
                "loc_iou",
                "type_acc",
            ]
        ]
        .sort_values(
            [
                "backbone",
                "stress_rmse_phys_mixed",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda z: f"{z:.4f}",
        )
    )

    grouped = (
        done_eval
        .groupby(
            ["backbone", "method"]
        )
        .agg(
            n=("run_id", "count"),
            stress_rmse=(
                "stress_rmse_phys_mixed",
                "mean",
            ),
            stress_rmse_std=(
                "stress_rmse_phys_mixed",
                "std",
            ),
            stress_mae=(
                "stress_mae_phys_mixed",
                "mean",
            ),
            loc_iou=("loc_iou", "mean"),
            type_acc=("type_acc", "mean"),
        )
        .reset_index()
        .sort_values(
            ["backbone", "stress_rmse"]
        )
    )

    grouped.to_csv(
        OUT / "live_method_ranking.csv",
        index=False,
    )

    print("\nGROUPED METHOD RANKING")
    print(
        grouped.to_string(
            index=False,
            float_format=lambda z: f"{z:.4f}",
        )
    )

print("\nRUNNING TRAIN JOBS")
if running_train_ids:
    for x in sorted(running_train_ids):
        print("  ", x)
else:
    print("  none")

print("\nRUNNING EVAL JOBS")
if running_eval_ids:
    for x in sorted(running_eval_ids):
        print("  ", x)
else:
    print("  none")

print("\nSaved:")
print(OUT / "live_train_status.csv")
print(OUT / "live_eval_status.csv")
