from __future__ import annotations

from pathlib import Path
import re
import math
import json
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, to_rgba

ROOT = Path.home() / "ConserveFM"
PLAN_CSV = ROOT / "manifests/experiments_v1/planned_runs.csv"
RUNS_DIR = ROOT / "runs"
FINAL_DIR = ROOT / "reports/final_metrics_s42"
OUT_DIR = ROOT / "reports/wow_history_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_METHOD = {
    "conservefm_full": "ConserveFM (original)",
    "no_corruption_pretrain": "No corruption pretraining",
    "direct_state_prediction": "Direct state prediction",
    "equal_constraint_weights": "Equal constraint weights",
    "manual_tuned_weights": "Manual tuned weights",
    "no_advection_constraint": "No advection constraint",
    "no_hydrostatic_constraint": "No hydrostatic constraint",
    "no_moisture_constraint": "No moisture constraint",
    "no_spectral_constraint": "No spectral constraint",
    "no_constraint_type_head": "No type head",
    "no_localization_head": "No localization head",
    "no_advection_corruption": "No advection corruption",
    "no_hydrostatic_corruption": "No hydrostatic corruption",
    "static_physics_loss": "Static physics loss",
    "residual_repair": "Residual repair",
    "static_learned_weights": "Static learned weights",
    "hard_projection": "Hard projection",
    "raw_fm": "Raw FM",
    "No_Pretrain": "No corruption pretraining",
    "Direct_State": "Direct state prediction",
    "Smart_Final": "ConserveFM (smart-final)",
    "OLD_Full": "ConserveFM (original)",
    "FIX1": "ConserveFM (fix-1)",
    "FIX2": "ConserveFM (fix-2)",
    "FIX2_14K": "ConserveFM (fix-2 14k)",
}

METHOD_COLOR = {
    "conservefm_full": "#6A4C93",
    "no_corruption_pretrain": "#2A9D8F",
    "direct_state_prediction": "#F4A261",
    "equal_constraint_weights": "#E76F51",
    "manual_tuned_weights": "#C44569",
    "no_advection_constraint": "#577590",
    "no_hydrostatic_constraint": "#8E9AAF",
    "no_moisture_constraint": "#2B9348",
    "no_spectral_constraint": "#9C6644",
    "no_constraint_type_head": "#3A86FF",
    "no_localization_head": "#8338EC",
    "no_advection_corruption": "#4361EE",
    "no_hydrostatic_corruption": "#4D908E",
    "static_physics_loss": "#BC4749",
    "residual_repair": "#6D597A",
    "static_learned_weights": "#7F5539",
    "No_Pretrain": "#2A9D8F",
    "Direct_State": "#F4A261",
    "Smart_Final": "#3A86FF",
    "OLD_Full": "#6A4C93",
    "FIX1": "#BC4749",
    "FIX2": "#8338EC",
    "FIX2_14K": "#4361EE",
}

STATUS_COLOR = {
    "completed": "#2A9D8F",
    "interrupted": "#F4A261",
    "running": "#3A86FF",
    "planned": "#C9CED6",
    "failed": "#E63946",
    "unknown": "#ADB5BD",
}

BACKBONE_COLOR = {
    "climax": "#3A86FF",
    "fourcastnet": "#8D5A97",
    "prithvi_wxc": "#2A9D8F",
}

def setup_style():
    mpl.rcParams.update({
        "figure.facecolor": "#F6F7FB",
        "axes.facecolor": "#F6F7FB",
        "savefig.facecolor": "#F6F7FB",
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titlesize": 24,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.titlesize": 28,
        "axes.edgecolor": "#111111",
        "axes.linewidth": 1.2,
        "grid.color": "#A9B0BB",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.18,
        "axes.grid": False,
    })

def stroke_text(text_obj, lw=3, fg="white"):
    text_obj.set_path_effects([pe.withStroke(linewidth=lw, foreground=fg)])

def stroke_artist(artist, lw=4, fg="black"):
    artist.set_path_effects([pe.Stroke(linewidth=lw, foreground=fg), pe.Normal()])

def save_all(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            stem.with_suffix("." + ext),
            dpi=220 if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.12,
        )
    plt.close(fig)

def pretty_method(name: str) -> str:
    if pd.isna(name):
        return "Unknown"
    s = str(name)
    if s in DISPLAY_METHOD:
        return DISPLAY_METHOD[s]
    return s.replace("_", " ").strip().title()

def canonical_final_name(name: str) -> str:
    s = str(name).strip()
    low = s.lower()
    mapping = {
        "no_pretrain": "No_Pretrain",
        "no pretrain": "No_Pretrain",
        "no corruption pretraining": "No_Pretrain",
        "direct_state": "Direct_State",
        "direct state": "Direct_State",
        "direct state prediction": "Direct_State",
        "smart_final": "Smart_Final",
        "conservefm (smart-final)": "Smart_Final",
        "old_full": "OLD_Full",
        "conservefm (original)": "OLD_Full",
        "fix1": "FIX1",
        "fix2": "FIX2",
        "fix2_14k": "FIX2_14K",
    }
    return mapping.get(low, s)

def safe_read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read {path}: {e}")
        return pd.DataFrame()

def find_candidate(base: Path, names: list[str]) -> Path | None:
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None

def find_physics_csv(base: Path) -> Path | None:
    cands = sorted(base.glob("*.csv"))
    for p in cands:
        low = p.name.lower()
        if "physics" in low or "table3" in low:
            return p
    return None

def detect_col(df: pd.DataFrame, patterns: list[str]) -> str | None:
    cols = list(df.columns)
    low_map = {c.lower(): c for c in cols}
    for pat in patterns:
        if pat.lower() in low_map:
            return low_map[pat.lower()]
    for pat in patterns:
        for c in cols:
            if pat.lower() in c.lower():
                return c
    return None

def gradient_rect(ax, rect, color, orientation="vertical", alpha0=0.18, alpha1=0.88):
    if rect is None:
        return
    rgba = np.array(to_rgba(color))
    x = rect.get_x()
    y = rect.get_y()
    w = rect.get_width()
    h = rect.get_height()
    if abs(w) < 1e-12 or abs(h) < 1e-12:
        return

    x0, x1 = (x, x + w) if w >= 0 else (x + w, x)
    y0, y1 = (y, y + h) if h >= 0 else (y + h, y)

    if orientation == "vertical":
        grad = np.ones((256, 1, 4))
        grad[..., :3] = rgba[:3]
        grad[..., 3] = np.linspace(alpha0, alpha1, 256)[:, None]
    else:
        grad = np.ones((1, 256, 4))
        grad[..., :3] = rgba[:3]
        grad[..., 3] = np.linspace(alpha0, alpha1, 256)[None, :]

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    im = ax.imshow(
        grad,
        extent=[x0, x1, y0, y1],
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=rect.get_zorder() - 0.01,
    )
    im.set_clip_path(rect)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    rect.set_facecolor((0, 0, 0, 0))
    rect.set_edgecolor("#111111")
    rect.set_linewidth(1.0)

def apply_bar_gradients(ax, bars, colors, orientation="vertical"):
    if not isinstance(colors, (list, tuple, np.ndarray, pd.Series)):
        colors = [colors] * len(bars)
    for rect, c in zip(bars, colors):
        gradient_rect(ax, rect, c, orientation=orientation)

def load_plan() -> pd.DataFrame:
    p = safe_read_csv(PLAN_CSV)
    if p.empty:
        return p
    for c in ["run_id", "job_type", "backbone", "family", "method", "seed", "status"]:
        if c not in p.columns:
            p[c] = ""
    p["status"] = p["status"].replace("", "unknown").fillna("unknown")
    p["backbone"] = p["backbone"].fillna("unknown")
    p["job_type"] = p["job_type"].fillna("unknown")
    p["method"] = p["method"].fillna("unknown")
    return p

def load_train_histories(plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = {}
    if not plan.empty and "run_id" in plan.columns:
        meta = plan.set_index("run_id")[["job_type", "backbone", "family", "method", "seed", "status"]].to_dict("index")

    rows = []
    summary = []

    for hist_path in sorted(RUNS_DIR.glob("train_*/train_history.csv")):
        df = safe_read_csv(hist_path)
        if df.empty or "val_repair_mse" not in df.columns:
            continue

        run_dir = hist_path.parent
        run_id = run_dir.name
        m = meta.get(run_id, {})

        df = df.copy()
        for c in ["epoch", "global_step", "train_total", "val_repair_mse", "elapsed_seconds"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        for _, r in df.iterrows():
            rows.append({
                "run_id": run_id,
                "stage": r.get("stage", ""),
                "epoch": r.get("epoch", np.nan),
                "global_step": r.get("global_step", np.nan),
                "train_total": r.get("train_total", np.nan),
                "val_repair_mse": r.get("val_repair_mse", np.nan),
                "elapsed_seconds": r.get("elapsed_seconds", np.nan),
                "job_type": m.get("job_type", "train"),
                "backbone": m.get("backbone", "unknown"),
                "family": m.get("family", "unknown"),
                "method": m.get("method", "unknown"),
                "seed": str(m.get("seed", "")),
                "plan_status": m.get("status", "unknown"),
            })

        best_idx = df["val_repair_mse"].astype(float).idxmin()
        best_row = df.loc[best_idx]
        last_row = df.iloc[-1]
        ck_best = (run_dir / "checkpoint_best.pt").exists()
        ck_last = (run_dir / "checkpoint_last.pt").exists()

        summary.append({
            "run_id": run_id,
            "backbone": m.get("backbone", "unknown"),
            "family": m.get("family", "unknown"),
            "method": m.get("method", "unknown"),
            "seed": str(m.get("seed", "")),
            "plan_status": m.get("status", "unknown"),
            "best_val": float(best_row["val_repair_mse"]),
            "best_step": float(best_row["global_step"]) if "global_step" in best_row else np.nan,
            "last_val": float(last_row["val_repair_mse"]),
            "max_global_step": float(df["global_step"].max()),
            "n_history_rows": int(len(df)),
            "last_stage": str(last_row.get("stage", "")),
            "last_epoch": int(last_row.get("epoch", -1)) if pd.notna(last_row.get("epoch", np.nan)) else -1,
            "checkpoint_best": ck_best,
            "checkpoint_last": ck_last,
        })

    return pd.DataFrame(rows), pd.DataFrame(summary)

def load_final_bundle() -> dict[str, pd.DataFrame]:
    final_csv = find_candidate(FINAL_DIR, ["FINAL_SUMMARY.csv", "final_summary.csv"])
    corr_csv = find_candidate(FINAL_DIR, ["stress_by_corruption.csv"])
    sev_csv = find_candidate(FINAL_DIR, ["stress_by_severity.csv"])
    lead_csv = find_candidate(FINAL_DIR, ["stress_by_lead.csv"])
    phys_csv = find_physics_csv(FINAL_DIR)

    out = {
        "final": safe_read_csv(final_csv),
        "corr": safe_read_csv(corr_csv),
        "sev": safe_read_csv(sev_csv),
        "lead": safe_read_csv(lead_csv),
        "phys": safe_read_csv(phys_csv),
    }

    if not out["final"].empty:
        mcol = detect_col(out["final"], ["model", "method"])
        if mcol:
            out["final"]["__method__"] = out["final"][mcol].map(canonical_final_name)
        else:
            out["final"]["__method__"] = out["final"].index.astype(str)

    for key in ["corr", "sev", "lead", "phys"]:
        if not out[key].empty:
            mcol = detect_col(out[key], ["model", "method"])
            if mcol:
                out[key]["__method__"] = out[key][mcol].map(canonical_final_name)

    return out

def annotate_value(ax, x, y, s, ha="center", va="bottom", fontsize=12, color="#111111"):
    txt = ax.text(x, y, s, ha=ha, va=va, fontsize=fontsize, color=color, zorder=10)
    stroke_text(txt, lw=2.5, fg="#F6F7FB")
    return txt

def plot_preliminary_campaign(plan: pd.DataFrame, train_summary: pd.DataFrame):
    fig = plt.figure(figsize=(20, 11), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    status_order = ["completed", "interrupted", "running", "planned", "failed", "unknown"]

    if not plan.empty:
        grp = (
            plan.groupby(["backbone", "job_type", "status"])
            .size()
            .reset_index(name="n")
        )
        grp["bucket"] = grp["backbone"].astype(str) + "\n" + grp["job_type"].astype(str)
        pv = grp.pivot_table(index="bucket", columns="status", values="n", fill_value=0)
        for s in status_order:
            if s not in pv.columns:
                pv[s] = 0
        pv = pv[status_order]
        pv = pv.sort_index()

        bottoms = np.zeros(len(pv))
        x = np.arange(len(pv))
        for s in status_order:
            vals = pv[s].values
            bars = ax1.bar(
                x, vals, bottom=bottoms, color=STATUS_COLOR.get(s, "#BBBBBB"),
                edgecolor="#111111", linewidth=1.0, width=0.72, label=s.title()
            )
            apply_bar_gradients(ax1, bars, STATUS_COLOR.get(s, "#BBBBBB"), orientation="vertical")
            bottoms += vals

        for xi, total in zip(x, bottoms):
            annotate_value(ax1, xi, total + max(bottoms) * 0.02, f"{int(total)}", fontsize=12)

        ax1.set_xticks(x)
        ax1.set_xticklabels(pv.index, rotation=0)
        ax1.set_ylabel("Jobs")
        ax1.set_title("Preliminary 138-job campaign\nstatus composition")
        ax1.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False)
        ax1.grid(axis="y", alpha=0.14)

    if not train_summary.empty:
        top = train_summary.sort_values("max_global_step", ascending=False).head(20).copy()
        top["label"] = top.apply(
            lambda r: f"{pretty_method(r['method'])}  s{r['seed']}",
            axis=1
        )
        top = top.iloc[::-1]

        y = np.arange(len(top))
        cols = [BACKBONE_COLOR.get(b, "#888888") for b in top["backbone"]]
        bars = ax2.barh(
            y,
            top["max_global_step"],
            color=cols,
            edgecolor="#111111",
            linewidth=1.0,
            height=0.72,
        )
        apply_bar_gradients(ax2, bars, cols, orientation="horizontal")

        for i, (_, r) in enumerate(top.iterrows()):
            if r.get("checkpoint_best", False):
                ax2.scatter(
                    r["max_global_step"], i,
                    marker="*", s=190,
                    color="#FFD166",
                    edgecolor="#111111",
                    linewidths=0.9,
                    zorder=6,
                )
            annotate_value(
                ax2,
                r["max_global_step"] + top["max_global_step"].max() * 0.01,
                i,
                f"{int(r['max_global_step']):,}",
                ha="left",
                va="center",
                fontsize=11,
            )

        ax2.set_yticks(y)
        ax2.set_yticklabels(top["label"])
        ax2.set_xlabel("Maximum global step reached")
        ax2.set_title("Deepest reached training checkpoints\n(across the whole history)")
        ax2.grid(axis="x", alpha=0.14)

    fig.suptitle("ConserveFM experimental history — preliminary campaign", y=1.02)
    save_all(fig, OUT_DIR / "fig01_preliminary_campaign")

def plot_train_leaderboard(train_summary: pd.DataFrame):
    if train_summary.empty:
        return

    df = train_summary.copy()
    df = df[df["method"].notna() & (df["method"] != "unknown")]
    if df.empty:
        return

    grp = (
        df.groupby("method")
        .agg(
            mean_best_val=("best_val", "mean"),
            std_best_val=("best_val", "std"),
            seeds=("best_val", "count"),
            med_step=("max_global_step", "median"),
        )
        .reset_index()
        .sort_values("mean_best_val", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(18, 12), constrained_layout=True)
    y = np.arange(len(grp))
    colors = [METHOD_COLOR.get(m, "#6C757D") for m in grp["method"]]
    bars = ax.barh(
        y,
        grp["mean_best_val"],
        xerr=grp["std_best_val"].fillna(0.0),
        color=colors,
        edgecolor="#111111",
        linewidth=1.0,
        alpha=1.0,
        height=0.72,
        error_kw=dict(ecolor="#222222", lw=1.2, capsize=3),
    )
    apply_bar_gradients(ax, bars, colors, orientation="horizontal")

    ax.set_yticks(y)
    ax.set_yticklabels([pretty_method(m) for m in grp["method"]])
    ax.invert_yaxis()
    ax.set_xlabel("Best validation repair MSE (lower is better)")
    ax.set_title("Training leaderboard over the entire experiment history")
    ax.grid(axis="x", alpha=0.14)

    step_norm = np.sqrt(np.maximum(df["max_global_step"].fillna(0).values, 1))
    step_norm = 35 + 90 * (step_norm - step_norm.min()) / max(step_norm.max() - step_norm.min(), 1e-9)

    for i, m in enumerate(grp["method"]):
        sub = df[df["method"] == m]
        yy = i + np.linspace(-0.17, 0.17, len(sub))
        xx = sub["best_val"].values
        ss = 35 + 90 * (
            (np.sqrt(np.maximum(sub["max_global_step"].values, 1)) - np.sqrt(np.maximum(df["max_global_step"].min(), 1)))
            / max(np.sqrt(np.maximum(df["max_global_step"].max(), 1)) - np.sqrt(np.maximum(df["max_global_step"].min(), 1)), 1e-9)
        )
        ax.scatter(
            xx, yy,
            s=ss,
            color=METHOD_COLOR.get(m, "#6C757D"),
            edgecolor="#111111",
            linewidths=0.9,
            zorder=6,
        )

    for i, (_, r) in enumerate(grp.iterrows()):
        annotate_value(
            ax,
            r["mean_best_val"] + max(grp["mean_best_val"]) * 0.02,
            i,
            f"{r['mean_best_val']:.4f}  (n={int(r['seeds'])})",
            ha="left",
            va="center",
            fontsize=11,
        )

    save_all(fig, OUT_DIR / "fig02_training_leaderboard")

def plot_final_dashboard(final_df: pd.DataFrame):
    if final_df.empty:
        return

    df = final_df.copy()
    if "__method__" not in df.columns:
        return

    order = [m for m in ["No_Pretrain", "Direct_State", "Smart_Final", "OLD_Full", "FIX1", "FIX2", "FIX2_14K"] if m in df["__method__"].tolist()]
    if not order:
        order = df["__method__"].tolist()

    df = df.set_index("__method__").loc[order].reset_index()

    metrics = [
        (["clean_nrmse"], "Clean NRMSE ↓", False),
        (["clean_acc"], "Clean ACC ↑", True),
        (["macro_stress_nrmse", "stress_nrmse"], "Stress NRMSE ↓", False),
        (["macro_stress_acc", "stress_acc"], "Stress ACC ↑", True),
        (["macro_localization_iou", "loc_iou", "localization_iou"], "Localization IoU ↑", True),
        (["macro_type_accuracy", "type_accuracy"], "Type accuracy ↑", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12), constrained_layout=True)
    axes = axes.ravel()

    cols = [METHOD_COLOR.get(m, "#6C757D") for m in df["__method__"]]
    labels = [pretty_method(m) for m in df["__method__"]]

    for ax, (patterns, title, higher_better) in zip(axes, metrics):
        col = detect_col(df, patterns)
        if col is None:
            ax.axis("off")
            continue

        vals = pd.to_numeric(df[col], errors="coerce").values
        x = np.arange(len(df))
        bars = ax.bar(x, vals, color=cols, edgecolor="#111111", linewidth=1.0, width=0.72)
        apply_bar_gradients(ax, bars, cols, orientation="vertical")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=16, ha="right")
        ax.set_title(title)

        best_idx = np.nanargmax(vals) if higher_better else np.nanargmin(vals)
        for i, v in enumerate(vals):
            txt = f"{v:.4f}" if abs(v) < 10 else f"{v:.3f}"
            annotate_value(ax, i, v + (np.nanmax(vals) - np.nanmin(vals) + 1e-9) * 0.04, txt, fontsize=11)
            if i == best_idx:
                ax.scatter(i, v, s=180, marker="*", color="#FFD166", edgecolor="#111111", linewidths=0.9, zorder=7)

        ax.grid(axis="y", alpha=0.14)

    fig.suptitle("Focused diagnostic and final outcome — main comparable metrics", y=1.02)
    save_all(fig, OUT_DIR / "fig03_final_dashboard")

def heatmap_panel(ax, data: pd.DataFrame, title: str, value_fmt="{:.3f}"):
    if data.empty:
        ax.axis("off")
        return
    vmin = np.nanmin(data.values)
    vmax = np.nanmax(data.values)
    cmap = LinearSegmentedColormap.from_list(
        "wow_heat",
        ["#F3E9FF", "#CDB4DB", "#8EC5FC", "#5B8DEF", "#4C1D95"]
    )
    im = ax.imshow(data.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels([pretty_method(m) for m in data.index])

    ax.set_title(title)

    ax.set_xticks(np.arange(-.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#111111", linewidth=0.8, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            txt = ax.text(
                j, i, value_fmt.format(v),
                ha="center", va="center",
                fontsize=11, color="#111111"
            )
            stroke_text(txt, lw=2.5, fg="#F6F7FB")

    return im

def plot_stress_breakdowns(bundle: dict[str, pd.DataFrame]):
    corr = bundle["corr"].copy()
    sev = bundle["sev"].copy()
    lead = bundle["lead"].copy()

    if corr.empty and sev.empty and lead.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 9), constrained_layout=True)
    order = [m for m in ["No_Pretrain", "Direct_State", "Smart_Final", "OLD_Full", "FIX1", "FIX2", "FIX2_14K"]]

    if not corr.empty:
        vcol = detect_col(corr, ["nrmse", "rmse"])
        ccol = detect_col(corr, ["corruption"])
        if vcol and ccol and "__method__" in corr.columns:
            pv = corr.pivot_table(index="__method__", columns=ccol, values=vcol, aggfunc="mean")
            pv = pv.reindex([m for m in order if m in pv.index])
            heatmap_panel(axes[0], pv, "Stress breakdown by corruption")
        else:
            axes[0].axis("off")
    else:
        axes[0].axis("off")

    if not sev.empty:
        vcol = detect_col(sev, ["nrmse", "rmse"])
        scol = detect_col(sev, ["severity"])
        if vcol and scol and "__method__" in sev.columns:
            pv = sev.pivot_table(index="__method__", columns=scol, values=vcol, aggfunc="mean")
            pv = pv.reindex([m for m in order if m in pv.index])
            pv = pv.reindex(sorted(pv.columns, key=lambda x: float(x)), axis=1)
            heatmap_panel(axes[1], pv, "Stress breakdown by severity")
        else:
            axes[1].axis("off")
    else:
        axes[1].axis("off")

    if not lead.empty:
        vcol = detect_col(lead, ["nrmse", "rmse"])
        lcol = detect_col(lead, ["lead"])
        if vcol and lcol and "__method__" in lead.columns:
            pv = lead.pivot_table(index="__method__", columns=lcol, values=vcol, aggfunc="mean")
            pv = pv.reindex([m for m in order if m in pv.index])
            pv = pv.reindex(sorted(pv.columns, key=lambda x: float(re.sub(r'[^0-9.]', '', str(x)) or "0")), axis=1)
            heatmap_panel(axes[2], pv, "Stress breakdown by lead")
        else:
            axes[2].axis("off")
    else:
        axes[2].axis("off")

    fig.suptitle("Robustness anatomy — where each model wins or fails", y=1.03)
    save_all(fig, OUT_DIR / "fig04_stress_breakdowns")

def plot_pareto(bundle: dict[str, pd.DataFrame]):
    final_df = bundle["final"].copy()
    phys_df = bundle["phys"].copy()

    if final_df.empty or phys_df.empty or "__method__" not in final_df.columns or "__method__" not in phys_df.columns:
        return

    stress_col = detect_col(final_df, ["macro_stress_nrmse", "stress_nrmse", "nrmse"])
    phys_col = detect_col(phys_df, ["physics ratio gm", "physics_ratio_gm", "gm", "Physics ratio GM"])
    if stress_col is None or phys_col is None:
        return

    a = final_df[["__method__", stress_col]].copy()
    b = phys_df[["__method__", phys_col]].copy()
    a.columns = ["__method__", "__stress__"]
    b.columns = ["__method__", "__phys__"]

    df = a.merge(b, on="__method__", how="inner")
    if df.empty:
        return

    def pareto_mask(x, y):
        mask = []
        for i in range(len(x)):
            dominated = False
            for j in range(len(x)):
                if i == j:
                    continue
                better_or_equal = (x[j] <= x[i]) and (y[j] <= y[i])
                strictly_better = (x[j] < x[i]) or (y[j] < y[i])
                if better_or_equal and strictly_better:
                    dominated = True
                    break
            mask.append(not dominated)
        return np.array(mask, dtype=bool)

    x = df["__stress__"].astype(float).values
    y = df["__phys__"].astype(float).values
    mask = pareto_mask(x, y)

    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)

    for _, r in df.iterrows():
        m = r["__method__"]
        col = METHOD_COLOR.get(m, "#6C757D")
        size = 340 if m == "Smart_Final" else 240
        ax.scatter(
            float(r["__stress__"]), float(r["__phys__"]),
            s=size,
            color=col,
            edgecolor="#111111",
            linewidths=1.2,
            zorder=5,
        )
        txt = ax.text(
            float(r["__stress__"]) + 0.0022,
            float(r["__phys__"]) + 0.010,
            pretty_method(m),
            fontsize=13,
            color="#111111",
            zorder=6,
        )
        stroke_text(txt, lw=2.6, fg="#F6F7FB")

    front = df[mask].sort_values("__stress__")
    ax.plot(front["__stress__"], front["__phys__"], color="#111111", linewidth=2.0, alpha=0.55, zorder=4)
    stroke_artist(ax.lines[-1], lw=3.6, fg="#8FA3BF")

    ax.set_xlabel("Stress NRMSE ↓")
    ax.set_ylabel("Physics violation GM ↓")
    ax.set_title("Accuracy–physics Pareto front")

    ax.grid(alpha=0.16)
    save_all(fig, OUT_DIR / "fig05_accuracy_physics_pareto")

def plot_seed42_curves(history_rows: pd.DataFrame):
    if history_rows.empty:
        return

    df = history_rows.copy()
    df = df[df["seed"].astype(str) == "42"].copy()
    if df.empty:
        return

    targets = [
        "conservefm_full",
        "no_corruption_pretrain",
        "direct_state_prediction",
        "equal_constraint_weights",
        "no_advection_constraint",
    ]
    df = df[df["method"].isin(targets)]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(17, 10), constrained_layout=True)

    for m in targets:
        sub = df[df["method"] == m].sort_values("global_step")
        if sub.empty:
            continue
        col = METHOD_COLOR.get(m, "#6C757D")
        line, = ax.plot(
            sub["global_step"], sub["val_repair_mse"],
            color=col,
            linewidth=3.0,
            marker="o",
            markersize=8,
            markeredgecolor="#111111",
            markeredgewidth=0.7,
            alpha=0.96,
            label=pretty_method(m),
        )
        stroke_artist(line, lw=4.8, fg="#111111")

    ax.set_xlabel("Global step")
    ax.set_ylabel("Validation repair MSE ↓")
    ax.set_title("Seed-42 learning trajectories of the key variants")
    ax.grid(alpha=0.14)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    save_all(fig, OUT_DIR / "fig06_seed42_learning_curves")

def build_index_text(plan: pd.DataFrame, train_summary: pd.DataFrame, bundle: dict[str, pd.DataFrame]):
    lines = []
    lines.append("WOW FIGURES INDEX")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Plan CSV               : {PLAN_CSV}")
    lines.append(f"Runs dir               : {RUNS_DIR}")
    lines.append(f"Final metrics dir      : {FINAL_DIR}")
    lines.append(f"Output dir             : {OUT_DIR}")
    lines.append("")
    if not plan.empty:
        lines.append(f"Plan rows              : {len(plan)}")
        lines.append(f"Train rows in plan     : {(plan['job_type'] == 'train').sum() if 'job_type' in plan else 'n/a'}")
        lines.append(f"Eval rows in plan      : {(plan['job_type'] == 'eval').sum() if 'job_type' in plan else 'n/a'}")
        lines.append("Plan status counts:")
        for k, v in plan["status"].value_counts(dropna=False).items():
            lines.append(f"  - {k}: {v}")
        lines.append("")
    if not train_summary.empty:
        lines.append(f"Train histories found  : {len(train_summary)}")
        lines.append("Top 10 deepest checkpoints:")
        top = train_summary.sort_values("max_global_step", ascending=False).head(10)
        for _, r in top.iterrows():
            lines.append(f"  - {r['run_id']}: step={int(r['max_global_step'])}, best_val={r['best_val']:.6f}")
        lines.append("")
    if not bundle["final"].empty:
        lines.append("Final bundle:")
        for key in ["final", "corr", "sev", "lead", "phys"]:
            lines.append(f"  - {key}: {'OK' if not bundle[key].empty else 'missing'}")
        lines.append("")
    lines.append("Generated figures:")
    for p in sorted(OUT_DIR.glob("fig*.png")):
        lines.append(f"  - {p.name}")
    (OUT_DIR / "FIGURES_INDEX.txt").write_text("\n".join(lines), encoding="utf-8")

def main():
    setup_style()
    print("=" * 88)
    print("MAKE WOW HISTORY FIGURES")
    print("=" * 88)

    plan = load_plan()
    history_rows, train_summary = load_train_histories(plan)
    bundle = load_final_bundle()

    print(f"[INFO] plan rows           : {len(plan) if not plan.empty else 0}")
    print(f"[INFO] train histories     : {len(train_summary) if not train_summary.empty else 0}")
    for key in ["final", "corr", "sev", "lead", "phys"]:
        print(f"[INFO] final bundle {key:5s} : {'OK' if not bundle[key].empty else 'missing'}")

    plot_preliminary_campaign(plan, train_summary)
    plot_train_leaderboard(train_summary)
    plot_final_dashboard(bundle["final"])
    plot_stress_breakdowns(bundle)
    plot_pareto(bundle)
    plot_seed42_curves(history_rows)

    build_index_text(plan, train_summary, bundle)

    print("")
    print("=" * 88)
    print("DONE")
    print("=" * 88)
    for p in sorted(OUT_DIR.glob("fig*.png")):
        print(p)

if __name__ == "__main__":
    main()
