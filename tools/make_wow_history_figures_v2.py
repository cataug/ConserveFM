from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patheffects as pe

# Reuse ONLY the data-loading logic from the old script.
TOOLS = Path.home() / "ConserveFM" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import make_wow_history_figures as H


ROOT = Path.home() / "ConserveFM"
FINAL = ROOT / "reports/final_metrics_s42"
PAPER = FINAL / "paper"

OUT = ROOT / "reports/wow_history_figures_v2"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# STYLE
# ============================================================

WHITE = "#FFFFFF"
BLACK = "#111111"
RED = "#B52232"
GRID = "#D9DCE3"

COLORS = {
    "No_Pretrain": "#16A39A",
    "Direct_State": "#F3A261",
    "Smart_Final": "#6272E8",
    "OLD_Full": "#C94F6D",

    "conservefm_full": "#7A5AA6",
    "no_corruption_pretrain": "#16A39A",
    "direct_state_prediction": "#F3A261",
    "equal_constraint_weights": "#E76F51",
    "manual_tuned_weights": "#C44569",
    "no_advection_constraint": "#4D83A6",
    "no_hydrostatic_constraint": "#7698B3",
    "no_moisture_constraint": "#4F9B62",
    "no_spectral_constraint": "#9C6644",
    "no_constraint_type_head": "#3A86FF",
    "no_localization_head": "#8338EC",
    "no_advection_corruption": "#4361EE",
    "no_hydrostatic_corruption": "#4D908E",
    "static_physics_loss": "#BC4749",
    "residual_repair": "#6D597A",
    "static_learned_weights": "#7F5539",

    "completed": "#16A39A",
    "interrupted": "#F4A261",
    "running": "#4D8BFF",
    "planned": "#BFC5CE",
    "failed": "#D62839",
    "unknown": "#8D99AE",
}

FINAL_LABEL = {
    "No_Pretrain": "ConserveFM-Acc",
    "Direct_State": "Direct state",
    "Smart_Final": "ConserveFM-Phys",
    "OLD_Full": "Original full",
}

PAPER_TO_CANONICAL = {
    "ConserveFM-Acc": "No_Pretrain",
    "Direct state": "Direct_State",
    "ConserveFM-Phys": "Smart_Final",
    "Original full": "OLD_Full",
}


def setup_style():
    mpl.rcParams.update({
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "axes.facecolor": WHITE,

        "font.family": "DejaVu Sans",
        "font.size": 20,

        "axes.titlesize": 27,
        "axes.labelsize": 23,

        "xtick.labelsize": 18,
        "ytick.labelsize": 18,

        "legend.fontsize": 17,

        "axes.linewidth": 1.3,
        "axes.edgecolor": RED,

        "text.color": BLACK,
        "axes.labelcolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_panel(ax, title=None):
    ax.set_facecolor(WHITE)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(RED)
        spine.set_linewidth(1.45)

    ax.tick_params(
        axis="both",
        width=1.0,
        length=5,
        color=BLACK,
        labelcolor=BLACK,
    )

    ax.grid(
        axis="both",
        color=GRID,
        linewidth=0.75,
        alpha=0.55,
        zorder=0,
    )

    if title:
        ax.set_title(
            title,
            pad=16,
            fontweight="bold",
            bbox=dict(
                facecolor=WHITE,
                edgecolor=RED,
                linewidth=1.25,
                boxstyle="round,pad=0.38",
            ),
        )


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"{name}.{ext}",
            dpi=260 if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.12,
            facecolor=WHITE,
        )

    plt.close(fig)


def text_stroke(obj, width=3):
    obj.set_path_effects([
        pe.withStroke(
            linewidth=width,
            foreground=WHITE,
        )
    ])


# ============================================================
# ROBUST GRADIENT BARS
#
# No imshow().
# Therefore NO autoscale corruption.
# ============================================================

def gradient_barh(
    ax,
    y,
    width,
    height,
    color,
    left=0.0,
    steps=44,
    alpha_low=0.20,
    alpha_high=0.92,
    zorder=3,
):
    if not np.isfinite(width) or width <= 0:
        return None

    dx = width / steps

    for i in range(steps):
        t = (i + 1) / steps

        rect = Rectangle(
            (
                left + i * dx,
                y - height / 2,
            ),
            dx * 1.04,
            height,
            facecolor=color,
            edgecolor="none",
            alpha=alpha_low + (alpha_high - alpha_low) * t,
            clip_on=True,
            zorder=zorder,
        )

        ax.add_patch(rect)

    border = Rectangle(
        (left, y - height / 2),
        width,
        height,
        facecolor="none",
        edgecolor=BLACK,
        linewidth=0.90,
        clip_on=True,
        zorder=zorder + 1,
    )

    ax.add_patch(border)
    return border


def gradient_bar(
    ax,
    x,
    height,
    width,
    color,
    bottom=0.0,
    steps=44,
    alpha_low=0.20,
    alpha_high=0.92,
    zorder=3,
):
    if not np.isfinite(height) or height <= 0:
        return None

    dy = height / steps

    for i in range(steps):
        t = (i + 1) / steps

        rect = Rectangle(
            (
                x - width / 2,
                bottom + i * dy,
            ),
            width,
            dy * 1.04,
            facecolor=color,
            edgecolor="none",
            alpha=alpha_low + (alpha_high - alpha_low) * t,
            clip_on=True,
            zorder=zorder,
        )

        ax.add_patch(rect)

    border = Rectangle(
        (
            x - width / 2,
            bottom,
        ),
        width,
        height,
        facecolor="none",
        edgecolor=BLACK,
        linewidth=0.90,
        clip_on=True,
        zorder=zorder + 1,
    )

    ax.add_patch(border)
    return border


# ============================================================
# LABEL HELPERS
# ============================================================

def nice_backbone(x):
    x = str(x)

    return {
        "climax": "ClimaX",
        "fourcastnet": "FourCastNet",
        "prithvi_wxc": "Prithvi WxC",
    }.get(
        x.lower(),
        x.replace("_", " "),
    )


def nice_method(m):
    m = str(m)

    if m in FINAL_LABEL:
        return FINAL_LABEL[m]

    return H.pretty_method(m)


def infer_run_label(row):
    method = str(row.get("method", "unknown"))
    seed = str(row.get("seed", "")).strip()

    if method not in ("", "unknown", "nan"):
        return f"{nice_method(method)} · s{seed}"

    run = str(row.get("run_id", "")).lower()

    if "smartfinal" in run:
        name = "ConserveFM-Phys"
    elif "fix2probe" in run:
        name = "Fix-2 probe"
    elif "full_fix" in run:
        name = "Fix-1 rescue"
    elif "no_corruption_pretrain" in run:
        name = "ConserveFM-Acc"
    elif "direct_state" in run:
        name = "Direct state"
    else:
        name = str(row.get("run_id", "")).replace("train_climax_", "")

    sm = re.search(r"_s(\d+)", run)

    if sm:
        return f"{name} · s{sm.group(1)}"

    return name


# ============================================================
# LOAD
# ============================================================

plan = H.load_plan()
history, train_summary = H.load_train_histories(plan)
bundle = H.load_final_bundle()

final = pd.read_csv(FINAL / "FINAL_SUMMARY.csv")

if "__method__" not in final.columns:
    mcol = H.detect_col(final, ["model", "method"])

    final["__method__"] = (
        final[mcol]
        .map(H.canonical_final_name)
    )


physics_boot = pd.DataFrame()
p = PAPER / "physics_ratios_bootstrap95.csv"

if p.exists():
    physics_boot = pd.read_csv(p)
    physics_boot["__method__"] = (
        physics_boot["Method"]
        .map(PAPER_TO_CANONICAL)
    )


pareto = pd.DataFrame()
p = PAPER / "accuracy_physics_pareto.csv"

if p.exists():
    pareto = pd.read_csv(p)
    pareto["__method__"] = (
        pareto["Method"]
        .map(PAPER_TO_CANONICAL)
    )


# ============================================================
# FIG 1
# 138-JOB PRELIMINARY CAMPAIGN
# ============================================================

def fig01_campaign():
    if plan.empty:
        return

    g = (
        plan
        .groupby(
            ["backbone", "job_type", "status"]
        )
        .size()
        .reset_index(name="n")
    )

    g["bucket"] = g.apply(
        lambda r:
        f"{nice_backbone(r['backbone'])} · {str(r['job_type']).title()}",
        axis=1,
    )

    pv = g.pivot_table(
        index="bucket",
        columns="status",
        values="n",
        fill_value=0,
    )

    status_order = [
        "completed",
        "running",
        "interrupted",
        "planned",
        "failed",
        "unknown",
    ]

    for s in status_order:
        if s not in pv.columns:
            pv[s] = 0

    pv["__total__"] = pv[status_order].sum(axis=1)

    pv = (
        pv
        .sort_values("__total__", ascending=True)
    )

    fig, ax = plt.subplots(
        figsize=(15.5, 8.5),
        constrained_layout=True,
    )

    style_panel(
        ax,
        "Preliminary experimental campaign · 138 planned jobs",
    )

    ys = np.arange(len(pv))
    left = np.zeros(len(pv))

    for status in status_order:
        vals = pv[status].to_numpy(dtype=float)

        for i, v in enumerate(vals):
            if v <= 0:
                continue

            gradient_barh(
                ax,
                ys[i],
                v,
                0.63,
                COLORS.get(status, "#AAAAAA"),
                left=left[i],
            )

            if v >= 4:
                t = ax.text(
                    left[i] + v / 2,
                    ys[i],
                    f"{int(v)}",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    zorder=8,
                )
                text_stroke(t, 3)

        left += vals

    xmax = float(pv["__total__"].max())

    for i, total in enumerate(
        pv["__total__"].to_numpy()
    ):
        t = ax.text(
            total + xmax * 0.018,
            ys[i],
            f"{int(total)}",
            va="center",
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        text_stroke(t, 3)

    ax.set_yticks(ys)
    ax.set_yticklabels(
        pv.index,
        fontsize=19,
    )

    ax.set_xlabel(
        "Number of scheduled jobs",
        fontsize=22,
        fontweight="bold",
    )

    ax.set_xlim(
        0,
        xmax * 1.13,
    )

    ax.set_ylim(
        -0.75,
        len(pv) - 0.25,
    )

    handles = []

    for status in status_order:
        if pv[status].sum() <= 0:
            continue

        handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=COLORS.get(status),
                edgecolor=BLACK,
                linewidth=0.8,
                label=status.title(),
            )
        )

    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        facecolor=WHITE,
        edgecolor=RED,
        framealpha=1.0,
        ncol=2,
    )

    ax.grid(
        axis="x",
        color=GRID,
        alpha=0.7,
    )

    ax.grid(
        axis="y",
        visible=False,
    )

    save(fig, "fig01_preliminary_138_job_campaign")


# ============================================================
# FIG 2
# TRAINING DEPTH — TOP RUNS
# ============================================================

def fig02_training_depth():
    if train_summary.empty:
        return

    q = (
        train_summary
        .sort_values(
            "max_global_step",
            ascending=False,
        )
        .head(14)
        .copy()
    )

    q["label"] = q.apply(
        infer_run_label,
        axis=1,
    )

    q = q.iloc[::-1]

    fig, ax = plt.subplots(
        figsize=(16.5, 10.5),
        constrained_layout=True,
    )

    style_panel(
        ax,
        "Deepest reached training checkpoints · preliminary search history",
    )

    y = np.arange(len(q))
    maxstep = float(
        q["max_global_step"].max()
    )

    for i, (_, r) in enumerate(
        q.iterrows()
    ):
        m = str(r.get("method", "unknown"))

        color = COLORS.get(
            m,
            "#5B8DEF",
        )

        gradient_barh(
            ax,
            i,
            float(r["max_global_step"]),
            0.62,
            color,
        )

        step = float(
            r["max_global_step"]
        )

        t = ax.text(
            step + maxstep * 0.012,
            i,
            f"{step/1000:.0f}k",
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

        text_stroke(t, 3)

        if bool(
            r.get(
                "checkpoint_best",
                False,
            )
        ):
            ax.scatter(
                step,
                i,
                s=190,
                marker="*",
                facecolor="#FFD166",
                edgecolor=BLACK,
                linewidth=1.0,
                zorder=10,
            )

    ax.set_yticks(y)

    ax.set_yticklabels(
        q["label"],
        fontsize=17,
    )

    ax.set_xlabel(
        "Maximum global training step reached",
        fontsize=22,
        fontweight="bold",
    )

    ax.set_xlim(
        0,
        maxstep * 1.14,
    )

    ax.set_ylim(
        -0.75,
        len(q) - 0.25,
    )

    ax.grid(
        axis="x",
        alpha=0.65,
    )

    ax.grid(
        axis="y",
        visible=False,
    )

    save(fig, "fig02_training_depth")


# ============================================================
# FIG 3
# PRELIMINARY TRAINING LEADERBOARD
# ============================================================

def fig03_training_leaderboard():
    if train_summary.empty:
        return

    q = train_summary.copy()

    q = q[
        q["method"].notna()
        & (q["method"] != "unknown")
    ]

    if q.empty:
        return

    g = (
        q.groupby("method")
        .agg(
            mean_best=("best_val", "mean"),
            std_best=("best_val", "std"),
            n=("best_val", "count"),
        )
        .reset_index()
        .sort_values("mean_best")
        .head(12)
    )

    # Reverse for horizontal display.
    g = g.iloc[::-1]

    baseline = max(
        0.0,
        float(g["mean_best"].min()) - 0.010,
    )

    fig, ax = plt.subplots(
        figsize=(16, 10),
        constrained_layout=True,
    )

    style_panel(
        ax,
        "Preliminary validation landscape across explored variants",
    )

    ys = np.arange(len(g))

    for i, (_, r) in enumerate(
        g.iterrows()
    ):
        m = r["method"]
        v = float(r["mean_best"])

        gradient_barh(
            ax,
            i,
            v - baseline,
            0.61,
            COLORS.get(m, "#778DA9"),
            left=baseline,
        )

        sd = (
            float(r["std_best"])
            if pd.notna(r["std_best"])
            else 0.0
        )

        if sd > 0:
            ax.errorbar(
                v,
                i,
                xerr=sd,
                fmt="none",
                ecolor=BLACK,
                elinewidth=1.4,
                capsize=4,
                zorder=8,
            )

        t = ax.text(
            v + 0.0012,
            i,
            f"{v:.4f}",
            va="center",
            ha="left",
            fontsize=16,
            fontweight="bold",
        )

        text_stroke(t, 3)

    ax.set_yticks(ys)

    ax.set_yticklabels(
        [nice_method(m) for m in g["method"]],
        fontsize=17,
    )

    ax.set_xlabel(
        "Best validation repair MSE ↓",
        fontsize=22,
        fontweight="bold",
    )

    ax.set_xlim(
        baseline,
        float(g["mean_best"].max()) + 0.014,
    )

    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    save(fig, "fig03_preliminary_validation_landscape")


# ============================================================
# FIG 4
# FINAL METRIC DASHBOARD
# ============================================================

def fig04_final_dashboard():
    q = final.copy()

    order = [
        "No_Pretrain",
        "Direct_State",
        "Smart_Final",
        "OLD_Full",
    ]

    q = (
        q.set_index("__method__")
        .reindex(order)
        .dropna(how="all")
        .reset_index()
    )

    physics_map = {}

    if not physics_boot.empty:
        for _, r in physics_boot.iterrows():
            physics_map[
                r["__method__"]
            ] = float(
                r["Physics GM"]
            )

    q["physics_gm"] = q[
        "__method__"
    ].map(physics_map)

    panels = [
        (
            "clean_nrmse",
            "Clean NRMSE ↓",
            False,
        ),
        (
            "macro_stress_nrmse",
            "Stress NRMSE ↓",
            False,
        ),
        (
            "macro_stress_acc",
            "Stress ACC ↑",
            True,
        ),
        (
            "physics_gm",
            "Physics violation GM ↓",
            False,
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17.5, 12),
        constrained_layout=True,
    )

    labels = [
        FINAL_LABEL.get(
            m,
            nice_method(m),
        )
        for m in q["__method__"]
    ]

    x = np.arange(len(q))

    for ax, (
        col,
        title,
        higher,
    ) in zip(
        axes.ravel(),
        panels,
    ):
        style_panel(
            ax,
            title,
        )

        vals = pd.to_numeric(
            q[col],
            errors="coerce",
        ).to_numpy()

        finite = vals[
            np.isfinite(vals)
        ]

        if not len(finite):
            ax.axis("off")
            continue

        ymin = max(
            0.0,
            float(finite.min())
            - 0.12 * (
                float(finite.max())
                - float(finite.min())
                + 1e-4
            ),
        )

        for i, (
            m,
            v,
        ) in enumerate(
            zip(
                q["__method__"],
                vals,
            )
        ):
            if not np.isfinite(v):
                continue

            gradient_bar(
                ax,
                i,
                v - ymin,
                0.62,
                COLORS.get(
                    m,
                    "#778DA9",
                ),
                bottom=ymin,
            )

            t = ax.text(
                i,
                v + (
                    finite.max()
                    - finite.min()
                    + 1e-3
                ) * 0.045,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=17,
                fontweight="bold",
            )

            text_stroke(t, 3)

        best = (
            int(np.nanargmax(vals))
            if higher
            else int(np.nanargmin(vals))
        )

        ax.scatter(
            best,
            vals[best],
            marker="*",
            s=245,
            facecolor="#FFD166",
            edgecolor=BLACK,
            linewidth=1.0,
            zorder=12,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            labels,
            rotation=12,
            ha="right",
            fontsize=17,
        )

        ax.set_ylim(
            ymin,
            finite.max()
            + (
                finite.max()
                - ymin
            ) * 0.18,
        )

        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    save(fig, "fig04_final_metric_dashboard")


# ============================================================
# HEATMAP
# ============================================================

def heat_panel(
    ax,
    table,
    title,
):
    style_panel(ax, title)

    vals = table.to_numpy(
        dtype=float
    )

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "wow",
        [
            "#FAFAFF",
            "#D9E4FF",
            "#A9C8FF",
            "#7189E8",
            "#5141A5",
        ],
    )

    im = ax.imshow(
        vals,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(
            table.shape[1]
        )
    )

    ax.set_xticklabels(
        table.columns,
        rotation=18,
        ha="right",
        fontsize=16,
    )

    ax.set_yticks(
        np.arange(
            table.shape[0]
        )
    )

    ax.set_yticklabels(
        [
            FINAL_LABEL.get(
                m,
                nice_method(m),
            )
            for m in table.index
        ],
        fontsize=17,
    )

    # Thin black cell borders.
    ax.set_xticks(
        np.arange(-0.5, table.shape[1], 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, table.shape[0], 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color=BLACK,
        linewidth=0.55,
        alpha=0.45,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for i in range(
        table.shape[0]
    ):
        for j in range(
            table.shape[1]
        ):
            v = vals[i, j]

            t = ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=BLACK,
            )

            text_stroke(t, 3)

    return im


# ============================================================
# FIG 5
# STRESS ANATOMY
# ============================================================

def fig05_stress_anatomy():
    corr = bundle["corr"].copy()
    sev = bundle["sev"].copy()
    lead = bundle["lead"].copy()

    if corr.empty:
        return

    order = [
        "No_Pretrain",
        "Direct_State",
        "Smart_Final",
        "OLD_Full",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24, 8.5),
        constrained_layout=True,
    )

    # Corruption
    vcol = H.detect_col(
        corr,
        ["repaired_nrmse", "nrmse", "rmse"],
    )

    ccol = H.detect_col(
        corr,
        ["corruption"],
    )

    pc = corr.pivot_table(
        index="__method__",
        columns=ccol,
        values=vcol,
        aggfunc="mean",
    )

    pc = pc.reindex(
        [m for m in order if m in pc.index]
    )

    heat_panel(
        axes[0],
        pc,
        "By corruption type",
    )

    # Severity
    vcol = H.detect_col(
        sev,
        ["repaired_nrmse", "nrmse", "rmse"],
    )

    scol = H.detect_col(
        sev,
        ["severity"],
    )

    ps = sev.pivot_table(
        index="__method__",
        columns=scol,
        values=vcol,
        aggfunc="mean",
    )

    ps = ps.reindex(
        [m for m in order if m in ps.index]
    )

    ps = ps.reindex(
        sorted(
            ps.columns,
            key=float,
        ),
        axis=1,
    )

    heat_panel(
        axes[1],
        ps,
        "By corruption severity",
    )

    # Lead
    vcol = H.detect_col(
        lead,
        ["repaired_nrmse", "nrmse", "rmse"],
    )

    lcol = H.detect_col(
        lead,
        ["lead_hours", "lead"],
    )

    pl = lead.pivot_table(
        index="__method__",
        columns=lcol,
        values=vcol,
        aggfunc="mean",
    )

    pl = pl.reindex(
        [m for m in order if m in pl.index]
    )

    pl = pl.reindex(
        sorted(
            pl.columns,
            key=float,
        ),
        axis=1,
    )

    pl.columns = [
        f"{int(float(c))} h"
        for c in pl.columns
    ]

    heat_panel(
        axes[2],
        pl,
        "By forecast lead",
    )

    save(
        fig,
        "fig05_stress_anatomy",
    )


# ============================================================
# FIG 6
# ACCURACY–PHYSICS PARETO
# ============================================================

def fig06_pareto():
    if pareto.empty:
        return

    fig, ax = plt.subplots(
        figsize=(13.5, 9.5),
        constrained_layout=True,
    )

    style_panel(
        ax,
        "Accuracy–physics Pareto frontier",
    )

    q = pareto.sort_values(
        "Stress NRMSE mean"
    )

    front = q[
        q["Pareto"].astype(bool)
    ].sort_values(
        "Stress NRMSE mean"
    )

    # Pale trade-off envelope.
    if len(front) >= 2:
        ax.fill_between(
            front["Stress NRMSE mean"],
            front["Physics GM"],
            np.max(q["Physics GM"]) * 1.03,
            color="#7189E8",
            alpha=0.08,
            zorder=1,
        )

        ax.plot(
            front["Stress NRMSE mean"],
            front["Physics GM"],
            color=BLACK,
            linewidth=1.7,
            alpha=0.70,
            zorder=3,
        )

    for _, r in q.iterrows():
        m = r["__method__"]
        x = float(
            r["Stress NRMSE mean"]
        )
        y = float(
            r["Physics GM"]
        )

        # Transparent gradient halo.
        for size, alpha in [
            (820, 0.05),
            (620, 0.08),
            (450, 0.12),
        ]:
            ax.scatter(
                x,
                y,
                s=size,
                facecolor=COLORS.get(m, "#777777"),
                edgecolor="none",
                alpha=alpha,
                zorder=3,
            )

        ax.scatter(
            x,
            y,
            s=240,
            facecolor=COLORS.get(m, "#777777"),
            edgecolor=BLACK,
            linewidth=1.0,
            zorder=6,
        )

        label = FINAL_LABEL.get(
            m,
            nice_method(m),
        )

        offsets = {
            "No_Pretrain": (0.0012, 0.020),
            "Direct_State": (0.0012, 0.020),
            "Smart_Final": (0.0012, -0.045),
            "OLD_Full": (-0.013, 0.020),
        }

        dx, dy = offsets.get(
            m,
            (0.001, 0.015),
        )

        t = ax.text(
            x + dx,
            y + dy,
            label,
            fontsize=18,
            fontweight="bold",
        )

        text_stroke(t, 4)

    ax.annotate(
        "better",
        xy=(
            q["Stress NRMSE mean"].min() - 0.001,
            q["Physics GM"].min() - 0.015,
        ),
        xytext=(
            q["Stress NRMSE mean"].min() + 0.011,
            q["Physics GM"].min() + 0.15,
        ),
        arrowprops=dict(
            arrowstyle="->",
            color=RED,
            linewidth=2.0,
        ),
        fontsize=19,
        color=RED,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Macro stress NRMSE ↓",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Physics violation geometric mean ↓",
        fontweight="bold",
    )

    save(
        fig,
        "fig06_accuracy_physics_pareto",
    )


# ============================================================
# FIG 7
# PHYSICS RATIOS
# ============================================================

def fig07_physics_ratios():
    if physics_boot.empty:
        return

    q = physics_boot.copy()

    order = [
        "No_Pretrain",
        "Direct_State",
        "Smart_Final",
        "OLD_Full",
    ]

    q = (
        q.set_index("__method__")
        .reindex(order)
        .dropna(how="all")
        .reset_index()
    )

    constraints = [
        ("moisture ratio", "Moisture", "#23A89A"),
        ("hydrostatic ratio", "Hydrostatic", "#5B8DEF"),
        ("advection ratio", "Advection", "#E76F51"),
        ("spectral ratio", "Spectral", "#9B72CF"),
    ]

    fig, ax = plt.subplots(
        figsize=(16.5, 9.5),
        constrained_layout=True,
    )

    style_panel(
        ax,
        "Physical-violation repair ratios",
    )

    n = len(q)
    x = np.arange(n)

    cluster = 0.72
    bw = cluster / len(constraints)

    for j, (
        col,
        label,
        color,
    ) in enumerate(
        constraints
    ):
        xpos = (
            x
            - cluster / 2
            + bw / 2
            + j * bw
        )

        vals = q[col].to_numpy(
            dtype=float
        )

        for xx, v in zip(
            xpos,
            vals,
        ):
            gradient_bar(
                ax,
                xx,
                v,
                bw * 0.88,
                color,
            )

        # Legend handles.
        ax.plot(
            [],
            [],
            marker="s",
            markersize=13,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor=BLACK,
            label=label,
        )

    # GM black diamond.
    gm = q[
        "Physics GM"
    ].to_numpy(
        dtype=float
    )

    ax.scatter(
        x,
        gm,
        marker="D",
        s=125,
        facecolor=WHITE,
        edgecolor=BLACK,
        linewidth=1.5,
        zorder=12,
        label="Geometric mean",
    )

    for xx, v in zip(
        x,
        gm,
    ):
        t = ax.text(
            xx,
            v + 0.055,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

        text_stroke(t, 3)

    # Important threshold.
    ax.axhline(
        1.0,
        color=RED,
        linestyle="--",
        linewidth=1.8,
        zorder=2,
    )

    ax.text(
        n - 0.55,
        1.025,
        "no net improvement",
        ha="right",
        va="bottom",
        color=RED,
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            FINAL_LABEL.get(
                m,
                nice_method(m),
            )
            for m in q["__method__"]
        ],
        fontsize=18,
    )

    ax.set_ylabel(
        "Violation after repair / violation before repair ↓",
        fontweight="bold",
    )

    ax.set_ylim(
        0,
        max(
            1.5,
            np.nanmax(
                q[
                    [
                        c[0]
                        for c in constraints
                    ]
                ].to_numpy()
            ) * 1.12,
        ),
    )

    ax.legend(
        loc="upper center",
        ncol=5,
        frameon=True,
        facecolor=WHITE,
        edgecolor=RED,
        framealpha=1,
    )

    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    save(
        fig,
        "fig07_physics_violation_ratios",
    )


# ============================================================
# FIG 8
# LEARNING CURVES
# ============================================================

def fig08_learning_curves():
    if history.empty:
        return

    q = history.copy()

    q = q[
        q["seed"].astype(str) == "42"
    ]

    targets = [
        "conservefm_full",
        "no_corruption_pretrain",
        "direct_state_prediction",
        "equal_constraint_weights",
        "no_advection_constraint",
    ]

    q = q[
        q["method"].isin(targets)
    ]

    if q.empty:
        return

    fig, ax = plt.subplots(
        figsize=(15.5, 9),
        constrained_layout=True,
    )

    style_panel(
        ax,
        "Learning trajectories of representative preliminary variants · seed 42",
    )

    for method in targets:
        s = (
            q[
                q["method"] == method
            ]
            .sort_values("global_step")
        )

        if s.empty:
            continue

        color = COLORS.get(
            method,
            "#777777",
        )

        # Soft halo.
        ax.plot(
            s["global_step"],
            s["val_repair_mse"],
            linewidth=9,
            color=color,
            alpha=0.10,
            zorder=2,
        )

        ax.plot(
            s["global_step"],
            s["val_repair_mse"],
            linewidth=3.1,
            marker="o",
            markersize=9,
            color=color,
            markeredgecolor=BLACK,
            markeredgewidth=0.9,
            label=nice_method(method),
            zorder=5,
        )

    ax.set_xlabel(
        "Global training step",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Validation repair MSE ↓",
        fontweight="bold",
    )

    ax.legend(
        frameon=True,
        facecolor=WHITE,
        edgecolor=RED,
        framealpha=1,
        ncol=2,
    )

    save(
        fig,
        "fig08_learning_trajectories",
    )


# ============================================================
# RUN ALL
# ============================================================

setup_style()

print("=" * 92)
print("WOW HISTORY FIGURES V2")
print("=" * 92)
print("plan rows       :", len(plan))
print("train histories :", len(train_summary))
print("physics table   :", "OK" if not physics_boot.empty else "MISSING")
print("pareto table    :", "OK" if not pareto.empty else "MISSING")
print()

fig01_campaign()
fig02_training_depth()
fig03_training_leaderboard()
fig04_final_dashboard()
fig05_stress_anatomy()
fig06_pareto()
fig07_physics_ratios()
fig08_learning_curves()

print("=" * 92)
print("GENERATED")
print("=" * 92)

for p in sorted(
    OUT.glob("*.png")
):
    print(p)

print("\nOUTPUT:", OUT)
