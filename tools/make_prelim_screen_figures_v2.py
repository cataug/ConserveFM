from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib import patheffects as pe


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path.home() / "ConserveFM"

PRELIM = ROOT / "reports/prelim20_salvage"
FINAL = ROOT / "reports/final_metrics_s42"
PAPER = FINAL / "paper"

# Append directly to existing V2 folder.
OUT = ROOT / "reports/wow_history_figures_v2"
OUT.mkdir(parents=True, exist_ok=True)

EVAL_CSV = PRELIM / "live_eval_status.csv"
TRAIN_CSV = PRELIM / "live_train_status.csv"
JOBS_CSV = PRELIM / "jobs.csv"
FINAL_CSV = FINAL / "FINAL_SUMMARY.csv"
PARETO_CSV = PAPER / "accuracy_physics_pareto.csv"

if not EVAL_CSV.exists():
    raise SystemExit(f"Missing: {EVAL_CSV}")


# =====================================================================
# STYLE
# =====================================================================

WHITE = "#FFFFFF"
BLACK = "#111111"
RED = "#B52132"
GRID = "#D9DDE5"

FAMILY_COLORS = {
    "Core / prediction": "#287C8E",
    "Corruption ablations": "#5276C7",
    "Constraint ablations": "#7C64B5",
    "Weighting / routing": "#D67A36",
    "Heads / objective": "#C54A6A",
}

FINAL_COLORS = {
    "No_Pretrain": "#159E91",
    "Direct_State": "#F09A50",
    "Smart_Final": "#596DE0",
    "OLD_Full": "#B84A69",
}

PRETTY = {
    "conservefm_full": "Original full",
    "no_corruption_pretrain": "No corruption pretraining",
    "direct_state_prediction": "Direct state prediction",
    "static_physics_loss": "Static physics loss",
    "residual_repair": "Residual repair",

    "no_moisture_corruption": "No moisture corruption",
    "no_hydrostatic_corruption": "No hydrostatic corruption",
    "no_advection_corruption": "No advection corruption",
    "no_spectral_corruption": "No spectral corruption",
    "no_range_extreme_corruption": "No range/extreme corruption",

    "no_moisture_constraint": "No moisture constraint",
    "no_hydrostatic_constraint": "No hydrostatic constraint",
    "no_advection_constraint": "No advection constraint",
    "no_spectral_constraint": "No spectral constraint",

    "equal_constraint_weights": "Equal constraint weights",
    "manual_tuned_weights": "Manual tuned weights",
    "static_learned_weights": "Static learned weights",

    "no_localization_head": "No localization head",
    "no_constraint_type_head": "No type head",
    "no_minimal_change": "No minimal-change term",
}


def setup_style():
    mpl.rcParams.update({
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "axes.facecolor": WHITE,

        "font.family": "DejaVu Sans",
        "font.size": 22,

        "axes.titlesize": 29,
        "axes.labelsize": 25,

        "xtick.labelsize": 19,
        "ytick.labelsize": 19,

        "legend.fontsize": 18,

        "text.color": BLACK,
        "axes.labelcolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,

        "axes.edgecolor": RED,
        "axes.linewidth": 1.45,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_panel(ax, title=None):
    ax.set_facecolor(WHITE)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(RED)
        spine.set_linewidth(1.55)

    ax.grid(
        color=GRID,
        linewidth=0.75,
        alpha=0.62,
        zorder=0,
    )

    ax.tick_params(
        axis="both",
        length=5,
        width=1.0,
        color=BLACK,
    )

    if title:
        ax.set_title(
            title,
            pad=17,
            fontweight="bold",
            bbox=dict(
                facecolor=WHITE,
                edgecolor=RED,
                linewidth=1.35,
                boxstyle="round,pad=0.42",
            ),
        )


def stroke_text(obj, lw=3):
    obj.set_path_effects([
        pe.withStroke(
            linewidth=lw,
            foreground=WHITE,
        )
    ])


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"{name}.{ext}",
            dpi=300 if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.12,
            facecolor=WHITE,
        )

    plt.close(fig)


# =====================================================================
# GRADIENT BARS
# =====================================================================

def gradient_barh(
    ax,
    y,
    width,
    height,
    color,
    left=0.0,
    steps=48,
    alpha_low=0.16,
    alpha_high=0.92,
    zorder=3,
):
    if not np.isfinite(width) or width <= 0:
        return

    dx = width / steps

    for i in range(steps):
        a = alpha_low + (
            alpha_high - alpha_low
        ) * ((i + 1) / steps)

        ax.add_patch(
            Rectangle(
                (
                    left + i * dx,
                    y - height / 2,
                ),
                dx * 1.04,
                height,
                facecolor=color,
                edgecolor="none",
                alpha=a,
                clip_on=True,
                zorder=zorder,
            )
        )

    ax.add_patch(
        Rectangle(
            (
                left,
                y - height / 2,
            ),
            width,
            height,
            facecolor="none",
            edgecolor=BLACK,
            linewidth=0.95,
            clip_on=True,
            zorder=zorder + 1,
        )
    )


# =====================================================================
# METHOD FAMILIES
# =====================================================================

def family(method: str) -> str:

    if method in {
        "conservefm_full",
        "no_corruption_pretrain",
        "direct_state_prediction",
        "static_physics_loss",
        "residual_repair",
    }:
        return "Core / prediction"

    if "_corruption" in method:
        return "Corruption ablations"

    if "_constraint" in method:
        return "Constraint ablations"

    if method in {
        "equal_constraint_weights",
        "manual_tuned_weights",
        "static_learned_weights",
    }:
        return "Weighting / routing"

    if method in {
        "no_localization_head",
        "no_constraint_type_head",
        "no_minimal_change",
    }:
        return "Heads / objective"

    return "Core / prediction"


def pretty(method):
    return PRETTY.get(
        method,
        method.replace("_", " ").title(),
    )


# =====================================================================
# LOAD + AGGREGATE
# =====================================================================

eval_df = pd.read_csv(EVAL_CSV)

eval_df = eval_df[
    (eval_df["backbone"] == "climax")
    & (eval_df["status"] == "completed")
].copy()

for c in [
    "stress_rmse_phys_mixed",
    "stress_mae_phys_mixed",
    "loc_iou",
    "type_acc",
]:
    eval_df[c] = pd.to_numeric(
        eval_df[c],
        errors="coerce",
    )


agg = (
    eval_df
    .groupby("method")
    .agg(
        n=("run_id", "count"),

        rmse_mean=(
            "stress_rmse_phys_mixed",
            "mean",
        ),

        rmse_std=(
            "stress_rmse_phys_mixed",
            "std",
        ),

        mae_mean=(
            "stress_mae_phys_mixed",
            "mean",
        ),

        loc_mean=(
            "loc_iou",
            "mean",
        ),

        type_mean=(
            "type_acc",
            "mean",
        ),
    )
    .reset_index()
)

agg["family"] = agg["method"].map(family)
agg["label"] = agg["method"].map(pretty)

agg.to_csv(
    OUT / "fig09_11_preliminary_method_statistics.csv",
    index=False,
)


print("=" * 100)
print("PRELIMINARY SCREEN DATA")
print("=" * 100)
print("ClimaX runs :", len(eval_df))
print("Methods     :", len(agg))
print("Seeds/method:")
print(
    agg["n"]
    .value_counts()
    .sort_index()
    .to_string()
)


# =====================================================================
# FIG 09 — ALL 20 CLIMAX METHODS, MEAN ± STD
# =====================================================================

def fig09_ranking():

    q = (
        agg
        .sort_values(
            "rmse_mean",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    # best on top
    q = q.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(
        figsize=(17.5, 14.5),
        constrained_layout=True,
    )

    style_panel(
        ax,
        "Preliminary 3-seed screening · 20 ClimaX variants",
    )

    y = np.arange(len(q))

    # Cropped baseline makes differences readable while keeping all values.
    xmin = max(
        0.0,
        float(q["rmse_mean"].min()) * 0.88,
    )

    xmax = float(
        (
            q["rmse_mean"]
            + q["rmse_std"].fillna(0)
        ).max()
    )

    span = xmax - xmin

    for i, (_, r) in enumerate(
        q.iterrows()
    ):
        fam = r["family"]
        color = FAMILY_COLORS[fam]

        gradient_barh(
            ax,
            i,
            float(r["rmse_mean"]) - xmin,
            0.63,
            color,
            left=xmin,
        )

        sd = (
            float(r["rmse_std"])
            if pd.notna(r["rmse_std"])
            else 0.0
        )

        if sd > 0:
            ax.errorbar(
                float(r["rmse_mean"]),
                i,
                xerr=sd,
                fmt="none",
                ecolor=BLACK,
                elinewidth=1.4,
                capsize=4.5,
                capthick=1.2,
                zorder=8,
            )

        t = ax.text(
            float(r["rmse_mean"]) + span * 0.018,
            i,
            (
                f"{r['rmse_mean']:.1f}"
                f" ± {sd:.1f}"
            ),
            ha="left",
            va="center",
            fontsize=16.5,
            fontweight="bold",
            zorder=9,
        )
        stroke_text(t, 3)

    # Highlight top-5.
    best5 = (
        agg
        .sort_values("rmse_mean")
        .head(5)["method"]
        .tolist()
    )

    for i, (_, r) in enumerate(
        q.iterrows()
    ):
        if r["method"] in best5:
            ax.scatter(
                xmin + span * 0.006,
                i,
                marker="*",
                s=175,
                facecolor="#FFD166",
                edgecolor=BLACK,
                linewidth=0.9,
                zorder=10,
            )

    ax.set_yticks(y)

    labels = []

    for _, r in q.iterrows():
        label = r["label"]

        if r["method"] in best5:
            label = "★ " + label

        labels.append(label)

    ax.set_yticklabels(
        labels,
        fontsize=18,
    )

    ax.set_xlabel(
        "Preliminary mixed-unit stress RMSE ↓  (screening metric only)",
        fontweight="bold",
    )

    ax.set_xlim(
        xmin,
        xmax + span * 0.25,
    )

    ax.set_ylim(
        -0.75,
        len(q) - 0.25,
    )

    ax.grid(
        axis="x",
        alpha=0.70,
    )

    ax.grid(
        axis="y",
        visible=False,
    )

    handles = []

    for fam, color in FAMILY_COLORS.items():
        handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=color,
                edgecolor=BLACK,
                linewidth=0.8,
                label=fam,
            )
        )

    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        facecolor=WHITE,
        edgecolor=RED,
        framealpha=1,
        fontsize=15.5,
    )

    ax.text(
        0.012,
        0.988,
        (
            "Uniform lightweight 60-condition stress screen · "
            "error bars = standard deviation across three seeds"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        bbox=dict(
            facecolor=WHITE,
            edgecolor=RED,
            linewidth=1.0,
            boxstyle="round,pad=0.32",
        ),
    )

    save(
        fig,
        "fig09_preliminary_20method_ranking",
    )


# =====================================================================
# FIG 10 — PERFORMANCE × STABILITY
# =====================================================================

def fig10_stability():

    q = agg.copy()

    fig, ax = plt.subplots(
        figsize=(15.5, 10.5),
        constrained_layout=True,
    )

    style_panel(
        ax,
        "Performance–stability map of the preliminary search",
    )

    xmed = float(
        q["rmse_mean"].median()
    )

    positive_std = q[
        q["rmse_std"] > 0
    ]["rmse_std"]

    ymed = float(
        positive_std.median()
    )

    # Desirable lower-left region.
    ax.axvspan(
        q["rmse_mean"].min() * 0.96,
        xmed,
        color="#63C5A5",
        alpha=0.06,
        zorder=0,
    )

    ax.axhspan(
        max(
            positive_std.min() * 0.65,
            0.20,
        ),
        ymed,
        color="#63C5A5",
        alpha=0.06,
        zorder=0,
    )

    ax.axvline(
        xmed,
        color=RED,
        linestyle="--",
        linewidth=1.35,
        alpha=0.55,
    )

    ax.axhline(
        ymed,
        color=RED,
        linestyle="--",
        linewidth=1.35,
        alpha=0.55,
    )

    for _, r in q.iterrows():

        x = float(r["rmse_mean"])
        y = max(
            float(r["rmse_std"]),
            0.25,
        )

        color = FAMILY_COLORS[
            r["family"]
        ]

        # Halo.
        for size, alpha in [
            (570, 0.045),
            (430, 0.070),
            (320, 0.110),
        ]:
            ax.scatter(
                x,
                y,
                s=size,
                color=color,
                edgecolor="none",
                alpha=alpha,
                zorder=3,
            )

        ax.scatter(
            x,
            y,
            s=160,
            color=color,
            edgecolor=BLACK,
            linewidth=0.95,
            zorder=5,
        )

    # Label informative points, not every single dot.
    label_methods = {
        "direct_state_prediction",
        "no_corruption_pretrain",
        "no_advection_constraint",
        "residual_repair",
        "equal_constraint_weights",
        "manual_tuned_weights",
        "conservefm_full",
        "no_hydrostatic_corruption",
    }

    offsets = {
        "direct_state_prediction": (5, 1.18),
        "no_corruption_pretrain": (5, 1.18),
        "no_advection_constraint": (5, 1.18),
        "residual_repair": (5, 1.30),
        "equal_constraint_weights": (5, 1.15),
        "manual_tuned_weights": (-130, 1.10),
        "conservefm_full": (-115, 0.75),
        "no_hydrostatic_corruption": (5, 1.22),
    }

    for _, r in q.iterrows():

        if r["method"] not in label_methods:
            continue

        dx, ymul = offsets.get(
            r["method"],
            (5, 1.18),
        )

        x = float(r["rmse_mean"])
        y = max(
            float(r["rmse_std"]),
            0.25,
        )

        t = ax.text(
            x + dx,
            y * ymul,
            r["label"],
            fontsize=16.5,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=8,
        )

        stroke_text(t, 3.5)

    ax.set_yscale("log")

    ax.set_xlabel(
        "Mean preliminary stress RMSE ↓",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Across-seed standard deviation ↓  (log scale)",
        fontweight="bold",
    )

    ax.text(
        0.025,
        0.055,
        "better + more stable",
        transform=ax.transAxes,
        fontsize=19,
        fontweight="bold",
        color="#21795D",
        bbox=dict(
            facecolor=WHITE,
            edgecolor="#21795D",
            linewidth=1.15,
            boxstyle="round,pad=0.35",
        ),
    )

    handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color,
            edgecolor=BLACK,
            linewidth=0.8,
            label=fam,
        )
        for fam, color
        in FAMILY_COLORS.items()
    ]

    ax.legend(
        handles=handles,
        frameon=True,
        facecolor=WHITE,
        edgecolor=RED,
        framealpha=1,
        loc="upper left",
        fontsize=15,
    )

    save(
        fig,
        "fig10_preliminary_performance_stability",
    )


# =====================================================================
# FIG 11 — ABLATION-FAMILY HEATMAP
# =====================================================================

def minmax_good(values, lower_better=False):

    x = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    finite = x[np.isfinite(x)]

    if len(finite) == 0:
        return pd.Series(
            np.nan,
            index=x.index,
        )

    lo = finite.min()
    hi = finite.max()

    if abs(hi - lo) < 1e-12:
        z = pd.Series(
            0.5,
            index=x.index,
        )
    else:
        z = (x - lo) / (hi - lo)

    if lower_better:
        z = 1.0 - z

    return z


def fig11_heatmap():

    q = agg.copy()

    # Stability works better after log compression.
    q["std_log"] = np.log1p(
        q["rmse_std"].clip(lower=0)
    )

    q["Forecast"] = minmax_good(
        q["rmse_mean"],
        lower_better=True,
    )

    q["Stability"] = minmax_good(
        q["std_log"],
        lower_better=True,
    )

    q["MAE"] = minmax_good(
        q["mae_mean"],
        lower_better=True,
    )

    q["Localization"] = minmax_good(
        q["loc_mean"],
        lower_better=False,
    )

    q["Type-ID"] = minmax_good(
        q["type_mean"],
        lower_better=False,
    )

    family_order = [
        "Core / prediction",
        "Corruption ablations",
        "Constraint ablations",
        "Weighting / routing",
        "Heads / objective",
    ]

    q["family_ord"] = q[
        "family"
    ].map({
        x: i
        for i, x in enumerate(
            family_order
        )
    })

    q = (
        q
        .sort_values(
            [
                "family_ord",
                "rmse_mean",
            ]
        )
        .reset_index(drop=True)
    )

    metrics = [
        "Forecast",
        "Stability",
        "MAE",
        "Localization",
        "Type-ID",
    ]

    M = q[metrics].to_numpy(
        dtype=float
    )

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "screen_score",
        [
            "#F7F7FB",
            "#DCE7FF",
            "#AFC8F4",
            "#67A9CF",
            "#1C8B72",
        ],
    )

    fig, ax = plt.subplots(
        figsize=(15.5, 15.5),
        constrained_layout=True,
    )

    style_panel(
        ax,
        "Ablation-family screening map",
    )

    masked = np.ma.masked_invalid(M)

    ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # NaN cells white.
    ax.set_facecolor(WHITE)

    ax.set_xticks(
        np.arange(len(metrics))
    )

    ax.set_xticklabels(
        [
            "Forecast\nscore",
            "Stability\nscore",
            "MAE\nscore",
            "Localization\nsignal",
            "Type-ID\nsignal",
        ],
        fontsize=19,
        fontweight="bold",
    )

    ax.set_yticks(
        np.arange(len(q))
    )

    ax.set_yticklabels(
        q["label"],
        fontsize=17.5,
    )

    # black cell borders
    ax.set_xticks(
        np.arange(-0.5, len(metrics), 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, len(q), 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color=BLACK,
        linewidth=0.55,
        alpha=0.38,
    )

    ax.grid(
        which="major",
        visible=False,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # Values.
    for i in range(len(q)):
        for j in range(len(metrics)):

            v = M[i, j]

            if not np.isfinite(v):
                text = "—"
            else:
                text = f"{v:.2f}"

            t = ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=BLACK,
            )

            stroke_text(t, 2.6)

    # family separators and labels
    starts = []

    for fam in family_order:
        idx = q.index[
            q["family"] == fam
        ].tolist()

        if not idx:
            continue

        starts.append(
            (
                fam,
                min(idx),
                max(idx),
            )
        )

    for k, (
        fam,
        lo,
        hi,
    ) in enumerate(starts):

        color = FAMILY_COLORS[fam]

        # left family stripe
        ax.add_patch(
            Rectangle(
                (
                    -0.72,
                    lo - 0.47,
                ),
                0.12,
                hi - lo + 0.94,
                facecolor=color,
                edgecolor=BLACK,
                linewidth=0.7,
                clip_on=False,
                zorder=6,
            )
        )

        if lo > 0:
            ax.axhline(
                lo - 0.5,
                color=RED,
                linewidth=1.6,
                alpha=0.72,
            )

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(
                0,
                1,
            ),
            cmap=cmap,
        ),
        ax=ax,
        pad=0.025,
        fraction=0.035,
    )

    cbar.set_label(
        "Relative screening score  (1 = better within column)",
        fontsize=19,
        fontweight="bold",
    )

    cbar.ax.tick_params(
        labelsize=16,
    )

    ax.text(
        0.015,
        0.992,
        (
            "Scores are column-wise normalized; "
            "NaN means that the corresponding diagnostic head is absent."
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15.5,
        bbox=dict(
            facecolor=WHITE,
            edgecolor=RED,
            linewidth=1.0,
            boxstyle="round,pad=0.30",
        ),
    )

    save(
        fig,
        "fig11_preliminary_ablation_family_heatmap",
    )


# =====================================================================
# FIG 12 — PRELIMINARY → FINAL FUNNEL
# =====================================================================

def get_external_progress():

    ptr = (
        ROOT
        / "reports"
        / "iclr_extra_latest_reports.txt"
    )

    if not ptr.exists():
        return None, None

    try:
        report_dir = Path(
            ptr.read_text().strip()
        )

        finished = len(
            list(
                report_dir.glob("*.rc")
            )
        )

        success = 0

        for p in report_dir.glob("*.rc"):
            try:
                if p.read_text().strip() == "0":
                    success += 1
            except Exception:
                pass

        return finished, success

    except Exception:
        return None, None


def fig12_funnel():

    # Counts are deliberately structural, not claimed as independent datasets.
    stages = [
        (
            138,
            "Planned campaign",
            "train + eval jobs",
            "#D7DCE5",
        ),
        (
            66,
            "Learned runs screened",
            "31 reused + 35 bounded-budget completions",
            "#AFC8E8",
        ),
        (
            20,
            "ClimaX variants",
            "uniform 3-seed preliminary screen",
            "#7EA8DA",
        ),
        (
            4,
            "Full normalized finalists",
            "common clean + 60-condition stress protocol",
            "#547CC4",
        ),
        (
            3,
            "Pareto-optimal modes",
            "Acc · Direct state · Phys",
            "#3658A8",
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(14.5, 13.5),
        constrained_layout=True,
    )

    style_panel(
        ax,
        "Experimental funnel: broad preliminary search → focused final evaluation",
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.6, len(stages) + 0.65)
    ax.axis("off")

    # Restore red outer border after axis off.
    outer = Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor=RED,
        linewidth=1.55,
        zorder=50,
    )
    ax.add_patch(outer)

    max_count = stages[0][0]

    centers = []

    for i, (
        count,
        title,
        subtitle,
        color,
    ) in enumerate(stages):

        # sqrt scaling prevents last boxes from becoming unreadably tiny.
        width = (
            1.55
            * math.sqrt(
                count / max_count
            )
            + 0.30
        )

        height = 0.72
        y = (
            len(stages)
            - i
            - 0.35
        )

        centers.append(
            (0.0, y)
        )

        # faint halo
        ax.add_patch(
            FancyBboxPatch(
                (
                    -width / 2 - 0.045,
                    y - height / 2 - 0.045,
                ),
                width + 0.09,
                height + 0.09,
                boxstyle="round,pad=0.03",
                facecolor=color,
                edgecolor="none",
                alpha=0.09,
                zorder=1,
            )
        )

        # transparent segmented gradient
        steps = 40

        for s in range(steps):

            xx = (
                -width / 2
                + width
                * s
                / steps
            )

            a = (
                0.18
                + 0.67
                * (
                    1
                    - abs(
                        (s + 0.5)
                        / steps
                        - 0.5
                    )
                    * 1.7
                )
            )

            a = max(
                0.14,
                min(
                    a,
                    0.88,
                ),
            )

            ax.add_patch(
                Rectangle(
                    (
                        xx,
                        y - height / 2,
                    ),
                    width / steps * 1.04,
                    height,
                    facecolor=color,
                    edgecolor="none",
                    alpha=a,
                    zorder=2,
                )
            )

        ax.add_patch(
            FancyBboxPatch(
                (
                    -width / 2,
                    y - height / 2,
                ),
                width,
                height,
                boxstyle="round,pad=0.025",
                facecolor="none",
                edgecolor=BLACK,
                linewidth=1.15,
                zorder=4,
            )
        )

        t = ax.text(
            0,
            y + 0.11,
            f"{count}",
            ha="center",
            va="center",
            fontsize=27,
            fontweight="bold",
            color=BLACK,
            zorder=6,
        )
        stroke_text(t, 3)

        t = ax.text(
            0,
            y - 0.07,
            title,
            ha="center",
            va="center",
            fontsize=19,
            fontweight="bold",
            color=BLACK,
            zorder=6,
        )
        stroke_text(t, 2.5)

        ax.text(
            0,
            y - 0.25,
            subtitle,
            ha="center",
            va="center",
            fontsize=14.5,
            color=BLACK,
            zorder=6,
        )

        if i < len(stages) - 1:

            next_y = (
                len(stages)
                - (i + 1)
                - 0.35
            )

            ax.add_patch(
                FancyArrowPatch(
                    (
                        0,
                        y - height / 2 - 0.05,
                    ),
                    (
                        0,
                        next_y + height / 2 + 0.05,
                    ),
                    arrowstyle="-|>",
                    mutation_scale=19,
                    linewidth=1.4,
                    color=RED,
                    alpha=0.70,
                    zorder=3,
                )
            )

    finished, success = (
        get_external_progress()
    )

    if finished is not None:

        ax.text(
            0,
            -0.15,
            (
                "External confirmation layer: "
                f"{finished}/10 transfer/OOD jobs finished"
                + (
                    f" · {success} successful"
                    if success is not None
                    else ""
                )
            ),
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            bbox=dict(
                facecolor=WHITE,
                edgecolor=RED,
                linewidth=1.25,
                boxstyle="round,pad=0.42",
            ),
        )

    save(
        fig,
        "fig12_preliminary_to_final_funnel",
    )


# =====================================================================
# UPDATE INDEX
# =====================================================================

def update_index():

    index = OUT / "FIGURES_INDEX.txt"

    old = ""

    if index.exists():
        old = index.read_text()

    marker = "\nADDITIONAL PRELIMINARY-SCREEN FIGURES\n"

    if marker in old:
        old = old.split(marker)[0].rstrip()

    lines = [
        old,
        marker.strip(),
        "=" * 80,
        "fig09_preliminary_20method_ranking",
        "  20 ClimaX variants, mean +/- std over 3 preliminary seeds.",
        "",
        "fig10_preliminary_performance_stability",
        "  Mean stress screening performance versus cross-seed variability.",
        "",
        "fig11_preliminary_ablation_family_heatmap",
        "  Relative within-column scores across ablation families.",
        "",
        "fig12_preliminary_to_final_funnel",
        "  138-job campaign -> 66 learned runs -> 20 variants -> 4 finalists -> 3 Pareto modes.",
        "",
    ]

    index.write_text(
        "\n".join(
            x
            for x in lines
            if x is not None
        )
    )


# =====================================================================
# RUN
# =====================================================================

setup_style()

fig09_ranking()
fig10_stability()
fig11_heatmap()
fig12_funnel()

update_index()

print()
print("=" * 100)
print("ADDED TO WOW V2")
print("=" * 100)

for name in [
    "fig09_preliminary_20method_ranking",
    "fig10_preliminary_performance_stability",
    "fig11_preliminary_ablation_family_heatmap",
    "fig12_preliminary_to_final_funnel",
]:
    print(OUT / f"{name}.png")

print()
print("OUTPUT:", OUT)
