"""
Visualization Module for Geostress & Fracture Analysis.

Generates publication-quality plots:
- Stereonet projections (poles, density contours)
- Rose diagrams of fracture strike
- Mohr circles with fracture data
- Stress polygons
- Slip/dilation tendency plots
- Depth profiles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm

try:
    from data_loader import AZIMUTH_COL, DIP_COL, WELL_COL, FRACTURE_TYPE_COL, DEPTH_COL
except ImportError:
    from .data_loader import AZIMUTH_COL, DIP_COL, WELL_COL, FRACTURE_TYPE_COL, DEPTH_COL


# ──────────────────────────────────────────────
# Color scheme for fracture types
# ──────────────────────────────────────────────

FRAC_COLORS = {
    "Boundary": "#e41a1c",
    "Brecciated": "#377eb8",
    "Continuous": "#4daf4a",
    "Discontinuous": "#984ea3",
    "Vuggy": "#ff7f00",
}

def _get_color(frac_type):
    return FRAC_COLORS.get(frac_type, "#999999")


# ──────────────────────────────────────────────
# Rose Diagram
# ──────────────────────────────────────────────

def plot_rose_diagram(azimuths: np.ndarray, title: str = "Fracture Strike Rose Diagram",
                      bins: int = 36, ax=None) -> plt.Axes:
    """Plot a rose diagram of fracture strike directions."""
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    # Convert dip direction to strike (rotate 90° CCW)
    strikes = (azimuths - 90) % 360

    # Bidirectional: add 180° to each measurement
    strikes_bi = np.concatenate([strikes, (strikes + 180) % 360])

    bin_edges = np.linspace(0, 360, bins + 1)
    counts, _ = np.histogram(strikes_bi, bins=bin_edges)

    # Convert to radians for polar plot
    bin_centers = np.radians(0.5 * (bin_edges[:-1] + bin_edges[1:]))
    width = np.radians(360.0 / bins)

    ax.bar(bin_centers, counts, width=width, bottom=0,
           color="#377eb8", edgecolor="black", alpha=0.7, linewidth=0.5)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, pad=20, fontsize=12, fontweight="bold")

    return ax


# ──────────────────────────────────────────────
# Stereonet (Lower Hemisphere Equal-Area)
# ──────────────────────────────────────────────

def plot_stereonet(df: pd.DataFrame, title: str = "Fracture Poles Stereonet",
                   color_by: str = "fracture_type", ax=None):
    """Plot fracture poles on a lower-hemisphere equal-area stereonet."""
    try:
        import mplstereonet
    except ImportError:
        print("mplstereonet not available, using manual projection")
        return _plot_stereonet_manual(df, title, color_by, ax)

    if ax is None:
        fig, ax = mplstereonet.subplots(figsize=(7, 7))

    if color_by == "fracture_type":
        for ftype in df[FRACTURE_TYPE_COL].unique():
            mask = df[FRACTURE_TYPE_COL] == ftype
            strikes = (df.loc[mask, AZIMUTH_COL] - 90) % 360
            dips = df.loc[mask, DIP_COL]
            # Plot poles
            ax.pole(strikes, dips, marker="o", markersize=4,
                    color=_get_color(ftype), label=ftype, alpha=0.6)
    else:
        strikes = (df[AZIMUTH_COL] - 90) % 360
        dips = df[DIP_COL]
        ax.pole(strikes, dips, marker="o", markersize=4, color="#377eb8", alpha=0.6)

    # Add density contours
    strikes_all = (df[AZIMUTH_COL] - 90) % 360
    dips_all = df[DIP_COL]
    ax.density_contourf(strikes_all, dips_all, measurement="poles",
                         cmap="Reds", alpha=0.3)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.05, 1))
    ax.grid(True)

    return ax


def _plot_stereonet_manual(df, title, color_by, ax):
    """Fallback stereonet using equal-area projection without mplstereonet."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    # Draw primitive circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    # Equal-area (Schmidt) projection of poles
    az_rad = np.radians(df[AZIMUTH_COL].values)
    dip_rad = np.radians(df[DIP_COL].values)

    # Pole to plane: trend = az + 180, plunge = 90 - dip
    pole_trend = az_rad + np.pi
    pole_plunge = np.pi / 2 - dip_rad

    # Equal-area projection
    r = np.sqrt(2) * np.sin(pole_plunge / 2)
    x = r * np.sin(pole_trend)
    y = r * np.cos(pole_trend)

    if color_by == "fracture_type":
        for ftype in df[FRACTURE_TYPE_COL].unique():
            mask = df[FRACTURE_TYPE_COL] == ftype
            ax.scatter(x[mask], y[mask], s=15, color=_get_color(ftype),
                      label=ftype, alpha=0.6, edgecolors="none")
        ax.legend(fontsize=8)
    else:
        ax.scatter(x, y, s=15, color="#377eb8", alpha=0.6, edgecolors="none")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("E")
    ax.set_ylabel("N")

    # Cardinal directions
    for angle, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        rad = np.radians(angle)
        ax.text(1.15 * np.sin(rad), 1.15 * np.cos(rad), label,
               ha="center", va="center", fontsize=10, fontweight="bold")

    return ax


# ──────────────────────────────────────────────
# Mohr Circle
# ──────────────────────────────────────────────

def plot_mohr_circle(result: dict, title: str = "Mohr Circle with Fracture Data",
                     ax=None) -> plt.Axes:
    """Plot Mohr circles with fracture data points and failure envelope.

    Parameters
    ----------
    result : dict from geostress.invert_stress()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))

    s1, s2, s3 = result["sigma1"], result["sigma2"], result["sigma3"]
    mu = result["mu"]

    # Draw the three Mohr circles
    for si, sj, color, label in [
        (s1, s3, "#e41a1c", "σ1-σ3"),
        (s1, s2, "#4daf4a", "σ1-σ2"),
        (s2, s3, "#377eb8", "σ2-σ3"),
    ]:
        center = (si + sj) / 2
        radius = (si - sj) / 2
        circle = plt.Circle((center, 0), radius, fill=False, color=color,
                            linewidth=2, label=label)
        ax.add_patch(circle)

    # Plot fracture data points (σn, τ)
    ax.scatter(result["sigma_n"], result["tau"], s=10, c="#333333",
              alpha=0.5, zorder=5, label="Fractures")

    # Mohr-Coulomb failure envelope
    sn_range = np.linspace(0, s1 * 1.1, 100)
    tau_mc = mu * sn_range
    ax.plot(sn_range, tau_mc, "r--", linewidth=2, label=f"μ={mu:.2f}")

    ax.set_xlim(0, s1 * 1.15)
    ax.set_ylim(0, (s1 - s3) / 2 * 1.3)
    ax.set_xlabel("Normal Stress σn (MPa)", fontsize=11)
    ax.set_ylabel("Shear Stress τ (MPa)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return ax


# ──────────────────────────────────────────────
# Slip / Dilation Tendency
# ──────────────────────────────────────────────

def plot_tendency(df: pd.DataFrame, values: np.ndarray,
                  title: str = "Slip Tendency", cmap: str = "RdYlGn_r",
                  ax=None) -> plt.Axes:
    """Plot tendency values on a polar stereonet-style plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # Primitive circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1)

    # Project poles
    az_rad = np.radians(df[AZIMUTH_COL].values)
    dip_rad = np.radians(df[DIP_COL].values)
    pole_trend = az_rad + np.pi
    pole_plunge = np.pi / 2 - dip_rad
    r = np.sqrt(2) * np.sin(pole_plunge / 2)
    x = r * np.sin(pole_trend)
    y = r * np.cos(pole_trend)

    sc = ax.scatter(x, y, c=values, s=20, cmap=cmap, alpha=0.8,
                   edgecolors="black", linewidth=0.3, vmin=0, vmax=max(1, values.max()))

    plt.colorbar(sc, ax=ax, label=title, shrink=0.7)
    ax.set_title(title, fontsize=12, fontweight="bold")

    for angle, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        rad = np.radians(angle)
        ax.text(1.15 * np.sin(rad), 1.15 * np.cos(rad), label,
               ha="center", va="center", fontsize=10, fontweight="bold")

    return ax


# ──────────────────────────────────────────────
# Depth Profile
# ──────────────────────────────────────────────

def plot_depth_profile(df: pd.DataFrame, title: str = "Fractures vs Depth",
                       ax=None) -> plt.Axes:
    """Plot fracture azimuth and dip vs depth (tadpole plot style)."""
    df_depth = df.dropna(subset=[DEPTH_COL])
    if len(df_depth) == 0:
        return None

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    else:
        axes = [ax, ax.twinx()]

    # Azimuth vs depth
    for ftype in df_depth[FRACTURE_TYPE_COL].unique():
        mask = df_depth[FRACTURE_TYPE_COL] == ftype
        axes[0].scatter(df_depth.loc[mask, AZIMUTH_COL],
                       df_depth.loc[mask, DEPTH_COL],
                       s=10, color=_get_color(ftype), label=ftype, alpha=0.6)

    axes[0].set_xlabel("Azimuth (°)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_xlim(0, 360)
    axes[0].invert_yaxis()
    axes[0].set_title("Azimuth vs Depth")
    axes[0].legend(fontsize=7, loc="lower left")

    # Dip vs depth
    for ftype in df_depth[FRACTURE_TYPE_COL].unique():
        mask = df_depth[FRACTURE_TYPE_COL] == ftype
        axes[1].scatter(df_depth.loc[mask, DIP_COL],
                       df_depth.loc[mask, DEPTH_COL],
                       s=10, color=_get_color(ftype), alpha=0.6)

    axes[1].set_xlabel("Dip (°)")
    axes[1].set_xlim(0, 90)
    axes[1].invert_yaxis()
    axes[1].set_title("Dip vs Depth")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    return axes


# ──────────────────────────────────────────────
# Summary Dashboard
# ──────────────────────────────────────────────

def plot_analysis_dashboard(df: pd.DataFrame, inversion_result: dict,
                            well_name: str = "", save_path: str = None):
    """Create a 2x2 summary dashboard."""
    fig = plt.figure(figsize=(14, 12))

    # Rose diagram
    ax1 = fig.add_subplot(2, 2, 1, projection="polar")
    plot_rose_diagram(df[AZIMUTH_COL].values,
                     title=f"Strike Rose - {well_name}", ax=ax1)

    # Stereonet
    ax2 = fig.add_subplot(2, 2, 2)
    _plot_stereonet_manual(df, f"Fracture Poles - {well_name}", "fracture_type", ax2)

    # Mohr circle
    ax3 = fig.add_subplot(2, 2, 3)
    plot_mohr_circle(inversion_result, f"Mohr Circle - {well_name}", ax3)

    # Slip tendency
    ax4 = fig.add_subplot(2, 2, 4)
    plot_tendency(df, inversion_result["slip_tend"],
                 f"Slip Tendency - {well_name}", ax=ax4)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Dashboard saved to {save_path}")

    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from data_loader import load_all_fractures, fracture_plane_normal
    from geostress import invert_stress

    df = load_all_fractures("../data/raw")

    for well in df[WELL_COL].unique():
        df_well = df[df[WELL_COL] == well].reset_index(drop=True)
        normals = fracture_plane_normal(df_well[AZIMUTH_COL].values,
                                        df_well[DIP_COL].values)
        avg_depth = df_well[DEPTH_COL].mean()
        if np.isnan(avg_depth):
            avg_depth = 3300.0

        result = invert_stress(normals, regime="strike_slip", depth_m=avg_depth)

        fig = plot_analysis_dashboard(df_well, result, well_name=f"Well {well}",
                                      save_path=f"../outputs/dashboard_{well}.png")
        plt.close(fig)

    print("Dashboards saved to outputs/")


# ──────────────────────────────────────────────
# Model Comparison Charts
# ──────────────────────────────────────────────

def plot_model_comparison(ranking_data: list, title: str = "Model Comparison") -> plt.Figure:
    """Bar chart comparing model accuracies and balanced accuracies.

    Parameters
    ----------
    ranking_data : list of dicts with keys: model, cv_accuracy_mean, balanced_accuracy
    title : chart title

    Returns
    -------
    matplotlib Figure
    """
    if not ranking_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No model data", ha="center", va="center")
        return fig

    models = [d.get("model", "?") for d in ranking_data]
    acc = [d.get("cv_accuracy_mean", 0) * 100 for d in ranking_data]
    bal = [d.get("balanced_accuracy", 0) * 100 for d in ranking_data]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, acc, width, label="Standard Accuracy",
                   color="#3b82f6", alpha=0.8)
    bars2 = ax.bar(x + width / 2, bal, width, label="Balanced Accuracy",
                   color="#f59e0b", alpha=0.8)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.axhline(y=90, color="#16a34a", linestyle="--", alpha=0.3, label="90% target")
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


def plot_learning_curve(train_sizes, train_scores, val_scores,
                        balanced_scores=None,
                        title="Learning Curve") -> plt.Figure:
    """Line chart showing accuracy vs training set size.

    Parameters
    ----------
    train_sizes : list of ints
    train_scores : list of floats (0-1)
    val_scores : list of floats (0-1)
    balanced_scores : optional list of balanced accuracy scores
    title : chart title

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(train_sizes, [s * 100 for s in train_scores],
            "o-", color="#3b82f6", label="Training", linewidth=2, markersize=6)
    ax.plot(train_sizes, [s * 100 for s in val_scores],
            "s-", color="#dc2626", label="Validation", linewidth=2, markersize=6)
    if balanced_scores:
        ax.plot(train_sizes, [s * 100 for s in balanced_scores],
                "^-", color="#16a34a", label="Balanced (Val)", linewidth=2, markersize=6)

    # Fill between train and val to show overfitting gap
    ax.fill_between(train_sizes,
                    [s * 100 for s in train_scores],
                    [s * 100 for s in val_scores],
                    alpha=0.1, color="#6366f1")

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Annotate the gap
    if len(train_scores) > 0:
        gap = (train_scores[-1] - val_scores[-1]) * 100
        ax.annotate(f"Gap: {gap:.1f}%",
                    xy=(train_sizes[-1], val_scores[-1] * 100),
                    xytext=(train_sizes[-1] * 0.7, val_scores[-1] * 100 - 10),
                    arrowprops=dict(arrowstyle="->", color="#6366f1"),
                    fontsize=9, color="#6366f1")

    fig.tight_layout()
    return fig


def plot_bootstrap_ci(class_names, per_class_data,
                      title="Per-Class F1 with 95% CI") -> plt.Figure:
    """Horizontal bar chart with error bars showing bootstrap CIs.

    Parameters
    ----------
    class_names : list of str
    per_class_data : dict of {class: {f1: {mean, ci_low, ci_high}}}
    title : chart title

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    y_pos = np.arange(len(class_names))
    means = []
    lows = []
    highs = []

    for cls in class_names:
        f1 = per_class_data.get(cls, {}).get("f1")
        if f1:
            means.append(f1["mean"] * 100)
            lows.append(f1["mean"] * 100 - f1["ci_low"] * 100)
            highs.append(f1["ci_high"] * 100 - f1["mean"] * 100)
        else:
            means.append(0)
            lows.append(0)
            highs.append(0)

    colors = ["#dc2626" if m < 30 else "#d97706" if m < 60 else "#16a34a"
              for m in means]

    ax.barh(y_pos, means, xerr=[lows, highs], align="center",
            color=colors, alpha=0.8, capsize=5, ecolor="#374151")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("F1 Score (%)")
    ax.set_title(title)
    ax.set_xlim(0, 105)
    ax.axvline(x=50, color="#6b7280", linestyle="--", alpha=0.3)
    ax.grid(axis="x", alpha=0.3)

    # Value labels
    for i, (m, l, h) in enumerate(zip(means, lows, highs)):
        ax.text(min(m + h + 2, 100), i,
                f"{m:.1f}% [{m - l:.0f}-{m + h:.0f}]",
                va="center", fontsize=8)

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names,
                          title="Confusion Matrix — Misclassification Analysis") -> plt.Figure:
    """Annotated heatmap of the confusion matrix.

    Parameters
    ----------
    cm : list of lists or 2D array
        Confusion matrix (rows=true, cols=predicted).
    class_names : list of str
        Class labels in the same order as cm rows/cols.
    title : str
        Chart title.

    Returns
    -------
    matplotlib Figure
    """
    cm = np.asarray(cm, dtype=float)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))

    # Normalize per row (recall-based) for colour intensity
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_norm = cm / row_sums

    im = ax.imshow(cm_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Annotate each cell with raw count + percentage
    for i in range(n):
        for j in range(n):
            val = int(cm[i, j])
            pct = cm_norm[i, j] * 100
            text_color = "white" if cm_norm[i, j] > 0.6 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=9,
                    color=text_color, fontweight=weight)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Recall (row-normalized)", fontsize=9)

    fig.tight_layout()
    return fig


def plot_abstention_chart(confidence_distribution: list,
                          threshold: float = 0.60,
                          abstention_rate: float = 0.0,
                          accuracy_overall: float = 0.0,
                          accuracy_confident: float = 0.0,
                          title: str = "Prediction Abstention — Confidence Distribution") -> plt.Figure:
    """Bar chart of confidence distribution with abstention threshold line.

    Parameters
    ----------
    confidence_distribution : list of dicts with 'range' and 'count'
    threshold : float, abstention threshold
    accuracy_overall, accuracy_confident : floats for annotation

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ranges = [d["range"] for d in confidence_distribution]
    counts = [d["count"] for d in confidence_distribution]

    # Color bars: red below threshold, green above
    colors = []
    for r in ranges:
        upper = float(r.split("-")[1])
        if upper <= threshold:
            colors.append("#dc3545")  # red — abstained
        elif float(r.split("-")[0]) < threshold:
            colors.append("#ffc107")  # amber — borderline
        else:
            colors.append("#198754")  # green — confident

    bars = ax.bar(range(len(ranges)), counts, color=colors, edgecolor="white", linewidth=0.5)

    # Threshold line
    for i, r in enumerate(ranges):
        lo = float(r.split("-")[0])
        hi = float(r.split("-")[1])
        if lo <= threshold <= hi:
            frac = (threshold - lo) / (hi - lo)
            x_pos = i - 0.5 + frac
            ax.axvline(x=x_pos, color="#333", linestyle="--", linewidth=2, alpha=0.8)
            ax.text(x_pos + 0.1, max(counts) * 0.9,
                    f"Threshold: {threshold:.0%}",
                    fontsize=10, fontweight="bold", color="#333")
            break

    # Value labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                    str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(ranges)))
    ax.set_xticklabels(ranges, fontsize=9)
    ax.set_xlabel("Max Predicted Probability", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Legend annotation
    legend_text = (
        f"Abstention rate: {abstention_rate:.1f}%\n"
        f"Overall accuracy: {accuracy_overall:.1%}\n"
        f"Confident-only accuracy: {accuracy_confident:.1%}\n"
        f"Accuracy gain: {accuracy_confident - accuracy_overall:+.1%}"
    )
    ax.text(0.98, 0.95, legend_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_sensitivity_heatmap(
    friction_values: list,
    pp_values: list,
    cs_matrix: list,
    title: str = "Sensitivity Heatmap — Critically Stressed %",
) -> plt.Figure:
    """2D heatmap: friction vs pore pressure, colored by critically stressed %.

    Parameters
    ----------
    friction_values : list of floats (x-axis)
    pp_values : list of floats (y-axis, in MPa)
    cs_matrix : 2D list [len(pp)][len(friction)] of critically stressed %
    title : str

    Returns
    -------
    matplotlib Figure
    """
    Z = np.array(cs_matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(Z, cmap="RdYlGn_r", aspect="auto",
                   extent=[friction_values[0], friction_values[-1],
                           pp_values[-1], pp_values[0]],
                   vmin=0, vmax=100)

    # Contour lines for risk thresholds
    X, Y = np.meshgrid(friction_values, pp_values)
    cs10 = ax.contour(X, Y, Z, levels=[10], colors=["green"], linewidths=2, linestyles="--")
    cs30 = ax.contour(X, Y, Z, levels=[30], colors=["orange"], linewidths=2, linestyles="--")
    cs50 = ax.contour(X, Y, Z, levels=[50], colors=["red"], linewidths=2, linestyles="-")

    ax.clabel(cs10, fmt="10%% (GREEN)", fontsize=8)
    ax.clabel(cs30, fmt="30%% (AMBER)", fontsize=8)
    ax.clabel(cs50, fmt="50%% (RED)", fontsize=8)

    # Annotate cells
    for i in range(len(pp_values)):
        for j in range(len(friction_values)):
            val = Z[i, j]
            text_color = "white" if val > 50 else "black"
            ax.text(friction_values[j], pp_values[i], f"{val:.0f}%",
                    ha="center", va="center", fontsize=7,
                    color=text_color, fontweight="bold")

    ax.set_xlabel("Friction Coefficient (μ)", fontsize=11)
    ax.set_ylabel("Pore Pressure (MPa)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Critically Stressed %", fontsize=10)

    # Risk legend
    legend_text = "Risk zones:\n  GREEN: <10%\n  AMBER: 10-30%\n  RED: >30%"
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    fig.tight_layout()
    return fig


def plot_batch_comparison(well_results: dict, title: str = "Field Comparison") -> plt.Figure:
    """Create a multi-panel comparison chart for batch well analysis.

    Shows per-well SHmax, accuracy, and critically stressed % in a
    compact dashboard layout for stakeholder field-level assessment.

    Parameters
    ----------
    well_results : dict mapping well name -> {stress, classification, risk}
    title : Chart title

    Returns
    -------
    matplotlib.Figure
    """
    wells = list(well_results.keys())
    n = len(wells)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No well data available", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    risk_colors = {"GREEN": "#38a169", "AMBER": "#d69e2e", "RED": "#e53e3e"}
    bar_x = range(n)

    # Panel 1: SHmax azimuth
    ax = axes[0]
    shmax_vals = []
    for w in wells:
        s = well_results[w].get("stress", {})
        shmax_vals.append(s.get("shmax", 0) if "error" not in s else 0)
    ax.bar(bar_x, shmax_vals, color="#3182ce", alpha=0.85, edgecolor="white")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(wells, fontsize=10)
    ax.set_ylabel("SHmax Azimuth (deg)", fontsize=10)
    ax.set_title("Maximum Horizontal Stress", fontsize=11, fontweight="bold")
    for i, v in enumerate(shmax_vals):
        if v > 0:
            ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(shmax_vals + [180]) * 1.15)

    # Panel 2: Classification accuracy
    ax = axes[1]
    acc_vals = []
    for w in wells:
        c = well_results[w].get("classification", {})
        acc_vals.append(c.get("accuracy", 0) * 100 if "error" not in c else 0)
    colors_acc = ["#38a169" if v >= 70 else "#d69e2e" if v >= 50 else "#e53e3e" for v in acc_vals]
    ax.bar(bar_x, acc_vals, color=colors_acc, alpha=0.85, edgecolor="white")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(wells, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("Fracture Classification", fontsize=11, fontweight="bold")
    ax.axhline(70, color="#38a169", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(n - 0.5, 71, "70% target", fontsize=8, color="#38a169", ha="right")
    for i, v in enumerate(acc_vals):
        if v > 0:
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 110)

    # Panel 3: Critically stressed %
    ax = axes[2]
    cs_vals = []
    cs_colors = []
    for w in wells:
        r = well_results[w].get("risk", {})
        pct = r.get("pct_critically_stressed", 0) if "error" not in r else 0
        cs_vals.append(pct)
        level = r.get("risk_level", "GREEN") if "error" not in r else "GREEN"
        cs_colors.append(risk_colors.get(level, "#718096"))
    ax.bar(bar_x, cs_vals, color=cs_colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(wells, fontsize=10)
    ax.set_ylabel("Critically Stressed (%)", fontsize=10)
    ax.set_title("Risk Assessment", fontsize=11, fontweight="bold")
    ax.axhline(10, color="#38a169", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(30, color="#d69e2e", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(n - 0.5, 11, "GREEN/AMBER", fontsize=7, color="#38a169", ha="right")
    ax.text(n - 0.5, 31, "AMBER/RED", fontsize=7, color="#d69e2e", ha="right")
    for i, v in enumerate(cs_vals):
        if v > 0:
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(cs_vals + [40]) * 1.2)

    fig.tight_layout()
    return fig
