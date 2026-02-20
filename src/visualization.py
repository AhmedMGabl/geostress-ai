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
