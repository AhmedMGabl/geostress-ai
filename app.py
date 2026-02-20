"""GeoStress AI - FastAPI Web Application."""

import os
import io
import base64
import asyncio
import threading
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse

from src.data_loader import (
    load_all_fractures, load_single_file, fracture_summary,
    fracture_plane_normal, AZIMUTH_COL, DIP_COL, DEPTH_COL,
    WELL_COL, FRACTURE_TYPE_COL,
)
from src.geostress import invert_stress
from src.fracture_analysis import (
    classify_fracture_types, cluster_fracture_sets, identify_critically_stressed,
)
from src.visualization import (
    plot_rose_diagram, _plot_stereonet_manual,
    plot_mohr_circle, plot_tendency, plot_depth_profile,
    plot_analysis_dashboard,
)

# ── Globals ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
plot_lock = threading.Lock()

# App state
demo_df: pd.DataFrame = None
uploaded_df: pd.DataFrame = None


# ── Helpers ──────────────────────────────────────────

def fig_to_base64(fig, dpi=120) -> str:
    """Serialize a matplotlib figure to a base64 data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def render_plot(plot_func, *args, **kwargs) -> str | None:
    """Thread-safe wrapper: call a plot function and return base64 image."""
    with plot_lock:
        result = plot_func(*args, **kwargs)
        if result is None:
            return None
        if isinstance(result, plt.Figure):
            fig = result
        elif isinstance(result, np.ndarray):
            fig = result.flat[0].figure
        elif isinstance(result, list):
            fig = result[0].figure
        else:
            fig = result.figure
        return fig_to_base64(fig)


def get_df(source: str = "demo") -> pd.DataFrame:
    """Get the active DataFrame based on source."""
    global demo_df, uploaded_df
    if source == "uploaded" and uploaded_df is not None:
        return uploaded_df
    return demo_df


# ── App lifecycle ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global demo_df
    demo_df = load_all_fractures(str(DATA_DIR))
    print(f"Loaded {len(demo_df)} demo fractures from {DATA_DIR}")
    yield


app = FastAPI(title="GeoStress AI", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Page routes ──────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── Data API ─────────────────────────────────────────

@app.get("/api/data/summary")
async def data_summary(source: str = "demo"):
    df = get_df(source)
    summary = fracture_summary(df)
    summary_reset = summary.reset_index()
    rows = summary_reset.to_dict(orient="records")
    # Replace NaN with None for JSON
    for row in rows:
        for k, v in row.items():
            if isinstance(v, float) and np.isnan(v):
                row[k] = None
    return {
        "total_fractures": len(df),
        "wells": df[WELL_COL].unique().tolist(),
        "fracture_types": df[FRACTURE_TYPE_COL].unique().tolist(),
        "summary": rows,
    }


@app.get("/api/data/wells")
async def list_wells(source: str = "demo"):
    df = get_df(source)
    wells = []
    for w in df[WELL_COL].unique():
        dw = df[df[WELL_COL] == w]
        avg_depth = dw[DEPTH_COL].mean()
        wells.append({
            "name": w,
            "count": len(dw),
            "avg_depth": round(float(avg_depth), 1) if not np.isnan(avg_depth) else 3300.0,
        })
    return {"wells": wells}


@app.post("/api/data/upload")
async def upload_file(file: UploadFile = File(...)):
    global uploaded_df
    if not file.filename.endswith((".xls", ".xlsx")):
        raise HTTPException(400, "Only .xls and .xlsx files supported")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        new_df = load_single_file(tmp_path)
        uploaded_df = new_df
        return {
            "filename": file.filename,
            "rows": len(new_df),
            "wells": new_df[WELL_COL].unique().tolist(),
            "fracture_types": new_df[FRACTURE_TYPE_COL].unique().tolist(),
            "source": "uploaded",
        }
    finally:
        os.unlink(tmp_path)


# ── Visualization API ────────────────────────────────

@app.get("/api/viz/rose")
async def viz_rose(well: str = "3P", source: str = "demo"):
    df = get_df(source)
    df_well = df[df[WELL_COL] == well]
    if len(df_well) == 0:
        raise HTTPException(404, f"No data for well {well}")

    img = await asyncio.to_thread(
        render_plot, plot_rose_diagram,
        df_well[AZIMUTH_COL].values,
        f"Well {well} ({len(df_well)} fractures)"
    )
    return {"image": img, "well": well, "count": len(df_well)}


@app.get("/api/viz/stereonet")
async def viz_stereonet(well: str = "3P", source: str = "demo"):
    df = get_df(source)
    df_well = df[df[WELL_COL] == well].reset_index(drop=True)
    if len(df_well) == 0:
        raise HTTPException(404, f"No data for well {well}")

    def make_stereonet():
        with plot_lock:
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            _plot_stereonet_manual(df_well, f"Fracture Poles - Well {well}", "fracture_type", ax)
            fig.tight_layout()
            return fig_to_base64(fig)

    img = await asyncio.to_thread(make_stereonet)
    return {"image": img, "well": well}


@app.get("/api/viz/depth-profile")
async def viz_depth_profile(source: str = "demo"):
    df = get_df(source)
    df_with_depth = df.dropna(subset=[DEPTH_COL])
    if len(df_with_depth) == 0:
        return {"image": None, "message": "No depth data available"}

    img = await asyncio.to_thread(
        render_plot, plot_depth_profile,
        df_with_depth, "Fracture Distribution vs Depth"
    )
    return {"image": img}


# ── Analysis API ─────────────────────────────────────

@app.post("/api/analysis/inversion")
async def run_inversion(request: Request):
    body = await request.json()
    well = body.get("well", "3P")
    regime = body.get("regime", "strike_slip")
    depth_m = float(body.get("depth_m", 3300.0))
    cohesion = float(body.get("cohesion", 0.0))
    source = body.get("source", "demo")

    df = get_df(source)
    df_well = df[df[WELL_COL] == well].reset_index(drop=True)
    if len(df_well) == 0:
        raise HTTPException(404, f"No data for well {well}")

    normals = fracture_plane_normal(
        df_well[AZIMUTH_COL].values, df_well[DIP_COL].values
    )
    avg_depth = df_well[DEPTH_COL].mean()
    if np.isnan(avg_depth):
        avg_depth = depth_m

    # Run inversion in thread
    result = await asyncio.to_thread(
        invert_stress, normals, regime=regime, depth_m=avg_depth, cohesion=cohesion
    )

    # Generate plots
    mohr_img = await asyncio.to_thread(
        render_plot, plot_mohr_circle, result, f"Mohr Circle - Well {well}"
    )
    slip_img = await asyncio.to_thread(
        render_plot, plot_tendency, df_well, result["slip_tend"],
        f"Slip Tendency - Well {well}", "RdYlGn_r"
    )
    dilation_img = await asyncio.to_thread(
        render_plot, plot_tendency, df_well, result["dilation_tend"],
        f"Dilation Tendency - Well {well}", "RdYlBu_r"
    )

    # Dashboard
    def make_dashboard():
        with plot_lock:
            fig = plot_analysis_dashboard(df_well, result, well_name=f"Well {well}")
            return fig_to_base64(fig, dpi=100)

    dashboard_img = await asyncio.to_thread(make_dashboard)

    cs_mask = identify_critically_stressed(
        result["sigma_n"], result["tau"], result["mu"], cohesion
    )

    return {
        "sigma1": round(float(result["sigma1"]), 2),
        "sigma2": round(float(result["sigma2"]), 2),
        "sigma3": round(float(result["sigma3"]), 2),
        "R": round(float(result["R"]), 4),
        "shmax_azimuth_deg": round(float(result["shmax_azimuth_deg"]), 1),
        "mu": round(float(result["mu"]), 4),
        "regime": regime,
        "fracture_count": len(df_well),
        "mohr_circle_img": mohr_img,
        "slip_tendency_img": slip_img,
        "dilation_tendency_img": dilation_img,
        "dashboard_img": dashboard_img,
        "critically_stressed_count": int(cs_mask.sum()),
        "critically_stressed_total": len(cs_mask),
        "critically_stressed_pct": round(100 * float(cs_mask.sum()) / len(cs_mask), 1),
    }


@app.post("/api/analysis/classify")
async def run_classification(request: Request):
    body = await request.json()
    classifier = body.get("classifier", "random_forest")
    source = body.get("source", "demo")

    df = get_df(source)

    clf_result = await asyncio.to_thread(
        classify_fracture_types, df, classifier=classifier
    )

    class_names = clf_result["label_encoder"].classes_.tolist()
    cm = clf_result["confusion_matrix"].tolist()
    feat_imp = {k: round(float(v), 4) for k, v in clf_result["feature_importances"].items()}

    return {
        "cv_mean_accuracy": round(float(clf_result["cv_mean_accuracy"]), 4),
        "cv_std_accuracy": round(float(clf_result["cv_std_accuracy"]), 4),
        "feature_importances": feat_imp,
        "confusion_matrix": cm,
        "class_names": class_names,
    }


@app.post("/api/analysis/cluster")
async def run_clustering(request: Request):
    body = await request.json()
    well = body.get("well", "3P")
    n_clusters = body.get("n_clusters", None)
    source = body.get("source", "demo")

    df = get_df(source)
    df_well = df[df[WELL_COL] == well].reset_index(drop=True)
    if len(df_well) == 0:
        raise HTTPException(404, f"No data for well {well}")

    clust = await asyncio.to_thread(
        cluster_fracture_sets, df_well, n_clusters=n_clusters
    )

    # Generate cluster plot
    def make_cluster_plot():
        with plot_lock:
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))
            ax.set_aspect("equal")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), "k-")

            az_rad = np.radians(df_well[AZIMUTH_COL].values)
            dip_rad = np.radians(df_well[DIP_COL].values)
            pole_trend = az_rad + np.pi
            pole_plunge = np.pi / 2 - dip_rad
            r = np.sqrt(2) * np.sin(pole_plunge / 2)
            x = r * np.sin(pole_trend)
            y = r * np.cos(pole_trend)

            colors = plt.cm.Set1(np.linspace(0, 1, clust["n_clusters"]))
            for c in range(clust["n_clusters"]):
                mask = clust["labels"] == c
                stats = clust["cluster_stats"].iloc[c]
                ax.scatter(
                    x[mask], y[mask], s=15, c=[colors[c]], alpha=0.6,
                    label=f'Set {c}: az={stats["mean_azimuth"]:.0f}, '
                          f'dip={stats["mean_dip"]:.0f} ({stats["count"]:.0f})'
                )

            ax.set_title(f'Fracture Sets - Well {well} ({clust["n_clusters"]} sets)')
            ax.legend(fontsize=8, loc="upper left")
            for angle, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
                rad = np.radians(angle)
                ax.text(1.15 * np.sin(rad), 1.15 * np.cos(rad), label,
                        ha="center", va="center", fontweight="bold")
            fig.tight_layout()
            return fig_to_base64(fig)

    cluster_img = await asyncio.to_thread(make_cluster_plot)

    stats = clust["cluster_stats"].reset_index(drop=True).to_dict(orient="records")
    for row in stats:
        for k, v in row.items():
            if isinstance(v, (np.integer, np.floating)):
                row[k] = round(float(v), 2)

    return {
        "n_clusters": int(clust["n_clusters"]),
        "cluster_stats": stats,
        "cluster_img": cluster_img,
        "well": well,
    }
