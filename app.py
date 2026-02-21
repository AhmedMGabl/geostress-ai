"""GeoStress AI - FastAPI Web Application (v2.0 - Industrial Grade)."""

import os
import io
import base64
import asyncio
import threading
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from functools import lru_cache

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
from src.geostress import invert_stress, bayesian_inversion
from src.fracture_analysis import (
    classify_fracture_types, cluster_fracture_sets, identify_critically_stressed,
)
from src.enhanced_analysis import (
    compare_models, classify_enhanced, cluster_enhanced,
    critically_stressed_enhanced, generate_interpretation,
    compute_pore_pressure, feedback_store,
    engineer_enhanced_features, compute_shap_explanations,
    validate_data_quality, retrain_with_corrections,
    sensitivity_analysis, compute_risk_matrix,
    generate_well_report, compare_wells,
    compute_uncertainty_budget,
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

# Cache for expensive computations
_model_comparison_cache = {}
_inversion_cache = {}


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


def _sanitize_for_json(obj):
    """Recursively convert numpy types to Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── App lifecycle ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global demo_df
    demo_df = load_all_fractures(str(DATA_DIR))
    print(f"Loaded {len(demo_df)} demo fractures from {DATA_DIR}")
    yield


app = FastAPI(title="GeoStress AI", version="2.3.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Global error handler for production safety ───────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled errors and return a clean JSON response.

    Never expose raw tracebacks in production.
    """
    import traceback
    traceback.print_exc()  # Log to server console for debugging
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)[:200],  # Truncate to avoid leaking details
            "suggestion": "Try again or contact support if the issue persists.",
        },
    )


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


@app.get("/api/data/quality")
async def data_quality(source: str = "demo"):
    """Run data quality validation checks."""
    df = get_df(source)
    return _sanitize_for_json(validate_data_quality(df))


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
    global uploaded_df, _model_comparison_cache, _inversion_cache
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
        # Clear caches when new data is uploaded
        _model_comparison_cache.clear()
        _inversion_cache.clear()
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


# ── Analysis API (Original - kept for backward compat) ──

@app.post("/api/analysis/inversion")
async def run_inversion(request: Request):
    body = await request.json()
    well = body.get("well", "3P")
    regime = body.get("regime", "strike_slip")
    depth_m = float(body.get("depth_m", 3300.0))
    cohesion = float(body.get("cohesion", 0.0))
    source = body.get("source", "demo")
    pore_pressure = body.get("pore_pressure", None)
    if pore_pressure is not None:
        pore_pressure = float(pore_pressure)

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

    # Run inversion with pore pressure support
    result = await asyncio.to_thread(
        invert_stress, normals, regime=regime, depth_m=avg_depth,
        cohesion=cohesion, pore_pressure=pore_pressure
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

    # Enhanced critically stressed analysis with pore pressure
    pp = result.get("pore_pressure", 0.0)
    cs_result = critically_stressed_enhanced(
        result["sigma_n"], result["tau"], result["mu"], cohesion, pp
    )

    # Generate stakeholder interpretation
    interpretation = generate_interpretation(result, cs_result, well)

    return _sanitize_for_json({
        "sigma1": round(float(result["sigma1"]), 2),
        "sigma2": round(float(result["sigma2"]), 2),
        "sigma3": round(float(result["sigma3"]), 2),
        "R": round(float(result["R"]), 4),
        "shmax_azimuth_deg": round(float(result["shmax_azimuth_deg"]), 1),
        "mu": round(float(result["mu"]), 4),
        "regime": regime,
        "fracture_count": len(df_well),
        "pore_pressure_mpa": round(pp, 2),
        "mohr_circle_img": mohr_img,
        "slip_tendency_img": slip_img,
        "dilation_tendency_img": dilation_img,
        "dashboard_img": dashboard_img,
        "critically_stressed_count": cs_result["count_critical"],
        "critically_stressed_total": cs_result["total"],
        "critically_stressed_pct": cs_result["pct_critical"],
        "risk_level": cs_result.get("high_risk_count", 0),
        "risk_categories": {
            "high": cs_result["high_risk_count"],
            "moderate": cs_result["moderate_risk_count"],
            "low": cs_result["low_risk_count"],
        },
        "interpretation": interpretation,
    })


@app.post("/api/analysis/classify")
async def run_classification(request: Request):
    body = await request.json()
    classifier = body.get("classifier", "random_forest")
    source = body.get("source", "demo")
    use_enhanced = body.get("enhanced", True)

    df = get_df(source)

    if use_enhanced:
        clf_result = await asyncio.to_thread(
            classify_enhanced, df, classifier=classifier
        )
    else:
        clf_result = await asyncio.to_thread(
            classify_fracture_types, df, classifier=classifier
        )

    class_names = clf_result.get("class_names",
                                  clf_result.get("label_encoder", {}).classes_.tolist()
                                  if hasattr(clf_result.get("label_encoder", {}), "classes_")
                                  else [])
    cm = clf_result["confusion_matrix"]
    if hasattr(cm, "tolist"):
        cm = cm.tolist()
    feat_imp = {k: round(float(v), 4)
                for k, v in clf_result["feature_importances"].items()}

    return _sanitize_for_json({
        "cv_mean_accuracy": round(float(clf_result["cv_mean_accuracy"]), 4),
        "cv_std_accuracy": round(float(clf_result["cv_std_accuracy"]), 4),
        "cv_f1_mean": round(float(clf_result.get("cv_f1_mean", 0)), 4),
        "feature_importances": feat_imp,
        "confusion_matrix": cm,
        "class_names": class_names,
    })


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


# ── NEW: Multi-Model Comparison API ─────────────────

@app.post("/api/analysis/compare-models")
async def compare_all_models(request: Request):
    """Compare all available ML models on fracture classification.

    Pass fast=true for quicker results (~3x speedup, <0.5% accuracy loss).
    """
    body = await request.json()
    source = body.get("source", "demo")
    fast = body.get("fast", False)

    df = get_df(source)

    cache_key = f"{source}_{len(df)}_{'fast' if fast else 'full'}"
    if cache_key in _model_comparison_cache:
        return _model_comparison_cache[cache_key]

    result = await asyncio.to_thread(compare_models, df, fast=fast)
    response = _sanitize_for_json(result)
    _model_comparison_cache[cache_key] = response
    return response


# ── NEW: Feedback API ────────────────────────────────

@app.post("/api/feedback/submit")
async def submit_feedback(request: Request):
    """Submit expert feedback on analysis results."""
    body = await request.json()
    well = body.get("well", "")
    analysis_type = body.get("analysis_type", "general")
    rating = int(body.get("rating", 3))
    comment = body.get("comment", "")
    expert_name = body.get("expert_name", "anonymous")

    entry = feedback_store.add_feedback(
        well, analysis_type, rating, comment, expert_name
    )
    return {"status": "ok", "entry": entry}


@app.post("/api/feedback/flag")
async def flag_fracture(request: Request):
    """Flag a fracture for expert review."""
    body = await request.json()
    well = body.get("well", "")
    fracture_idx = int(body.get("fracture_idx", 0))
    reason = body.get("reason", "")
    suggested_type = body.get("suggested_type", "")

    entry = feedback_store.flag_fracture(
        well, fracture_idx, reason, suggested_type
    )
    return {"status": "ok", "entry": entry}


@app.get("/api/feedback/summary")
async def get_feedback_summary():
    """Get summary of all collected feedback."""
    return _sanitize_for_json(feedback_store.get_summary())


@app.post("/api/feedback/correct-label")
async def correct_label(request: Request):
    """Record an expert correction of a fracture classification."""
    body = await request.json()
    well = body.get("well", "")
    fracture_idx = int(body.get("fracture_idx", 0))
    original_type = body.get("original_type", "")
    corrected_type = body.get("corrected_type", "")
    expert_name = body.get("expert_name", "anonymous")

    if not corrected_type:
        raise HTTPException(400, "corrected_type is required")

    entry = feedback_store.correct_label(
        well, fracture_idx, original_type, corrected_type, expert_name
    )
    return {
        "status": "ok",
        "entry": entry,
        "total_corrections": feedback_store.get_corrections_count(),
    }


@app.post("/api/feedback/retrain")
async def retrain_model(request: Request):
    """Retrain the model using expert-corrected labels.

    This closes the feedback loop: expert corrections -> better model.
    """
    body = await request.json()
    source = body.get("source", "demo")
    classifier = body.get("classifier", "xgboost")

    df = get_df(source)
    result = await asyncio.to_thread(
        retrain_with_corrections, df, classifier=classifier
    )
    return _sanitize_for_json(result)


# ── NEW: Enhanced Features Info ──────────────────────

@app.get("/api/analysis/features")
async def get_feature_info(source: str = "demo"):
    """Return the enhanced feature set computed from current data."""
    df = get_df(source)
    features = engineer_enhanced_features(df)

    # Summary stats for each feature
    stats = {}
    for col in features.columns:
        vals = features[col].dropna()
        stats[col] = {
            "mean": round(float(vals.mean()), 3),
            "std": round(float(vals.std()), 3),
            "min": round(float(vals.min()), 3),
            "max": round(float(vals.max()), 3),
        }

    return {
        "feature_count": len(features.columns),
        "feature_names": features.columns.tolist(),
        "stats": stats,
        "n_samples": len(features),
    }


# ── NEW: SHAP Explainability API ─────────────────────

_shap_cache = {}


@app.post("/api/analysis/shap")
async def shap_explanations(request: Request):
    """Compute SHAP explanations for stakeholder-friendly feature importance.

    Returns global importance, per-class drivers, and sample-level explanations.
    """
    body = await request.json()
    source = body.get("source", "demo")
    classifier = body.get("classifier", "gradient_boosting")

    df = get_df(source)

    cache_key = f"{source}_{len(df)}_{classifier}"
    if cache_key in _shap_cache:
        return _shap_cache[cache_key]

    result = await asyncio.to_thread(
        compute_shap_explanations, df, classifier=classifier
    )
    response = _sanitize_for_json(result)
    _shap_cache[cache_key] = response
    return response


# ── Sensitivity Analysis ─────────────────────────────

_sensitivity_cache = {}


@app.post("/api/analysis/sensitivity")
async def run_sensitivity(request: Request):
    """Run parameter sensitivity analysis on stress inversion results.

    Shows how results change when friction, pore pressure, and regime
    assumptions are varied. Returns tornado diagram data and risk implications.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", None)
    regime = body.get("regime", "strike_slip")
    depth_m = float(body.get("depth", 3000))
    pore_pressure = body.get("pore_pressure", None)

    df = get_df(source)
    if well:
        df = df[df[WELL_COL] == well]

    cache_key = f"sens_{source}_{well}_{regime}_{depth_m}"
    if cache_key in _sensitivity_cache:
        return _sensitivity_cache[cache_key]

    normals = fracture_plane_normal(
        df[AZIMUTH_COL].values, df[DIP_COL].values
    )
    pp = float(pore_pressure) if pore_pressure else None

    inv_result = await asyncio.to_thread(
        invert_stress, normals, regime=regime,
        depth_m=depth_m, pore_pressure=pp,
    )

    sens_result = await asyncio.to_thread(
        sensitivity_analysis, normals, inv_result, depth_m=depth_m,
    )

    response = _sanitize_for_json(sens_result)
    _sensitivity_cache[cache_key] = response
    return response


# ── Bayesian MCMC Inversion ──────────────────────────

@app.post("/api/analysis/bayesian")
async def run_bayesian(request: Request):
    """Run Bayesian MCMC inversion for proper uncertainty bounds.

    Produces posterior distributions and confidence intervals on all
    5 stress parameters. Gives stakeholders error bars, not just point
    estimates.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", None)
    regime = body.get("regime", "strike_slip")
    depth_m = float(body.get("depth", 3000))
    pore_pressure = body.get("pore_pressure", None)
    fast = body.get("fast", True)

    df = get_df(source)
    if well:
        df = df[df[WELL_COL] == well]

    normals = fracture_plane_normal(
        df[AZIMUTH_COL].values, df[DIP_COL].values
    )
    pp = float(pore_pressure) if pore_pressure else None

    # Run optimization first (gives initial point for MCMC)
    inv_result = await asyncio.to_thread(
        invert_stress, normals, regime=regime,
        depth_m=depth_m, pore_pressure=pp,
    )

    pp_val = inv_result.get("pore_pressure", 0.0)

    # Run MCMC
    bayes_result = await asyncio.to_thread(
        bayesian_inversion, normals, inv_result,
        regime=regime, pore_pressure=pp_val,
        depth_m=depth_m, fast=fast,
    )

    return _sanitize_for_json(bayes_result)


# ── Risk Assessment Matrix ───────────────────────────

@app.post("/api/analysis/risk-matrix")
async def run_risk_matrix(request: Request):
    """Compute comprehensive operational risk assessment.

    Combines all analysis results into a single go/no-go framework.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", None)
    regime = body.get("regime", "strike_slip")
    depth_m = float(body.get("depth", 3000))
    pore_pressure = body.get("pore_pressure", None)

    df = get_df(source)
    if well:
        df = df[df[WELL_COL] == well]

    normals = fracture_plane_normal(
        df[AZIMUTH_COL].values, df[DIP_COL].values
    )
    pp = float(pore_pressure) if pore_pressure else None

    # Run all prerequisite analyses
    inv_result = await asyncio.to_thread(
        invert_stress, normals, regime=regime,
        depth_m=depth_m, pore_pressure=pp,
    )

    pp_val = inv_result.get("pore_pressure", 0)
    cs_result = critically_stressed_enhanced(
        inv_result["sigma_n"], inv_result["tau"],
        mu=inv_result["mu"], pore_pressure=pp_val,
    )

    quality_result = validate_data_quality(df)

    # Try to get model comparison from cache
    cache_key_mc = f"{source}_{len(df)}_fast"
    model_comparison = _model_comparison_cache.get(cache_key_mc, None)

    # Sensitivity analysis
    sens_result = await asyncio.to_thread(
        sensitivity_analysis, normals, inv_result, depth_m=depth_m,
    )

    risk = compute_risk_matrix(
        inv_result, cs_result, quality_result,
        model_comparison=model_comparison,
        sensitivity_result=sens_result,
    )

    return _sanitize_for_json(risk)


# ── Well Report Generation ───────────────────────────

@app.post("/api/report/well")
async def generate_report(request: Request):
    """Generate a comprehensive stakeholder report for a well.

    Aggregates all analyses into a single printable report.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", None)
    regime = body.get("regime", "strike_slip")
    depth_m = float(body.get("depth", 3000))
    pore_pressure = body.get("pore_pressure", None)

    df = get_df(source)
    if well:
        well_name = well
        df_well = df[df[WELL_COL] == well]
    else:
        well_name = "All Wells"
        df_well = df

    normals = fracture_plane_normal(
        df_well[AZIMUTH_COL].values, df_well[DIP_COL].values
    )
    pp = float(pore_pressure) if pore_pressure else None

    inv_result = await asyncio.to_thread(
        invert_stress, normals, regime=regime,
        depth_m=depth_m, pore_pressure=pp,
    )

    pp_val = inv_result.get("pore_pressure", 0)
    cs_result = critically_stressed_enhanced(
        inv_result["sigma_n"], inv_result["tau"],
        mu=inv_result["mu"], pore_pressure=pp_val,
    )

    quality_result = validate_data_quality(df_well)

    # Try cached model comparison
    cache_key_mc = f"{source}_{len(df)}_fast"
    model_comparison = _model_comparison_cache.get(cache_key_mc, None)

    sens_result = await asyncio.to_thread(
        sensitivity_analysis, normals, inv_result, depth_m=depth_m,
    )

    risk = compute_risk_matrix(
        inv_result, cs_result, quality_result,
        model_comparison=model_comparison,
        sensitivity_result=sens_result,
    )

    report = generate_well_report(
        well_name, inv_result, cs_result, quality_result,
        model_comparison=model_comparison,
        sensitivity_result=sens_result,
        risk_matrix=risk,
    )

    return _sanitize_for_json(report)


# ── Multi-Well Comparison ────────────────────────────

@app.post("/api/analysis/compare-wells")
async def run_well_comparison(request: Request):
    """Compare analysis results across all wells.

    Checks stress field consistency, model transferability,
    and flags inter-well anomalies.
    """
    body = await request.json()
    source = body.get("source", "demo")
    depth_m = float(body.get("depth", 3000))

    df = get_df(source)
    result = await asyncio.to_thread(
        compare_wells, df, depth_m=depth_m,
    )
    return _sanitize_for_json(result)


# ── Auto-Analysis Overview ───────────────────────────

@app.post("/api/analysis/overview")
async def run_overview(request: Request):
    """Quick analysis overview for immediate insights on page load.

    Runs a fast stress inversion, data quality check, and risk estimate
    to give stakeholders the big picture without clicking through tabs.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", None)
    regime = body.get("regime", "strike_slip")
    depth_m = float(body.get("depth", 3000))

    df = get_df(source)
    if well:
        well_name = well
        df_well = df[df[WELL_COL] == well]
    else:
        well_name = "All Wells"
        df_well = df

    overview = {
        "well": well_name,
        "n_fractures": len(df_well),
        "n_wells": df[WELL_COL].nunique() if WELL_COL in df.columns else 0,
    }

    # Data quality (instant)
    quality = validate_data_quality(df_well)
    overview["data_quality"] = {
        "score": quality["score"],
        "grade": quality["grade"],
    }

    # Fast stress inversion
    try:
        normals = fracture_plane_normal(
            df_well[AZIMUTH_COL].values, df_well[DIP_COL].values
        )
        inv = await asyncio.to_thread(
            invert_stress, normals, regime=regime, depth_m=depth_m,
        )
        pp_val = inv.get("pore_pressure", 0.0)

        overview["stress"] = {
            "sigma1": round(inv["sigma1"], 1),
            "sigma3": round(inv["sigma3"], 1),
            "shmax": round(inv["shmax_azimuth_deg"], 0),
            "regime": regime,
            "mu": round(inv["mu"], 3),
        }

        # Quick critically stressed
        cs = critically_stressed_enhanced(
            inv["sigma_n"], inv["tau"],
            mu=inv["mu"], pore_pressure=pp_val,
        )
        overview["critically_stressed"] = {
            "pct": round(cs["pct_critical"], 1),
            "high_risk": cs["high_risk_count"],
            "total": cs["total"],
        }

        # Quick risk estimate (without model comparison or sensitivity)
        risk = compute_risk_matrix(inv, cs, quality)
        overview["risk"] = {
            "score": risk["overall_score"],
            "level": risk["overall_level"],
            "go_nogo": risk["go_nogo"],
        }

    except Exception as e:
        overview["stress"] = {"error": str(e)[:100]}
        overview["risk"] = {"level": "UNKNOWN", "go_nogo": "Cannot assess"}

    return _sanitize_for_json(overview)


# ── Uncertainty Budget ────────────────────────────────

@app.post("/api/analysis/uncertainty-budget")
async def run_uncertainty_budget(request: Request):
    """Compute an uncertainty budget ranking all analysis uncertainty sources.

    Aggregates uncertainties from the entire analysis pipeline — parameter
    sensitivity, Bayesian posteriors, data quality, ML confidence, cross-well
    consistency, and pore pressure estimation — into a single ranked view.

    Tells stakeholders where to invest next to reduce uncertainty.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", None)
    regime = body.get("regime", "strike_slip")
    depth_m = float(body.get("depth", 3000))
    pore_pressure = body.get("pore_pressure", None)
    run_bayesian_flag = body.get("include_bayesian", False)

    df = get_df(source)
    df_well = df
    if well:
        df_well = df[df[WELL_COL] == well]

    normals = fracture_plane_normal(
        df_well[AZIMUTH_COL].values, df_well[DIP_COL].values
    )
    pp = float(pore_pressure) if pore_pressure else None

    # 1. Stress inversion (always needed)
    inv_result = await asyncio.to_thread(
        invert_stress, normals, regime=regime,
        depth_m=depth_m, pore_pressure=pp,
    )

    # 2. Sensitivity analysis
    sens_result = await asyncio.to_thread(
        sensitivity_analysis, normals, inv_result, depth_m=depth_m,
    )

    # 3. Data quality
    quality_result = validate_data_quality(df_well)

    # 4. Bayesian (optional — expensive)
    bayes_result = None
    if run_bayesian_flag:
        pp_val = inv_result.get("pore_pressure", 0.0)
        bayes_result = await asyncio.to_thread(
            bayesian_inversion, normals, inv_result,
            regime=regime, pore_pressure=pp_val,
            depth_m=depth_m, fast=True,
        )

    # 5. Model comparison from cache (don't re-run)
    cache_key_mc = f"{source}_{len(df)}_fast"
    model_comparison = _model_comparison_cache.get(cache_key_mc, None)

    # 6. Well comparison (only if multiple wells)
    well_comparison = None
    if df[WELL_COL].nunique() >= 2:
        well_comparison = await asyncio.to_thread(
            compare_wells, df, depth_m=depth_m,
        )

    budget = compute_uncertainty_budget(
        inv_result,
        sensitivity_result=sens_result,
        bayesian_result=bayes_result,
        quality_result=quality_result,
        model_comparison=model_comparison,
        well_comparison=well_comparison,
    )

    return _sanitize_for_json(budget)
