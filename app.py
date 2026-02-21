"""GeoStress AI - FastAPI Web Application (v2.6 - Industrial Grade)."""

import os
import io
import time
import base64
import asyncio
import threading
import tempfile
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from functools import lru_cache
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from src.data_loader import (
    load_all_fractures, load_single_file, fracture_summary,
    fracture_plane_normal, AZIMUTH_COL, DIP_COL, DEPTH_COL,
    WELL_COL, FRACTURE_TYPE_COL,
)
from src.geostress import invert_stress, bayesian_inversion, auto_detect_regime
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
    compute_uncertainty_budget, active_learning_query,
    detect_ood, assess_calibration, data_collection_recommendations,
    compute_learning_curve, bootstrap_class_metrics, scenario_comparison,
    hierarchical_classify, decision_support_matrix,
    expert_weighted_ensemble, monte_carlo_uncertainty,
    cross_validate_wells, validate_domain_constraints,
    executive_summary, data_sufficiency_check,
    prediction_safety_check, field_consistency_check,
    physics_constraint_check, research_methods_summary,
    physics_constrained_predict, misclassification_analysis,
    evidence_chain_analysis, model_bias_detection,
    prediction_reliability_report,
)
from src.visualization import (
    plot_rose_diagram, _plot_stereonet_manual,
    plot_mohr_circle, plot_tendency, plot_depth_profile,
    plot_analysis_dashboard,
    plot_model_comparison, plot_learning_curve, plot_bootstrap_ci,
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
_auto_regime_cache = {}
_classify_cache = {}
_misclass_cache = {}
_physics_predict_cache = {}

# Audit trail for regulatory compliance (max 1000 entries in-memory)
_audit_log: deque = deque(maxlen=1000)
_audit_lock = threading.Lock()

# Progress tracking for SSE (Server-Sent Events)
_progress_queues: dict[str, asyncio.Queue] = {}
_progress_lock = threading.Lock()


def _emit_progress(task_id: str, step: str, pct: int = 0, detail: str = ""):
    """Publish a progress update for a running task.

    Called from within long-running operations (potentially in threads).
    """
    with _progress_lock:
        q = _progress_queues.get(task_id)
    if q:
        try:
            q.put_nowait({"step": step, "pct": pct, "detail": detail})
        except asyncio.QueueFull:
            pass  # Drop if client isn't reading fast enough


async def _cached_inversion(normals, well, regime, depth_m, pore_pressure, source):
    """Cache inversion results to avoid re-running the expensive optimization.

    Keyed by (source, well, regime, depth, pp_rounded). Returns dict.
    """
    pp_key = round(pore_pressure, 1) if pore_pressure else "auto"
    cache_key = f"inv_{source}_{well}_{regime}_{depth_m}_{pp_key}"
    if cache_key in _inversion_cache:
        return _inversion_cache[cache_key]

    result = await asyncio.to_thread(
        invert_stress, normals, regime=regime,
        depth_m=depth_m, pore_pressure=pore_pressure,
    )
    _inversion_cache[cache_key] = result
    return result


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
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _audit_record(action: str, params: dict, result_summary: dict,
                  source: str = "demo", well: str = None, elapsed_s: float = 0):
    """Record an analysis action in the audit trail for regulatory compliance."""
    # Create a hash of the result for integrity verification
    result_str = json.dumps(_sanitize_for_json(result_summary), sort_keys=True, default=str)
    result_hash = hashlib.sha256(result_str.encode()).hexdigest()[:16]

    record = {
        "id": len(_audit_log) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "source": source,
        "well": well,
        "parameters": _sanitize_for_json(params),
        "result_hash": result_hash,
        "result_summary": _sanitize_for_json(result_summary),
        "elapsed_s": round(elapsed_s, 2),
        "app_version": "2.7.0",
    }
    with _audit_lock:
        _audit_log.append(record)
    return record["id"]


# ── App lifecycle ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global demo_df
    demo_df = load_all_fractures(str(DATA_DIR))
    print(f"Loaded {len(demo_df)} demo fractures from {DATA_DIR}")
    yield


app = FastAPI(title="GeoStress AI", version="2.7.0", lifespan=lifespan)
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


# ── Response timing middleware ────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Response-Time header to all API responses for performance monitoring."""
    import time as _time
    start = _time.perf_counter()
    response = await call_next(request)
    elapsed = _time.perf_counter() - start
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    if request.url.path.startswith("/api/") and elapsed > 2.0:
        print(f"SLOW: {request.method} {request.url.path} took {elapsed:.1f}s")
    return response


# ── Cache status endpoint ─────────────────────────────

@app.get("/api/cache/status")
async def cache_status():
    """Return current cache sizes and hit information."""
    return {
        "inversion": len(_inversion_cache),
        "model_comparison": len(_model_comparison_cache),
        "auto_regime": len(_auto_regime_cache),
        "classify": len(_classify_cache),
        "misclass": len(_misclass_cache),
        "physics_predict": len(_physics_predict_cache),
        "shap": len(_shap_cache),
        "sensitivity": len(_sensitivity_cache),
    }


# ── SSE Progress Streaming ────────────────────────────

@app.get("/api/progress/{task_id}")
async def progress_stream(task_id: str):
    """Server-Sent Events endpoint for long-running task progress.

    Frontend subscribes to this during long operations. Events contain
    step name, percentage, and detail text.
    """
    q = asyncio.Queue(maxsize=50)
    with _progress_lock:
        _progress_queues[task_id] = q

    async def event_generator():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30.0)
                    data = json.dumps(msg)
                    yield f"data: {data}\n\n"
                    if msg.get("pct", 0) >= 100:
                        break
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f": keepalive\n\n"
        finally:
            with _progress_lock:
                _progress_queues.pop(task_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
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
        # Clear all caches when new data is uploaded
        _model_comparison_cache.clear()
        _inversion_cache.clear()
        _auto_regime_cache.clear()
        _sensitivity_cache.clear()
        _shap_cache.clear()
        _classify_cache.clear()
        _misclass_cache.clear()
        _physics_predict_cache.clear()

        result = {
            "filename": file.filename,
            "rows": len(new_df),
            "wells": new_df[WELL_COL].unique().tolist(),
            "fracture_types": new_df[FRACTURE_TYPE_COL].unique().tolist(),
            "source": "uploaded",
        }

        # Auto OOD check against demo data
        if demo_df is not None:
            try:
                ood_result = detect_ood(demo_df, new_df)
                result["ood_check"] = {
                    "severity": ood_result["severity"],
                    "message": ood_result["message"],
                    "ood_detected": ood_result["ood_detected"],
                }
            except Exception:
                result["ood_check"] = None

        return _sanitize_for_json(result)
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
    source = body.get("source", "demo")

    # Input validation
    try:
        depth_m = float(body.get("depth_m", 3300.0))
    except (ValueError, TypeError):
        raise HTTPException(400, "depth_m must be a number")
    if depth_m <= 0 or depth_m > 15000:
        raise HTTPException(400, f"depth_m={depth_m} is out of valid range (0-15000m)")

    try:
        cohesion = float(body.get("cohesion", 0.0))
    except (ValueError, TypeError):
        raise HTTPException(400, "cohesion must be a number")
    if cohesion < 0 or cohesion > 100:
        raise HTTPException(400, f"cohesion={cohesion} is out of valid range (0-100 MPa)")

    pore_pressure = body.get("pore_pressure", None)
    if pore_pressure is not None:
        try:
            pore_pressure = float(pore_pressure)
        except (ValueError, TypeError):
            raise HTTPException(400, "pore_pressure must be a number or null")
        if pore_pressure < 0 or pore_pressure > 500:
            raise HTTPException(400, f"pore_pressure={pore_pressure} is out of valid range (0-500 MPa)")

    valid_regimes = {"normal", "strike_slip", "thrust", "auto"}
    if regime not in valid_regimes:
        raise HTTPException(400, f"regime must be one of {valid_regimes}")

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

    # Auto-detect regime or use specified one
    auto_detection = None
    if regime == "auto":
        auto_detection = await asyncio.to_thread(
            auto_detect_regime, normals, avg_depth, cohesion, pore_pressure,
        )
        result = auto_detection["best_result"]
        regime = auto_detection["best_regime"]
        # Cache the best result for downstream use
        pp_key = round(result["pore_pressure"], 1) if result.get("pore_pressure") else "auto"
        cache_key = f"inv_{source}_{well}_{regime}_{avg_depth}_{pp_key}"
        _inversion_cache[cache_key] = result
    else:
        result = await _cached_inversion(
            normals, well, regime, avg_depth, pore_pressure, source
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

    response = {
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
    }

    # Include auto-detection results if applicable
    if auto_detection:
        response["auto_regime"] = {
            "best_regime": auto_detection["best_regime"],
            "confidence": auto_detection["confidence"],
            "misfit_ratio": auto_detection["misfit_ratio"],
            "comparison": auto_detection["comparison"],
            "stakeholder_summary": auto_detection["stakeholder_summary"],
        }

    # Audit trail
    _audit_record("stress_inversion",
                  {"regime": regime, "depth_m": depth_m, "cohesion": cohesion,
                   "pore_pressure": pore_pressure},
                  {"sigma1": response["sigma1"], "sigma3": response["sigma3"],
                   "shmax": response["shmax_azimuth_deg"], "mu": response["mu"],
                   "critically_stressed_pct": response["critically_stressed_pct"]},
                  source=source, well=well)

    return _sanitize_for_json(response)


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

    # Generate comparison chart
    try:
        ranking = result.get("ranking", [])
        if isinstance(ranking, list) and len(ranking) > 0:
            # ranking is a list of dicts with 'model', 'accuracy', 'balanced_accuracy'
            chart_data = []
            for item in ranking:
                if isinstance(item, dict):
                    chart_data.append({
                        "model": item.get("model", "?"),
                        "cv_accuracy_mean": item.get("accuracy", 0),
                        "balanced_accuracy": item.get("balanced_accuracy", 0),
                    })
            if chart_data:
                chart_img = await asyncio.to_thread(
                    render_plot, plot_model_comparison, chart_data, "Model Comparison"
                )
                result["comparison_chart_img"] = chart_img
            else:
                result["comparison_chart_img"] = None
        else:
            result["comparison_chart_img"] = None
    except Exception:
        result["comparison_chart_img"] = None

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


@app.post("/api/feedback/trust-score")
async def compute_trust_score(request: Request):
    """Compute a comprehensive trust score for model predictions.

    Combines expert feedback ratings, calibration ECE, bootstrap CI width,
    OOD detection, and data quality into a single trust score.
    This is the RLHF-style component: expert feedback directly adjusts
    the model's reported confidence.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well")

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    if well and WELL_COL in df.columns:
        df_well = df[df[WELL_COL] == well]
    else:
        df_well = df

    # Gather all trust signals
    signals = {}

    # 1. Expert feedback signal
    summary = feedback_store.get_summary()
    expert_rating = summary.get("avg_rating")
    if expert_rating is not None:
        # Normalize 1-5 to 0-100
        signals["expert_feedback"] = {
            "score": round((expert_rating - 1) / 4 * 100, 0),
            "weight": 0.30,
            "detail": f"Average expert rating: {expert_rating:.1f}/5 from {summary['total_feedback']} reviews",
        }
    else:
        signals["expert_feedback"] = {
            "score": 50,  # Neutral when no feedback
            "weight": 0.10,  # Low weight without data
            "detail": "No expert feedback yet. Submit reviews to improve trust assessment.",
        }

    # 2. Data quality signal
    quality = validate_data_quality(df_well)
    signals["data_quality"] = {
        "score": quality["score"],
        "weight": 0.25,
        "detail": f"Grade {quality['grade']}: {quality['score']}/100",
    }

    # 3. Label corrections signal
    corrections = feedback_store.get_corrections_count()
    if corrections > 0:
        signals["corrections_applied"] = {
            "score": min(100, corrections * 10),  # More corrections = more refined
            "weight": 0.15,
            "detail": f"{corrections} expert corrections integrated",
        }
    else:
        signals["corrections_applied"] = {
            "score": 30,
            "weight": 0.10,
            "detail": "No corrections yet. Correct misclassified fractures to improve accuracy.",
        }

    # 4. Sample size signal
    n = len(df_well)
    size_score = min(100, n / 2)  # 200 samples = 100%
    signals["sample_size"] = {
        "score": round(size_score, 0),
        "weight": 0.15,
        "detail": f"{n} fractures. Target: >= 200 for reliable models.",
    }

    # 5. Calibration signal (fast)
    try:
        cal = await asyncio.to_thread(assess_calibration, df_well, 10, True)
        ece = cal.get("ece", 50)
        cal_score = max(0, 100 - ece * 1000)  # ECE of 0.027 -> 73
        signals["calibration"] = {
            "score": round(cal_score, 0),
            "weight": 0.15,
            "detail": f"ECE={ece:.3f}. {cal.get('reliability', 'Unknown')} calibration.",
        }
    except Exception:
        signals["calibration"] = {
            "score": 50,
            "weight": 0.10,
            "detail": "Could not assess calibration",
        }

    # Compute weighted trust score
    total_weight = sum(s["weight"] for s in signals.values())
    trust_score = sum(s["score"] * s["weight"] for s in signals.values()) / total_weight

    # Trust level
    if trust_score >= 80:
        trust_level = "HIGH"
        trust_msg = "Model predictions can be used for operational decisions with standard monitoring."
    elif trust_score >= 60:
        trust_level = "MODERATE"
        trust_msg = "Model predictions should be verified by domain experts before use in critical decisions."
    elif trust_score >= 40:
        trust_level = "LOW"
        trust_msg = "Model predictions have significant uncertainty. Expert review required for all decisions."
    else:
        trust_level = "VERY LOW"
        trust_msg = "Model predictions are unreliable. Do NOT use for operational decisions without comprehensive expert review."

    # How to improve
    improvements = []
    sorted_signals = sorted(signals.items(), key=lambda x: x[1]["score"])
    for name, sig in sorted_signals[:3]:
        if sig["score"] < 70:
            improvements.append({
                "factor": name.replace("_", " ").title(),
                "current_score": sig["score"],
                "action": sig["detail"],
            })

    return _sanitize_for_json({
        "trust_score": round(trust_score, 1),
        "trust_level": trust_level,
        "trust_message": trust_msg,
        "signals": signals,
        "improvements": improvements,
        "feedback_loop_active": expert_rating is not None,
        "corrections_count": corrections,
    })


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

    inv_result = await _cached_inversion(
        normals, well, regime, depth_m, pp, source
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

    # Run optimization first (gives initial point for MCMC) — cached
    inv_result = await _cached_inversion(
        normals, well, regime, depth_m, pp, source
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

    # Run all prerequisite analyses (cached inversion)
    inv_result = await _cached_inversion(
        normals, well, regime, depth_m, pp, source
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

    # Auto-regime detection for report
    auto_regime = None
    if regime == "auto":
        auto_cache_key = f"auto_{source}_{well_name}_{depth_m}"
        if auto_cache_key in _auto_regime_cache:
            auto_regime = _auto_regime_cache[auto_cache_key]
        else:
            auto_regime = await asyncio.to_thread(
                auto_detect_regime, normals, depth_m, 0.0, pp,
            )
            _auto_regime_cache[auto_cache_key] = auto_regime
        inv_result = auto_regime["best_result"]
        regime = auto_regime["best_regime"]
    else:
        inv_result = await _cached_inversion(
            normals, well, regime, depth_m, pp, source
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

    # Calibration and data recommendations for report
    try:
        cal_result = await asyncio.to_thread(assess_calibration, df_well, 10, True)
    except Exception:
        cal_result = None

    try:
        data_recs = data_collection_recommendations(df_well)
    except Exception:
        data_recs = None

    report = generate_well_report(
        well_name, inv_result, cs_result, quality_result,
        model_comparison=model_comparison,
        sensitivity_result=sens_result,
        risk_matrix=risk,
        auto_regime_result=auto_regime,
        calibration_result=cal_result,
        data_recommendations=data_recs,
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

    Runs stress inversion, calibration, and data recommendations IN PARALLEL
    using asyncio.gather to minimize wall-clock time. Returns timing breakdown
    so users can see performance.
    """
    t_start = time.monotonic()
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

    # Data quality (instant, < 10ms)
    quality = validate_data_quality(df_well)
    overview["data_quality"] = {
        "score": quality["score"],
        "grade": quality["grade"],
    }

    # ── Run 3 independent tasks in parallel ──────────
    normals = fracture_plane_normal(
        df_well[AZIMUTH_COL].values, df_well[DIP_COL].values
    )

    async def _stress_chain():
        """Stress inversion → critically stressed → risk (sequential chain)."""
        t0 = time.monotonic()
        result = {}
        try:
            nonlocal regime
            if regime == "auto" or regime == "strike_slip":
                auto_cache_key = f"auto_{source}_{well_name}_{depth_m}"
                if auto_cache_key in _auto_regime_cache:
                    auto_res = _auto_regime_cache[auto_cache_key]
                else:
                    auto_res = await asyncio.to_thread(
                        auto_detect_regime, normals, depth_m, 0.0, None,
                    )
                    _auto_regime_cache[auto_cache_key] = auto_res
                inv = auto_res["best_result"]
                regime = auto_res["best_regime"]
                result["regime_detection"] = {
                    "best_regime": auto_res["best_regime"],
                    "confidence": auto_res["confidence"],
                    "misfit_ratio": auto_res["misfit_ratio"],
                }
            else:
                inv = await _cached_inversion(
                    normals, well, regime, depth_m, None, source
                )

            pp_val = inv.get("pore_pressure", 0.0)
            result["stress"] = {
                "sigma1": round(inv["sigma1"], 1),
                "sigma3": round(inv["sigma3"], 1),
                "shmax": round(inv["shmax_azimuth_deg"], 0),
                "regime": regime,
                "mu": round(inv["mu"], 3),
            }

            cs = critically_stressed_enhanced(
                inv["sigma_n"], inv["tau"],
                mu=inv["mu"], pore_pressure=pp_val,
            )
            result["critically_stressed"] = {
                "pct": round(cs["pct_critical"], 1),
                "high_risk": cs["high_risk_count"],
                "total": cs["total"],
            }

            risk = compute_risk_matrix(inv, cs, quality)
            result["risk"] = {
                "score": risk["overall_score"],
                "level": risk["overall_level"],
                "go_nogo": risk["go_nogo"],
            }
        except Exception as e:
            result["stress"] = {"error": str(e)[:100]}
            result["risk"] = {"level": "UNKNOWN", "go_nogo": "Cannot assess"}
        result["_elapsed"] = round(time.monotonic() - t0, 2)
        return result

    async def _calibration():
        """Quick calibration check (fast mode)."""
        t0 = time.monotonic()
        try:
            cal = await asyncio.to_thread(assess_calibration, df_well, 10, True)
            return {
                "calibration": {"reliability": cal["reliability"], "ece": cal["ece"]},
                "_elapsed": round(time.monotonic() - t0, 2),
            }
        except Exception:
            return {"calibration": None, "_elapsed": round(time.monotonic() - t0, 2)}

    async def _data_recs():
        """Data collection recommendations."""
        t0 = time.monotonic()
        try:
            recs = data_collection_recommendations(df_well)
            return {
                "data_recommendations": {
                    "n_priority": len(recs["priority_actions"]),
                    "n_recommendations": len(recs["recommendations"]),
                    "completeness_pct": recs["data_completeness_pct"],
                },
                "_elapsed": round(time.monotonic() - t0, 2),
            }
        except Exception:
            return {"data_recommendations": None, "_elapsed": round(time.monotonic() - t0, 2)}

    # Fire all 3 concurrently
    stress_res, cal_res, recs_res = await asyncio.gather(
        _stress_chain(), _calibration(), _data_recs()
    )

    # Merge results
    for key in ("stress", "regime_detection", "critically_stressed", "risk"):
        if key in stress_res:
            overview[key] = stress_res[key]
    overview["calibration"] = cal_res.get("calibration")
    overview["data_recommendations"] = recs_res.get("data_recommendations")

    # Timing breakdown for transparency
    total_elapsed = round(time.monotonic() - t_start, 2)
    overview["_timing"] = {
        "total_s": total_elapsed,
        "stress_s": stress_res.get("_elapsed", 0),
        "calibration_s": cal_res.get("_elapsed", 0),
        "recommendations_s": recs_res.get("_elapsed", 0),
        "parallel": True,
    }

    # Audit trail
    _audit_record("overview", {"well": well_name, "regime": regime, "depth_m": depth_m},
                  {"risk": overview.get("risk", {}), "shmax": overview.get("stress", {}).get("shmax")},
                  source=source, well=well_name, elapsed_s=total_elapsed)

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

    # 1. Stress inversion (always needed — cached)
    inv_result = await _cached_inversion(
        normals, well, regime, depth_m, pp, source
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


# ── Active Learning ───────────────────────────────────

@app.post("/api/analysis/active-learning")
async def run_active_learning(request: Request):
    """Identify fractures the model is most uncertain about.

    Suggests the highest-value samples for expert review using entropy
    and margin sampling. This is the practical human-in-the-loop
    equivalent of RLHF: experts label the most uncertain cases,
    model retrains on the corrections.
    """
    body = await request.json()
    source = body.get("source", "demo")
    n_suggest = int(body.get("n_suggest", 20))
    classifier = body.get("classifier", "xgboost")

    df = get_df(source)

    result = await asyncio.to_thread(
        active_learning_query, df, n_suggest=n_suggest, classifier=classifier,
    )
    return _sanitize_for_json(result)


# ── Data & Results Export ────────────────────────────

@app.post("/api/export/data")
async def export_data(request: Request):
    """Export fracture data as CSV string for download."""
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", None)

    df = get_df(source)
    if well:
        df = df[df[WELL_COL] == well]
    if len(df) == 0:
        raise HTTPException(404, "No data to export")

    csv_str = df.to_csv(index=False)
    return {"csv": csv_str, "rows": len(df), "filename": f"fractures_{well or 'all'}.csv"}


@app.post("/api/export/inversion")
async def export_inversion(request: Request):
    """Export inversion results as structured JSON + CSV table.

    Runs inversion (or uses cache) and returns exportable formats
    with all key parameters, tendencies, and stakeholder interpretation.
    """
    body = await request.json()
    well = body.get("well", "3P")
    regime = body.get("regime", "strike_slip")
    depth_m = float(body.get("depth_m", 3300.0))
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

    # Handle auto regime
    if regime == "auto":
        auto_result = await asyncio.to_thread(
            auto_detect_regime, normals, avg_depth, 0.0, pore_pressure,
        )
        result = auto_result["best_result"]
        regime = auto_result["best_regime"]
    else:
        result = await _cached_inversion(
            normals, well, regime, avg_depth, pore_pressure, source
        )

    pp = result.get("pore_pressure", 0.0)
    cs_result = critically_stressed_enhanced(
        result["sigma_n"], result["tau"], result["mu"], 0.0, pp
    )

    # Build per-fracture CSV with tendencies
    export_df = df_well[[DEPTH_COL, AZIMUTH_COL, DIP_COL]].copy()
    if WELL_COL in df_well.columns:
        export_df["Well"] = df_well[WELL_COL]
    if FRACTURE_TYPE_COL in df_well.columns:
        export_df["Fracture_Type"] = df_well[FRACTURE_TYPE_COL]
    export_df["Slip_Tendency"] = result["slip_tend"]
    export_df["Dilation_Tendency"] = result["dilation_tend"]
    export_df["Normal_Stress_MPa"] = result["sigma_n"]
    export_df["Shear_Stress_MPa"] = result["tau"]
    export_df["Effective_Normal_Stress_MPa"] = result["effective_sigma_n"]
    export_df["Critically_Stressed"] = (
        result["tau"] > result["mu"] * (result["sigma_n"] - pp)
    )

    csv_str = export_df.to_csv(index=False)

    # Summary JSON
    summary = {
        "well": well,
        "regime": regime,
        "sigma1_MPa": round(float(result["sigma1"]), 2),
        "sigma2_MPa": round(float(result["sigma2"]), 2),
        "sigma3_MPa": round(float(result["sigma3"]), 2),
        "R_ratio": round(float(result["R"]), 4),
        "SHmax_azimuth_deg": round(float(result["shmax_azimuth_deg"]), 1),
        "friction_coefficient": round(float(result["mu"]), 4),
        "pore_pressure_MPa": round(pp, 2),
        "fracture_count": len(df_well),
        "critically_stressed_count": cs_result["count_critical"],
        "critically_stressed_pct": cs_result["pct_critical"],
    }

    return _sanitize_for_json({
        "csv": csv_str,
        "summary": summary,
        "rows": len(export_df),
        "filename": f"inversion_{well}_{regime}.csv",
    })


# ── OOD Detection ─────────────────────────────────
@app.post("/api/analysis/ood-check")
async def ood_check(request: Request):
    """Check if uploaded data is out-of-distribution vs demo data."""
    body = await request.json()
    source = body.get("source", "demo")

    if source == "uploaded" and uploaded_df is not None and demo_df is not None:
        result = await asyncio.to_thread(detect_ood, demo_df, uploaded_df)
        return _sanitize_for_json(result)
    elif source == "demo" and demo_df is not None:
        # Compare wells against each other
        wells = demo_df[WELL_COL].unique()
        if len(wells) >= 2:
            df_a = demo_df[demo_df[WELL_COL] == wells[0]]
            df_b = demo_df[demo_df[WELL_COL] == wells[1]]
            result = await asyncio.to_thread(detect_ood, df_a, df_b)
            result["note"] = f"Cross-well OOD: {wells[0]} vs {wells[1]}"
            return _sanitize_for_json(result)

    return {"ood_detected": False, "severity": "N/A",
            "message": "OOD check requires both reference and new data. Upload data to compare."}


# ── Calibration Assessment ────────────────────────
@app.post("/api/analysis/calibration")
async def calibration_assessment(request: Request):
    """Assess model probability calibration (are confidence values reliable?)."""
    body = await request.json()
    source = body.get("source", "demo")
    fast = body.get("fast", True)

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")

    result = await asyncio.to_thread(assess_calibration, df, 10, fast)
    return _sanitize_for_json(result)


# ── Data Collection Recommendations ───────────────
@app.post("/api/data/recommendations")
async def data_recommendations(request: Request):
    """Get actionable recommendations for what data to collect next."""
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well")

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")

    if well and WELL_COL in df.columns:
        df = df[df[WELL_COL] == well]

    result = await asyncio.to_thread(data_collection_recommendations, df)
    return _sanitize_for_json(result)


# ── Learning Curve ───────────────────────────────────

@app.post("/api/analysis/learning-curve")
async def run_learning_curve(request: Request):
    """Compute learning curve: how accuracy improves with more data.

    Shows stakeholders whether collecting more data would help, and
    projects how many samples are needed for target accuracy levels.
    """
    t0 = time.monotonic()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well")

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    if well and WELL_COL in df.columns:
        df = df[df[WELL_COL] == well]

    result = await asyncio.to_thread(compute_learning_curve, df, 8, True)

    # Generate learning curve chart
    if "error" not in result:
        try:
            chart_img = await asyncio.to_thread(
                render_plot, plot_learning_curve,
                result["train_sizes"], result["train_scores"],
                result["val_scores"], result.get("balanced_scores"),
                f"Learning Curve — {well or 'All Wells'}",
            )
            result["chart_img"] = chart_img
        except Exception:
            result["chart_img"] = None

    elapsed = round(time.monotonic() - t0, 2)

    _audit_record("learning_curve", {"well": well, "n_points": 8},
                  {"convergence": result.get("convergence")},
                  source=source, well=well, elapsed_s=elapsed)

    result["elapsed_s"] = elapsed
    return _sanitize_for_json(result)


# ── Bootstrap Confidence Intervals ────────────────────

@app.post("/api/analysis/bootstrap-ci")
async def run_bootstrap_ci(request: Request):
    """Compute bootstrap 95% CIs for per-class metrics.

    Industrial-grade: returns confidence ranges, not just point estimates.
    """
    t0 = time.monotonic()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well")
    n_bootstrap = int(body.get("n_bootstrap", 200))

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    if well and WELL_COL in df.columns:
        df = df[df[WELL_COL] == well]

    result = await asyncio.to_thread(
        bootstrap_class_metrics, df, n_bootstrap, 0.95, True
    )

    # Generate CI chart
    if "error" not in result and result.get("per_class"):
        try:
            chart_img = await asyncio.to_thread(
                render_plot, plot_bootstrap_ci,
                result["class_names"], result["per_class"],
                f"Per-Class F1 with 95% CI — {well or 'All Wells'}",
            )
            result["chart_img"] = chart_img
        except Exception:
            result["chart_img"] = None

    elapsed = round(time.monotonic() - t0, 2)

    _audit_record("bootstrap_ci", {"well": well, "n_bootstrap": n_bootstrap},
                  {"reliability": result.get("reliability"),
                   "accuracy": result.get("accuracy", {}).get("mean")},
                  source=source, well=well, elapsed_s=elapsed)

    result["elapsed_s"] = elapsed
    return _sanitize_for_json(result)


# ── Scenario Comparison ───────────────────────────────

@app.post("/api/analysis/scenarios")
async def run_scenarios(request: Request):
    """Compare multiple what-if stress inversion scenarios.

    Lets engineers compare different assumptions (regime, pore pressure)
    side-by-side to make informed drilling decisions.
    """
    t0 = time.monotonic()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    scenarios = body.get("scenarios", [])
    depth_m = float(body.get("depth", 3000))

    if not scenarios or len(scenarios) < 2:
        raise HTTPException(400, "Provide at least 2 scenarios to compare")
    if len(scenarios) > 6:
        raise HTTPException(400, "Maximum 6 scenarios per comparison")

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    df_well = df[df[WELL_COL] == well].reset_index(drop=True)
    if len(df_well) == 0:
        raise HTTPException(404, f"No data for well {well}")

    result = await asyncio.to_thread(
        scenario_comparison, df_well, scenarios, depth_m
    )
    elapsed = round(time.monotonic() - t0, 2)

    _audit_record("scenario_comparison",
                  {"well": well, "n_scenarios": len(scenarios), "depth_m": depth_m},
                  {"recommendation": result.get("recommendation", "")[:100]},
                  source=source, well=well, elapsed_s=elapsed)

    result["elapsed_s"] = elapsed
    return _sanitize_for_json(result)


# ── Decision Support Matrix ────────────────────────────

@app.post("/api/analysis/decision-matrix")
async def run_decision_matrix(request: Request):
    """Generate decision support matrix comparing all regime options.

    Gives stakeholders OPTIONS with trade-offs rather than a single answer.
    Includes go/no-go for each regime, risk comparison, and recommended action.
    """
    t0 = time.monotonic()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth_m = float(body.get("depth", 3000))

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    df_well = df[df[WELL_COL] == well].reset_index(drop=True) if well else df
    if len(df_well) == 0:
        raise HTTPException(404, f"No data for well {well}")

    result = await asyncio.to_thread(
        decision_support_matrix, df_well, well or "All", depth_m
    )
    elapsed = round(time.monotonic() - t0, 2)

    _audit_record("decision_matrix", {"well": well, "depth_m": depth_m},
                  {"recommended_action": result.get("recommended_action", "")[:80],
                   "confidence": result.get("confidence", {}).get("overall")},
                  source=source, well=well, elapsed_s=elapsed)

    result["elapsed_s"] = elapsed
    return _sanitize_for_json(result)


# ── Hierarchical Classification ────────────────────────

@app.post("/api/analysis/hierarchical")
async def run_hierarchical(request: Request):
    """Two-level hierarchical classification for rare fracture types.

    Splits classes into common vs rare, then classifies within each group.
    Compares hierarchical vs flat approach and recommends best strategy.
    """
    t0 = time.monotonic()
    body = await request.json()
    source = body.get("source", "demo")

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")

    result = await asyncio.to_thread(hierarchical_classify, df, 20, True)
    elapsed = round(time.monotonic() - t0, 2)

    _audit_record("hierarchical_classify", {},
                  {"applicable": result.get("applicable"),
                   "recommendation": result.get("recommendation", {}).get("approach")},
                  source=source, elapsed_s=elapsed)

    result["elapsed_s"] = elapsed
    return _sanitize_for_json(result)


# ── Expert-Weighted Ensemble ────────────────────────

@app.post("/api/analysis/expert-ensemble")
async def run_expert_ensemble(request: Request):
    """RLHF-style ensemble: model weights adjusted by expert feedback.

    Without expert feedback, uses accuracy-proportional weights.
    With feedback, models that align with expert corrections get higher weight.
    """
    t0 = time.monotonic()
    body = await request.json()
    source = body.get("source", "demo")

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")

    # Build expert weights from feedback patterns
    expert_weights = None
    summary = feedback_store.get_summary()
    if summary.get("avg_rating") is not None and summary["total_feedback"] >= 3:
        # Use feedback to boost/penalize models
        # Higher rating = models are more trustworthy → boost ensemble baseline
        rating_factor = summary["avg_rating"] / 3.0  # 3/5 = neutral
        expert_weights = {
            "random_forest": rating_factor * 1.1,  # RF typically best calibrated
            "xgboost": rating_factor * 1.05,
            "lightgbm": rating_factor * 1.05,
            "gradient_boosting": rating_factor * 0.95,
            "svm": rating_factor * 0.9,
            "mlp": rating_factor * 0.9,
        }

    result = await asyncio.to_thread(expert_weighted_ensemble, df, expert_weights, True)
    elapsed = round(time.monotonic() - t0, 2)

    _audit_record("expert_ensemble", {"expert_weights_applied": expert_weights is not None},
                  {"accuracy": result.get("expert_weight_accuracy"),
                   "improvement": result.get("improvement"),
                   "disagreement": result.get("disagreement_rate")},
                  source=source, elapsed_s=elapsed)

    result["elapsed_s"] = elapsed
    return _sanitize_for_json(result)


# ── Monte Carlo Uncertainty ────────────────────────

@app.post("/api/analysis/monte-carlo")
async def run_monte_carlo(request: Request):
    """Monte Carlo uncertainty propagation through the analysis chain.

    Perturbs measurement inputs and re-runs inversion N times to quantify
    how measurement errors affect final results.
    """
    t0 = time.monotonic()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well")
    regime = body.get("regime", "normal")
    depth_m = float(body.get("depth", 3000))
    n_sims = int(body.get("n_simulations", 200))
    az_std = float(body.get("azimuth_std", 5.0))
    dip_std = float(body.get("dip_std", 3.0))
    dep_std = float(body.get("depth_std", 2.0))

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")

    result = await asyncio.to_thread(
        monte_carlo_uncertainty, df, well, n_sims,
        az_std, dip_std, dep_std, regime, depth_m, True
    )
    elapsed = round(time.monotonic() - t0, 2)

    if "error" not in result:
        _audit_record("monte_carlo", {
            "well": well, "regime": regime, "n_sims": n_sims,
            "az_std": az_std, "dip_std": dip_std, "dep_std": dep_std,
        }, {
            "reliability": result.get("reliability"),
            "shmax_ci": result.get("statistics", {}).get("shmax", {}).get("ci_width"),
        }, source=source, well=well, elapsed_s=elapsed)

    result["elapsed_s"] = elapsed
    return _sanitize_for_json(result)


# ── Cross-Well Validation ──────────────────────────

@app.post("/api/analysis/cross-well-cv")
async def run_cross_well_cv(request: Request):
    """Leave-one-well-out cross-validation to test model transferability.

    Gold standard for testing whether predictions on NEW wells are reliable.
    """
    t0 = time.monotonic()
    body = await request.json()
    source = body.get("source", "demo")
    classifier = body.get("classifier", "random_forest")

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")

    result = await asyncio.to_thread(cross_validate_wells, df, classifier, True)
    elapsed = round(time.monotonic() - t0, 2)

    if "error" not in result:
        _audit_record("cross_well_cv", {"classifier": classifier},
                      {"transferability": result.get("transferability")},
                      source=source, elapsed_s=elapsed)

    result["elapsed_s"] = elapsed
    return _sanitize_for_json(result)


# ── Domain Constraint Validation ────────────────────

@app.post("/api/data/validate-constraints")
async def run_domain_validation(request: Request):
    """Validate data against physical and geological constraints.

    Catches impossible values, unusual combinations, and data anomalies
    before they corrupt analysis results.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well")

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    if well and WELL_COL in df.columns:
        df = df[df[WELL_COL] == well]

    result = await asyncio.to_thread(validate_domain_constraints, df)
    return _sanitize_for_json(result)


# ── Executive Summary ────────────────────────────────

@app.post("/api/analysis/executive-summary")
async def run_executive_summary(request: Request):
    """Generate a plain-language executive summary for non-technical stakeholders.

    Synthesizes stress, classification, and trust data into traffic-light
    risk communication with actionable recommendations.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth_m = float(body.get("depth", 3000))

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    df_well = df[df[WELL_COL] == well].reset_index(drop=True) if well else df

    # Gather inputs in parallel
    async def _inversion():
        from src.data_loader import fracture_plane_normal
        normals = fracture_plane_normal(
            df_well[AZIMUTH_COL].values, df_well[DIP_COL].values
        )
        # Use auto_detect_regime to find best regime, then invert
        auto = await asyncio.to_thread(
            auto_detect_regime, normals, depth_m
        )
        regime = auto["best_regime"]
        result = await asyncio.to_thread(
            invert_stress, normals, regime=regime, depth_m=depth_m
        )
        result["regime"] = regime
        result["confidence"] = auto.get("confidence", "UNKNOWN")
        return result

    async def _trust():
        try:
            cal = await asyncio.to_thread(assess_calibration, df_well, 10, True)
            quality = validate_data_quality(df_well)
            summary = feedback_store.get_summary()
            expert_rating = summary.get("avg_rating")
            # Simplified trust calc
            signals = [
                quality["score"] * 0.25,
                (min(100, len(df_well) / 2)) * 0.15,
                max(0, 100 - cal.get("ece", 0.5) * 1000) * 0.15,
            ]
            if expert_rating:
                signals.append(((expert_rating - 1) / 4 * 100) * 0.30)
                signals.append((min(100, feedback_store.get_corrections_count() * 10)) * 0.15)
                total_w = 1.0
            else:
                signals.append(50 * 0.10)  # neutral expert
                signals.append(30 * 0.10)  # no corrections
                total_w = 0.75
            score = sum(signals) / total_w
            level = "HIGH" if score >= 80 else "MODERATE" if score >= 60 else "LOW" if score >= 40 else "VERY LOW"
            return score, level
        except Exception:
            return 50.0, "UNKNOWN"

    inv_result, (trust_val, trust_lvl) = await asyncio.gather(
        _inversion(), _trust()
    )

    result = executive_summary(
        df_well, well_name=well,
        inversion_result=inv_result,
        trust_score=trust_val,
        trust_level=trust_lvl,
    )
    return _sanitize_for_json(result)


# ── Data Sufficiency ─────────────────────────────────

@app.post("/api/data/sufficiency")
async def run_sufficiency_check(request: Request):
    """Assess whether current data is sufficient for each analysis type."""
    body = await request.json()
    source = body.get("source", "demo")

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")

    result = data_sufficiency_check(df)
    return _sanitize_for_json(result)


# ── Prediction Safety ────────────────────────────────

@app.post("/api/analysis/safety-check")
async def run_safety_check(request: Request):
    """Run comprehensive safety check before operational use of predictions.

    Detects failure modes, anomalies, and conditions that would make
    predictions unreliable. Returns go/no-go recommendation.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well")

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    if well and WELL_COL in df.columns:
        df = df[df[WELL_COL] == well]

    # Get inversion result if available
    inv_result = None
    try:
        normals = fracture_plane_normal(
            df[AZIMUTH_COL].values, df[DIP_COL].values
        )
        inv_result = await asyncio.to_thread(
            invert_stress, normals, regime="normal", depth_m=3000
        )
    except Exception:
        pass

    result = prediction_safety_check(df, inversion_result=inv_result, fast=True)
    return _sanitize_for_json(result)


# ── Field Consistency ────────────────────────────────

@app.post("/api/analysis/field-consistency")
async def run_field_consistency(request: Request):
    """Assess physical consistency of results across all wells.

    Checks SHmax alignment, fracture type similarity, and recommends
    whether wells should be analyzed separately or together.
    """
    t0 = time.monotonic()
    body = await request.json()
    source = body.get("source", "demo")
    depth_m = float(body.get("depth", 3000))

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")

    result = await asyncio.to_thread(field_consistency_check, df, depth_m)
    elapsed = round(time.monotonic() - t0, 2)

    if "error" not in result:
        _audit_record("field_consistency", {"depth_m": depth_m},
                      {"shmax_consistency": result.get("shmax_consistency"),
                       "recommendation": result.get("recommendation")},
                      source=source, elapsed_s=elapsed)

    result["elapsed_s"] = elapsed
    return _sanitize_for_json(result)


# ── Research Methods ─────────────────────────────────

@app.get("/api/research/methods")
async def get_research_methods():
    """Return a summary of scientific methods and 2025-2026 research integrated."""
    return research_methods_summary()


@app.post("/api/analysis/physics-check")
async def run_physics_check(request: Request):
    """Validate inversion results against physical constraints."""
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth_m = float(body.get("depth", 3000))

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    df_well = df[df[WELL_COL] == well].reset_index(drop=True) if well else df

    normals = fracture_plane_normal(df_well[AZIMUTH_COL].values, df_well[DIP_COL].values)
    inv_result = await asyncio.to_thread(invert_stress, normals, regime="normal", depth_m=depth_m)
    result = physics_constraint_check(inv_result, depth_m)
    return _sanitize_for_json(result)


@app.post("/api/analysis/physics-predict")
async def run_physics_constrained_predict(request: Request):
    """Physics-constrained ML prediction: integrates physical constraints
    directly into the prediction confidence scoring.

    Unlike the standard classify endpoint, this adjusts per-sample confidence
    based on whether the underlying stress inversion is physically consistent.
    Predictions that conflict with physics are flagged.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth_m = float(body.get("depth", 3000))
    fast = body.get("fast", True)

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    df_well = df[df[WELL_COL] == well].reset_index(drop=True) if well else df

    cache_key = f"phys_{source}_{well}_{depth_m}"
    if cache_key in _physics_predict_cache:
        return _physics_predict_cache[cache_key]

    result = await asyncio.to_thread(
        physics_constrained_predict, df_well,
        inversion_result=None, depth_m=depth_m, fast=fast,
    )
    _audit_record("physics_constrained_predict", {"well": well, "depth": depth_m}, result)
    response = _sanitize_for_json(result)
    _physics_predict_cache[cache_key] = response
    return response


@app.post("/api/analysis/misclassification")
async def run_misclassification_analysis(request: Request):
    """Analyze WHERE and WHY the model fails.

    Critical for the RLHF feedback loop: shows which fracture types
    are confused, at what depths/orientations errors occur, and provides
    actionable recommendations for improvement.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well")
    fast = body.get("fast", True)

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    if well:
        df = df[df[WELL_COL] == well].reset_index(drop=True)

    cache_key = f"misclass_{source}_{well}_{len(df)}"
    if cache_key in _misclass_cache:
        return _misclass_cache[cache_key]

    result = await asyncio.to_thread(misclassification_analysis, df, fast=fast)
    response = _sanitize_for_json(result)
    _misclass_cache[cache_key] = response
    return response


@app.post("/api/analysis/evidence-chain")
async def run_evidence_chain(request: Request):
    """Generate comprehensive evidence chain for stakeholder decisions.

    Shows WHAT was concluded, WHY (evidence), HOW CONFIDENT,
    WHAT COULD GO WRONG, and WHAT TO DO NEXT — for every conclusion.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth_m = float(body.get("depth", 3000))

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    df_well = df[df[WELL_COL] == well].reset_index(drop=True) if well else df

    task_id = body.get("task_id", "")

    def _progress_cb(step, pct, detail=""):
        if task_id:
            _emit_progress(task_id, step, pct, detail)

    result = await asyncio.to_thread(
        evidence_chain_analysis, df_well,
        well_name=well, depth_m=depth_m,
        progress_fn=_progress_cb,
    )

    if task_id:
        _emit_progress(task_id, "Complete", 100)

    _audit_record("evidence_chain", {"well": well, "depth": depth_m}, result)
    return _sanitize_for_json(result)


@app.post("/api/analysis/model-bias")
async def run_model_bias_detection(request: Request):
    """Detect systematic biases in model predictions.

    Shows whether the model over-predicts certain types, has
    depth-dependent accuracy, or other systematic issues.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well")
    fast = body.get("fast", True)

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    if well:
        df = df[df[WELL_COL] == well].reset_index(drop=True)

    result = await asyncio.to_thread(model_bias_detection, df, fast=fast)
    return _sanitize_for_json(result)


@app.post("/api/analysis/reliability-report")
async def run_reliability_report(request: Request):
    """Generate comprehensive prediction reliability report.

    Combines accuracy, bias, limitations, and improvement roadmap
    into a single decision-support document.
    """
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth_m = float(body.get("depth", 3000))
    fast = body.get("fast", True)

    df = get_df(source)
    if df is None:
        raise HTTPException(400, "No data loaded")
    df_well = df[df[WELL_COL] == well].reset_index(drop=True) if well else df

    result = await asyncio.to_thread(
        prediction_reliability_report, df_well,
        well_name=well, depth_m=depth_m, fast=fast,
    )
    return _sanitize_for_json(result)


# ── Audit Trail ──────────────────────────────────────

@app.get("/api/audit/log")
async def get_audit_log(limit: int = 50, offset: int = 0):
    """Get the prediction audit trail for regulatory compliance.

    Every analysis action is recorded with timestamp, parameters,
    result hash, and timing. Returns most recent entries first.
    """
    with _audit_lock:
        entries = list(_audit_log)
    entries.reverse()  # Most recent first
    total = len(entries)
    page = entries[offset:offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": _sanitize_for_json(page),
    }


@app.post("/api/audit/export")
async def export_audit_log():
    """Export full audit log as CSV for regulatory archival."""
    with _audit_lock:
        entries = list(_audit_log)
    if not entries:
        return {"csv": "", "rows": 0}

    rows = []
    for e in entries:
        rows.append({
            "id": e["id"],
            "timestamp": e["timestamp"],
            "action": e["action"],
            "source": e["source"],
            "well": e["well"],
            "parameters": json.dumps(e["parameters"]),
            "result_hash": e["result_hash"],
            "elapsed_s": e["elapsed_s"],
            "app_version": e["app_version"],
        })
    audit_df = pd.DataFrame(rows)
    csv_str = audit_df.to_csv(index=False)
    return {"csv": csv_str, "rows": len(rows), "filename": "audit_trail.csv"}
