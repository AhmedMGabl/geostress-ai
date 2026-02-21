# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GeoStress AI** - An AI-powered geostress inversion and fracture analysis tool for the oil & gas industry. It estimates the in-situ stress tensor from borehole fracture orientation data using Mohr-Coulomb theory (with pore pressure correction), multi-model ML comparison, and Bayesian inference.

**Live**: https://geostress-ai.onrender.com/
**Repo**: https://github.com/AhmedMGabl/geostress-ai

## Data

- **Source**: Borehole image log fracture measurements from 2 wells (3P, 6P) with PETRONAS origin
- **Format**: Excel files in `data/raw/` with columns: Depth(m), Azimuth(deg), Dip(deg)
- **Fracture types**: Boundary, Brecciated, Continuous, Discontinuous, Vuggy
- **Total**: ~1,022 fracture measurements
- **Reference PDF**: `references/AI_potentiality.pdf` - structural geology background
- **Methodology**: `data/raw/Geostress_DEL_16Feb26.docx` - geostress inversion methodology

## Architecture

```
app.py              - FastAPI backend (v2.9), all API endpoints, serves templates
src/
  data_loader.py    - Load Excel files, parse fracture orientation data, compute normals
  geostress.py      - Stress tensor construction, Mohr-Coulomb (with Pp), Bayesian MCMC, auto regime detection
  fracture_analysis.py - Original ML classification (RF, GB), clustering (KMeans)
  enhanced_analysis.py - Multi-model comparison (6 models), physics-informed features,
                         uncertainty quantification, decision support, feedback loop
  visualization.py  - Rose diagrams, stereonets, Mohr circles, tendency plots, dashboards
templates/
  index.html        - SPA frontend (Bootstrap 5)
static/
  css/style.css     - Dark sidebar, earth-tone theme
  js/app.js         - All API calls, tab switching, result rendering
render.yaml         - Render.com deployment config
notebooks/
  01_full_analysis.ipynb - Complete analysis pipeline demo
```

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run web app locally
uvicorn app:app --reload --port 8000

# Run quick data load test
cd src && python data_loader.py

# Test enhanced analysis module directly
python -c "from src.enhanced_analysis import compare_models; from src.data_loader import load_all_fractures; print(compare_models(load_all_fractures('data/raw'))['ranking'])"
```

## API Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/` | Serve SPA |
| GET | `/api/data/summary` | Data summary |
| GET | `/api/data/wells` | Well list with stats |
| GET | `/api/data/quality` | Data quality validation (score/grade) |
| POST | `/api/data/upload` | Upload Excel file |
| GET | `/api/viz/rose` | Rose diagram |
| GET | `/api/viz/stereonet` | Stereonet plot |
| GET | `/api/viz/depth-profile` | Depth profile |
| POST | `/api/analysis/inversion` | Stress inversion (regime=auto runs all 3, picks best) |
| POST | `/api/analysis/classify` | ML classification (enhanced features) |
| POST | `/api/analysis/cluster` | Fracture clustering |
| POST | `/api/analysis/compare-models` | Multi-model comparison (6 models) |
| POST | `/api/feedback/submit` | Expert feedback submission |
| GET | `/api/feedback/summary` | Feedback analytics + insights |
| POST | `/api/feedback/correct-label` | Expert label correction |
| POST | `/api/feedback/retrain` | Retrain model with corrections |
| POST | `/api/analysis/shap` | SHAP explainability (feature attribution) |
| POST | `/api/analysis/sensitivity` | Parameter sensitivity analysis (tornado diagram) |
| POST | `/api/analysis/risk-matrix` | Comprehensive operational risk assessment |
| POST | `/api/analysis/compare-wells` | Multi-well comparison and cross-validation |
| POST | `/api/report/well` | Generate stakeholder well report |
| POST | `/api/analysis/bayesian` | Bayesian MCMC uncertainty (posterior CIs) |
| POST | `/api/analysis/overview` | Quick auto-analysis on page load |
| POST | `/api/analysis/uncertainty-budget` | Uncertainty source ranking + recommendations |
| POST | `/api/analysis/active-learning` | Active learning: most uncertain samples for review |
| GET | `/api/analysis/features` | Enhanced feature info |
| POST | `/api/export/data` | Export fracture data as CSV |
| POST | `/api/export/inversion` | Export inversion results + tendencies as CSV |
| POST | `/api/analysis/ood-check` | Out-of-distribution detection (uploaded vs demo) |
| POST | `/api/analysis/calibration` | Model probability calibration assessment (ECE, Brier) |
| POST | `/api/data/recommendations` | Actionable data collection recommendations |
| POST | `/api/analysis/learning-curve` | Learning curve: accuracy vs data size + projections |
| POST | `/api/analysis/bootstrap-ci` | Bootstrap 95% CIs for per-class metrics (200 resamples) |
| POST | `/api/analysis/scenarios` | What-if scenario comparison (2-6 regimes side-by-side) |
| GET | `/api/audit/log` | Prediction audit trail (timestamps, hashes, parameters) |
| POST | `/api/audit/export` | Export audit log as CSV for regulatory archival |
| POST | `/api/analysis/hierarchical` | Hierarchical 2-level classification (rare vs common) |
| POST | `/api/feedback/trust-score` | Composite trust score from 5 signals (RLHF-style) |
| POST | `/api/analysis/expert-ensemble` | Expert-weighted ensemble with feedback adjustment |
| POST | `/api/analysis/monte-carlo` | Monte Carlo uncertainty propagation (measurement errors) |
| POST | `/api/analysis/cross-well-cv` | Leave-one-well-out cross-validation |
| POST | `/api/data/validate-constraints` | Domain constraint validation (physical/geological) |
| POST | `/api/analysis/executive-summary` | Plain-language executive summary for non-technical stakeholders |
| POST | `/api/data/sufficiency` | Data sufficiency assessment per analysis type |
| POST | `/api/analysis/safety-check` | Prediction safety: go/no-go with failure mode detection |
| POST | `/api/analysis/field-consistency` | Cross-well SHmax/type consistency, separate vs combined recommendation |
| GET | `/api/research/methods` | Scientific methods and 2025-2026 research basis |
| POST | `/api/analysis/physics-check` | Physics constraint validation (Byerlee, stress ordering) |
| POST | `/api/analysis/physics-predict` | Physics-constrained ML prediction with adjusted confidence |
| POST | `/api/analysis/misclassification` | WHERE/WHY model fails: confused pairs, depth/dip patterns |
| POST | `/api/analysis/evidence-chain` | Full evidence chain for every conclusion (stakeholder decisions) |
| POST | `/api/analysis/model-bias` | Systematic bias detection (class, depth, dip biases) |
| POST | `/api/analysis/reliability-report` | Prediction reliability grade (A-D) with improvement roadmap |
| GET | `/api/cache/status` | Cache sizes for performance monitoring |
| POST | `/api/analysis/guided-wizard` | 5-step guided analysis pipeline (Data→Stress→Risk→Model→Decision) |
| POST | `/api/analysis/what-if` | Quick single-inversion with user-specified friction/Pp/depth |
| POST | `/api/export/pdf-report` | Multi-page PDF report for stakeholder distribution |
| POST | `/api/analysis/predict-with-abstention` | Safety-critical: refuse prediction when confidence < threshold |
| POST | `/api/analysis/batch` | Run stress+classification+risk for all wells at once |
| POST | `/api/data/anomaly-detection` | Flag individual suspicious measurements (IQR, duplicates, gaps) |
| POST | `/api/feedback/effectiveness` | Track measurable impact of expert corrections on accuracy |
| GET | `/api/progress/{task_id}` | SSE progress streaming for long-running operations |

## Domain Concepts

- **Fracture azimuth**: Direction the fracture plane dips toward (0-360°, clockwise from North)
- **Fracture dip**: Angle of the fracture plane from horizontal (0-90°)
- **Stress tensor**: 3x3 symmetric matrix with principal stresses σ1 > σ2 > σ3
- **R ratio**: (σ2-σ3)/(σ1-σ3), shape of the stress ellipsoid, range [0,1]
- **SHmax**: Maximum horizontal stress azimuth - key output for drilling decisions
- **Mohr-Coulomb**: τ = c + μ(σn - Pp) (effective stress: pore pressure reduces friction)
- **Pore pressure (Pp)**: Fluid pressure in rock pores, estimated hydrostatic = ρw·g·h
- **Slip tendency**: τ/σn - how close a fracture is to sliding
- **Dilation tendency**: (σ1-σn)/(σ1-σ3) - how likely a fracture is to open
- **Critically stressed**: Fractures above the Mohr-Coulomb line - likely fluid conduits

## Technical Notes

- Azimuth data is circular (0° ≈ 360°) - use sin/cos decomposition for ML features
- The `.xls` files (xlrd engine) have 3 columns (depth, azimuth, dip); `.xlsx` files sometimes have only 2
- Column headers in Excel files are actually the first data row (numeric values used as column names)
- Inversion uses `scipy.differential_evolution` with pore pressure in objective function
- Bayesian MCMC uses `emcee` package with 5 parameters: σ1, σ3, R, SHmax_azimuth, μ
- Enhanced features: 21 columns including pore pressure, overburden, temperature, fracture density, fabric eigenvalues
- Stacking ensemble (RF+XGBoost+LightGBM with LR meta-learner) is typically the best model
- SHAP TreeExplainer for XGBoost/LightGBM/RF; GradientBoosting only supports binary in SHAP
- Conformal prediction provides calibrated per-sample confidence scores
- Model comparison caches results per data source to avoid redundant computation
- Fast mode (100 estimators, 3-fold CV) gives ~3x speedup with <0.5% accuracy loss
- All matplotlib plots use `threading.Lock` for thread safety + `asyncio.to_thread` for async
- `matplotlib.use("Agg")` is required at app.py top for headless rendering on Render
- Sensitivity analysis varies friction, pore pressure, and regime to produce tornado diagrams
- Risk matrix combines critically stressed %, data quality, model confidence, sensitivity, and friction into go/no-go
- Well 6P only has 2 fracture types (Vuggy, Brecciated) vs 5 in 3P — cross-well models don't transfer
- SHmax varies ~134° between wells 3P and 6P — possible structural domain boundary
- Bayesian MCMC fast mode: 500 steps/150 burnin/16 walkers (~1-2s); full: 3000 steps/500 burnin/32 walkers (~8s)
- Bayesian SHmax 90% CI can be very wide (>300°) — point estimates are over-confident
- Uncertainty budget aggregates 6 sources: parameter sensitivity, Bayesian, data quality, ML confidence, cross-well, pore pressure
- Pore pressure is consistently the #1 uncertainty driver (0-100% critically stressed range)
- Including Bayesian in uncertainty budget reduces that source's score from 70 to ~42
- Active learning uses entropy + margin sampling to rank fractures by expert-review value
- Inversion results are cached by (source, well, regime, depth, pp) — cleared on data upload
- Learning curve plateaus at ~87% with current features — suggests feature engineering matters more than more data
- Boundary (13) and Continuous (46) fracture types are under-represented vs median 189
- Auto regime detection runs all 3 regimes (~6s), ranks by total Mohr-Coulomb misfit
- Both wells show LOW confidence for regime (ratio ~1.03-1.04) — data alone doesn't constrain regime
- Export endpoints return CSV strings for browser download (no server-side file creation)
- SMOTE oversampling applied when minority/majority ratio < 15% and min class >= 6 samples
- SMOTE improved Boundary F1 from 0% to ~22-38%, Continuous from 0% to ~14-30%
- MLP with SMOTE achieves best balanced accuracy (69.2%) despite lower standard accuracy
- Wells 3P and 6P are highly OOD from each other (95% Mahalanobis, 72% Isolation Forest outlier)
- Model calibration is EXCELLENT (ECE=2.7%) — RF probability estimates can be trusted
- `_sanitize_for_json` must handle np.bool_ (not caught by np.integer/np.floating)
- imbalanced-learn required for SMOTE (added to requirements.txt)
- Glossary tab provides plain-language explanations for non-technical stakeholders
- Data recommendations identify specific actions: min 30 samples/class, sparse depth zones, well diversity
- Overview endpoint runs stress/calibration/recommendations in parallel via asyncio.gather (saves ~1s)
- Cached overview runs in < 1s; first-time auto-regime detection takes ~6s
- Learning curve shows SLOWING convergence at ~87% accuracy — more data of same type won't help much
- 85% accuracy projected to need ~7,500 total samples (vs current 1,022)
- 95% accuracy is UNLIKELY with current features — need different data or features
- Bootstrap CIs: overall accuracy 86.9% [84.1-89.6], Boundary F1 3.7% [0-33.5%] — very unreliable
- Scenario comparison runs 2-6 stress inversions (~2s each) and compares metrics
- Normal fault regime has lowest misfit for Well 3P, but all regimes similar
- Audit trail uses deque(maxlen=1000) + SHA-256 hash for integrity
- Every inversion, overview, and scenario comparison is audit-logged
- `invert_stress` returns numpy scalars — use `_scalar()` helper to safely convert to float
- Hierarchical classification dramatically improves rare class detection: Boundary F1 11%→100%, Continuous 9%→99%
- Hierarchical balanced accuracy: 99.6% vs flat 56.6% — but on training data, real CV will be lower
- Server-side rendered charts: plot_model_comparison, plot_learning_curve, plot_bootstrap_ci in visualization.py
- Charts returned as base64 PNG in API responses (comparison_chart_img, chart_img keys)
- Trust score: weighted combination of expert_feedback(30%), data_quality(25%), corrections(15%), sample_size(15%), calibration(15%)
- Trust score without expert feedback: ~73.6 (MODERATE) — feedback shifts it significantly
- Expert ensemble: 6 models with accuracy-proportional weights; expert feedback adjusts via rating_factor * model_bias
- Cross-well CV: Wells 3P and 6P show POOR transferability (7.5% cross vs 81.6% within) — completely different fracture populations
- Monte Carlo: each simulation runs full differential_evolution inversion (~7s), fast mode limited to 30 sims
- Monte Carlo SHmax CI is very wide (~185°) for Well 3P Normal — SHmax direction is poorly constrained
- Dip uncertainty is #1 sensitivity driver for SHmax (>azimuth), depth has no effect on direction
- Domain validation catches: depth gaps, class imbalance, physically impossible values, distribution anomalies
- `fracture_plane_normal()` takes DEGREES (not radians) — don't double-convert
- Cross-well model fails because 6P only has 2 fracture types vs 5 in 3P — well-specific models needed
- Safety check detects: data anomalies (IQR outliers), high misfit (>0.5 = NO-GO), extreme R-ratio, unusual friction
- Field consistency: SHmax between 3P/6P is CONSISTENT (15° diff), but fracture types are DIFFERENT (2/5 shared)
- Field recommendation: SEPARATE analysis — combining wells would obscure real geological differences
- Executive summary uses traffic-light risk: RED (>30% CS or trust <40), AMBER (>10% CS or trust <60), GREEN
- `_azimuth_to_direction()` converts degrees to cardinal (N/NNE/NE/etc.) for plain-language descriptions
- Data sufficiency: 3/5 analyses READY with current 1022 samples, ML and cross-well are MARGINAL
- `auto_detect_regime` must be called separately before `invert_stress` — regime="auto" not valid for invert_stress
- `auto_detect_regime` returns dict with `all_results` key (not `results`), misfit values are numpy arrays
- Physics constraint check: validates Byerlee's friction (0.4-0.85), stress ordering, frictional equilibrium, R-ratio
- Physics-constrained predict: adjusts ML confidence by physics_score (0-1), penalizes violations/warnings
- Misclassification analysis: Continuous→Discontinuous is the biggest confusion (78.3%)
- Model bias: HIGH for Well 3P — dip-dependent accuracy (74% low-dip vs lower high-dip)
- Prediction reliability: Grade C for Well 3P (64.8% accuracy, 5 limitations)
- Evidence chain: 6 items covering data quality, regime, SHmax, critically stressed, physics, ML accuracy
- Response timing middleware logs SLOW requests (>2s) to server console
- Cache keys include source+well+params; cleared on upload; cache status endpoint at GET /api/cache/status
- `_audit_record()` takes 3 positional args: action, params, result_summary (not 2)
- `data_sufficiency_check` returns `analyses` as a LIST of dicts (not a dict keyed by name)
- Guided wizard: 5 steps (Data/Stress/Risk/Model/Decision), each PASS/WARN/FAIL, overall HALT/CAUTION/PROCEED_WITH_REVIEW/PROCEED
- What-if endpoint runs quick inversion with user-specified friction/Pp/depth, shows risk impact
- PDF report uses matplotlib PdfPages: cover, executive summary, rose+pole, confusion matrix, recommendations
- Upload uses original filename in temp dir (not NamedTemporaryFile) so parse_filename can extract well/type
- Upload validation returns quality score, sufficiency status, domain warnings, preview stats, OOD check
- Confusion matrix plot: row-normalized (recall-based) heatmap with raw count + percentage annotations
- 9 caches: inversion, model_comparison, auto_regime, classify, misclass, physics_predict, shap, sensitivity, wizard
- Prediction abstention: model refuses when max(predict_proba) < threshold (default 60%), flags for expert review
- Well 3P at 60% threshold: 49% abstained, +13.2% accuracy gain on confident predictions (79% vs 66%)
- Well 6P at 50% threshold: 0% abstained (only 2 classes = easy separation, 100% confident)
- Abstained samples show tentative prediction + top-2 candidates with probabilities for expert guidance
- `_safe_float()` helper in abstention handles NaN depths (Well 6P has NaN depth values)
- Batch analysis runs stress+classify+risk for all wells; uses `count_critical` key (not `n_critical`)
- Overview endpoint wraps each sub-analysis with `asyncio.wait_for` timeout (stress=10s, cal=5s, recs=3s)
- Anomaly detection flags: physical impossibility, IQR outliers (2.5x), duplicates, depth gaps, low-dip uncertainty
- Well 6P anomaly: 79.6% flagged (494 missing depths, 27 outliers) — data quality issue surfaced before analysis
- Feedback effectiveness: shows per-class correction priority (Continuous 17%, Boundary 23% accuracy = HIGH priority)
- 66 total API routes at v2.9
