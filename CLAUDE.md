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
app.py              - FastAPI backend (v2.0), all API endpoints, serves templates
src/
  data_loader.py    - Load Excel files, parse fracture orientation data, compute normals
  geostress.py      - Stress tensor construction, Mohr-Coulomb (with Pp), Bayesian MCMC
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
