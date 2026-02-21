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
| POST | `/api/analysis/inversion` | Stress inversion (with Pp, interpretation) |
| POST | `/api/analysis/classify` | ML classification (enhanced features) |
| POST | `/api/analysis/cluster` | Fracture clustering |
| POST | `/api/analysis/compare-models` | Multi-model comparison (6 models) |
| POST | `/api/feedback/submit` | Expert feedback submission |
| GET | `/api/feedback/summary` | Feedback analytics + insights |
| POST | `/api/feedback/correct-label` | Expert label correction |
| POST | `/api/feedback/retrain` | Retrain model with corrections |
| POST | `/api/analysis/shap` | SHAP explainability (feature attribution) |
| GET | `/api/analysis/features` | Enhanced feature info |

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
