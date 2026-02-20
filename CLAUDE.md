# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GeoStress AI** - An AI-powered geostress inversion and fracture analysis tool for the oil & gas industry. It estimates the in-situ stress tensor from borehole fracture orientation data using Mohr-Coulomb theory, Bayesian inference, and machine learning.

## Data

- **Source**: Borehole image log fracture measurements from 2 wells (3P, 6P) with PETRONAS origin
- **Format**: Excel files in `data/raw/` with columns: Depth(m), Azimuth(deg), Dip(deg)
- **Fracture types**: Boundary, Brecciated, Continuous, Discontinuous, Vuggy
- **Total**: ~1,022 fracture measurements
- **Reference PDF**: `references/AI_potentiality.pdf` - structural geology background
- **Methodology**: `data/raw/Geostress_DEL_16Feb26.docx` - geostress inversion methodology

## Architecture

```
src/
  data_loader.py    - Load Excel files, parse fracture orientation data, compute normals
  geostress.py      - Stress tensor construction, Mohr-Coulomb inversion, Bayesian MCMC
  fracture_analysis.py - ML classification (Random Forest), clustering (KMeans), critically stressed ID
  visualization.py  - Rose diagrams, stereonets, Mohr circles, tendency plots, dashboards
notebooks/
  01_full_analysis.ipynb - Complete analysis pipeline demo
```

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick data load test
cd src && python data_loader.py

# Run geostress inversion
cd src && python geostress.py

# Run fracture classification + clustering
cd src && python fracture_analysis.py

# Generate dashboard visualizations
cd src && python visualization.py

# Launch Jupyter notebook
jupyter notebook notebooks/01_full_analysis.ipynb
```

## Domain Concepts

- **Fracture azimuth**: Direction the fracture plane dips toward (0-360°, clockwise from North)
- **Fracture dip**: Angle of the fracture plane from horizontal (0-90°)
- **Stress tensor**: 3x3 symmetric matrix with principal stresses σ1 > σ2 > σ3
- **R ratio**: (σ2-σ3)/(σ1-σ3), shape of the stress ellipsoid, range [0,1]
- **SHmax**: Maximum horizontal stress azimuth - key output for drilling decisions
- **Mohr-Coulomb**: τ = c + μσn (failure criterion - fractures slip when shear exceeds friction)
- **Slip tendency**: τ/σn - how close a fracture is to sliding
- **Dilation tendency**: (σ1-σn)/(σ1-σ3) - how likely a fracture is to open
- **Critically stressed**: Fractures above the Mohr-Coulomb line - likely fluid conduits

## Technical Notes

- Azimuth data is circular (0° ≈ 360°) - use sin/cos decomposition for ML features
- The `.xls` files (xlrd engine) have 3 columns (depth, azimuth, dip); `.xlsx` files sometimes have only 2
- Column headers in Excel files are actually the first data row (numeric values used as column names)
- Inversion uses `scipy.differential_evolution` for global optimization
- Bayesian MCMC uses `emcee` package with 5 parameters: σ1, σ3, R, SHmax_azimuth, μ
