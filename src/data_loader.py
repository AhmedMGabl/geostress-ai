"""
Data loading and preprocessing for fracture orientation data.

Reads Excel files containing fracture measurements (depth, azimuth, dip)
from borehole image logs and produces clean, labeled DataFrames.
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path


# Column naming convention for all loaded data
DEPTH_COL = "depth_m"
AZIMUTH_COL = "azimuth_deg"
DIP_COL = "dip_deg"
WELL_COL = "well"
FRACTURE_TYPE_COL = "fracture_type"


def parse_filename(filename: str) -> tuple[str, str]:
    """Extract well name and fracture type from filename.

    Filenames follow pattern: {well}_{type}_Fracture.xls(x)
    e.g., '3P_Vuggy_Fracture.xls' -> ('3P', 'Vuggy')
          '3p_Boundary_Fracture.xlsx' -> ('3P', 'Boundary')
    """
    stem = Path(filename).stem  # Remove extension
    # Pattern: WellName_FractureType_Fracture
    match = re.match(r"(\d+[Pp])_(\w+)_Fracture", stem)
    if match:
        well = match.group(1).upper()
        frac_type = match.group(2).capitalize()
        return well, frac_type
    raise ValueError(f"Cannot parse filename: {filename}")


def load_single_file(filepath: str) -> pd.DataFrame:
    """Load a single Excel file and standardize columns.

    The Excel files have no proper headers - the first row of data
    appears as column names. Files have either 2 columns (azimuth, dip)
    or 3 columns (depth, azimuth, dip).
    """
    path = Path(filepath)
    engine = "xlrd" if path.suffix == ".xls" else "openpyxl"

    df = pd.read_excel(filepath, engine=engine)

    # The "column names" are actually the first data row (numeric values)
    first_row = pd.DataFrame([df.columns.tolist()], columns=range(len(df.columns)))
    first_row = first_row.apply(pd.to_numeric, errors="coerce")

    df.columns = range(len(df.columns))
    df = pd.concat([first_row, df], ignore_index=True)

    if len(df.columns) == 3:
        df.columns = [DEPTH_COL, AZIMUTH_COL, DIP_COL]
    elif len(df.columns) == 2:
        df.columns = [AZIMUTH_COL, DIP_COL]
        df[DEPTH_COL] = np.nan
    else:
        raise ValueError(f"Unexpected column count ({len(df.columns)}) in {filepath}")

    # Add metadata from filename
    well, frac_type = parse_filename(path.name)
    df[WELL_COL] = well
    df[FRACTURE_TYPE_COL] = frac_type

    return df


def load_all_fractures(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load all fracture data files and combine into a single DataFrame.

    Returns DataFrame with columns:
        depth_m, azimuth_deg, dip_deg, well, fracture_type
    """
    data_dir = Path(data_dir)
    frames = []

    # Prefer .xls files (they have depth), fall back to .xlsx
    seen = set()
    for ext in [".xls", ".xlsx"]:
        for fp in sorted(data_dir.glob(f"*_Fracture{ext}")):
            key = parse_filename(fp.name)
            if key not in seen:
                seen.add(key)
                frames.append(load_single_file(str(fp)))

    if not frames:
        raise FileNotFoundError(f"No fracture files found in {data_dir}")

    df = pd.concat(frames, ignore_index=True)

    # Ensure azimuth is in [0, 360)
    df[AZIMUTH_COL] = df[AZIMUTH_COL] % 360

    # Ensure dip is in [0, 90]
    df[DIP_COL] = df[DIP_COL].clip(0, 90)

    return df


def fracture_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print summary statistics of loaded fracture data."""
    summary = df.groupby([WELL_COL, FRACTURE_TYPE_COL]).agg(
        count=(DIP_COL, "count"),
        depth_min=(DEPTH_COL, "min"),
        depth_max=(DEPTH_COL, "max"),
        azimuth_mean=(AZIMUTH_COL, "mean"),
        dip_mean=(DIP_COL, "mean"),
    ).round(1)
    return summary


def fracture_plane_normal(azimuth_deg: np.ndarray, dip_deg: np.ndarray) -> np.ndarray:
    """Compute unit normal vectors for fracture planes.

    Convention (right-hand rule, dip direction = azimuth):
        n_x = sin(azimuth) * sin(dip)   (East)
        n_y = cos(azimuth) * sin(dip)   (North)
        n_z = cos(dip)                   (Vertical, positive down)

    Parameters
    ----------
    azimuth_deg : array of dip direction azimuths in degrees (0-360)
    dip_deg : array of dip angles in degrees (0-90)

    Returns
    -------
    normals : (N, 3) array of unit normal vectors [East, North, Down]
    """
    az = np.radians(azimuth_deg)
    dip = np.radians(dip_deg)

    nx = np.sin(az) * np.sin(dip)
    ny = np.cos(az) * np.sin(dip)
    nz = np.cos(dip)

    return np.column_stack([nx, ny, nz])


def qc_fracture_data(df: pd.DataFrame,
                     min_dip: float = 5.0,
                     max_dip: float = 85.0,
                     min_count_per_well: int = 10,
                     depth_gap_threshold_m: float = 200.0) -> dict:
    """Apply WSM-standard QC filters to fracture orientation data.

    Based on EAGE borehole image log QC standards and WSM 2025 criteria.

    Filters:
      - Low dip (<5°): likely bedding planes, not tectonic fractures
      - High dip (>85°): may be drilling-induced features in deviated wells
      - Insufficient data: wells with <10 fractures unreliable for stress
      - Missing depth: fractures without depth cannot be used for profiles
      - Depth gaps: zones >200m without data suggest incomplete coverage

    Returns QC report with per-fracture flags and overall pass rate.
    """
    qc_flags = pd.Series("PASS", index=df.index)

    # Flag near-horizontal picks (likely bedding, not tectonic fractures)
    low_dip = df[DIP_COL] < min_dip
    qc_flags[low_dip] = "LOW_DIP_EXCLUDED"

    # Flag near-vertical picks (may be drilling-induced in deviated wells)
    high_dip = df[DIP_COL] > max_dip
    qc_flags[high_dip] = "HIGH_DIP_REVIEW"

    # Flag fractures with missing/NaN depth
    missing_depth = df[DEPTH_COL].isna()
    qc_flags[missing_depth] = "MISSING_DEPTH"

    # Flag wells with insufficient data count
    for well in df[WELL_COL].unique():
        mask = df[WELL_COL] == well
        if mask.sum() < min_count_per_well:
            qc_flags[mask] = "INSUFFICIENT_DATA"

    # Detect depth coverage gaps per well
    depth_gaps = {}
    for well in df[WELL_COL].unique():
        well_df = df[df[WELL_COL] == well].dropna(subset=[DEPTH_COL])
        if len(well_df) < 2:
            continue
        sorted_depths = well_df[DEPTH_COL].sort_values().values
        gaps = np.diff(sorted_depths)
        large_gaps = gaps > depth_gap_threshold_m
        if large_gaps.any():
            gap_info = []
            gap_idx = np.where(large_gaps)[0]
            for gi in gap_idx:
                gap_info.append({
                    "top_m": round(float(sorted_depths[gi]), 1),
                    "bottom_m": round(float(sorted_depths[gi + 1]), 1),
                    "gap_m": round(float(gaps[gi]), 1),
                })
            depth_gaps[well] = gap_info

    # Azimuth scatter assessment per well (circular std)
    azimuth_quality = {}
    for well in df[WELL_COL].unique():
        well_df = df[(df[WELL_COL] == well) & (qc_flags == "PASS")]
        if len(well_df) < 2:
            azimuth_quality[well] = {"circular_std_deg": None, "preferred_orientation": False}
            continue
        az_rad = np.radians(well_df[AZIMUTH_COL].values)
        R_len = np.sqrt(np.mean(np.sin(az_rad))**2 + np.mean(np.cos(az_rad))**2)
        circ_std = np.degrees(np.sqrt(-2 * np.log(max(R_len, 1e-10))))
        azimuth_quality[well] = {
            "circular_std_deg": round(circ_std, 1),
            "resultant_length": round(R_len, 3),
            "preferred_orientation": R_len > 0.3,  # Rayleigh test threshold
        }

    passed = (qc_flags == "PASS").sum()
    flag_counts = qc_flags.value_counts().to_dict()

    return {
        "total": len(df),
        "passed": int(passed),
        "pass_rate": round(passed / max(len(df), 1), 3),
        "flags": flag_counts,
        "depth_gaps": depth_gaps,
        "azimuth_quality": azimuth_quality,
        "qc_flags": qc_flags,
        "min_dip_filter": min_dip,
        "max_dip_filter": max_dip,
        "wsm_note": ("QC based on WSM 2025 and EAGE borehole image log standards. "
                     "LOW_DIP fractures are likely bedding. HIGH_DIP may be "
                     "drilling-induced. Both are excluded from stress analysis."),
    }


if __name__ == "__main__":
    # Quick test
    df = load_all_fractures()
    print(f"Loaded {len(df)} fractures from {df[WELL_COL].nunique()} wells")
    print(f"Fracture types: {df[FRACTURE_TYPE_COL].unique().tolist()}")
    print()
    print(fracture_summary(df))
    print()
    qc = qc_fracture_data(df)
    print(f"QC: {qc['passed']}/{qc['total']} passed ({qc['pass_rate']*100:.1f}%)")
    print(f"Flags: {qc['flags']}")
