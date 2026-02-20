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


if __name__ == "__main__":
    # Quick test
    df = load_all_fractures()
    print(f"Loaded {len(df)} fractures from {df[WELL_COL].nunique()} wells")
    print(f"Fracture types: {df[FRACTURE_TYPE_COL].unique().tolist()}")
    print()
    print(fracture_summary(df))
