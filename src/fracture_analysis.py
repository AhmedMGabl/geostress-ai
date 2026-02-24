"""
Fracture Analysis Module.

ML-based fracture type classification and clustering analysis
using orientation data (azimuth, dip, depth).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN

try:
    from data_loader import (
        DEPTH_COL, AZIMUTH_COL, DIP_COL, WELL_COL, FRACTURE_TYPE_COL,
        fracture_plane_normal, circular_mean_deg, circular_std_deg,
    )
except ImportError:
    from .data_loader import (
        DEPTH_COL, AZIMUTH_COL, DIP_COL, WELL_COL, FRACTURE_TYPE_COL,
        fracture_plane_normal, circular_mean_deg, circular_std_deg,
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from fracture orientation data for ML.

    Converts angular data to Cartesian components to avoid
    the circular discontinuity problem (0° ≈ 360°).
    """
    feat = pd.DataFrame(index=df.index)

    az_rad = np.radians(df[AZIMUTH_COL])
    dip_rad = np.radians(df[DIP_COL])

    # Cartesian components of fracture normal (avoids circular issues)
    feat["nx"] = np.sin(az_rad) * np.sin(dip_rad)
    feat["ny"] = np.cos(az_rad) * np.sin(dip_rad)
    feat["nz"] = np.cos(dip_rad)

    # Azimuth as sin/cos pair
    feat["az_sin"] = np.sin(az_rad)
    feat["az_cos"] = np.cos(az_rad)

    # Double-angle (useful for orientation tensor analysis)
    feat["az2_sin"] = np.sin(2 * az_rad)
    feat["az2_cos"] = np.cos(2 * az_rad)

    # Dip directly (no circular issue)
    feat["dip"] = df[DIP_COL]

    # Depth if available
    if DEPTH_COL in df.columns and df[DEPTH_COL].notna().any():
        feat["depth"] = df[DEPTH_COL]
        feat["depth"] = feat["depth"].fillna(feat["depth"].median())

    return feat


def classify_fracture_types(
    df: pd.DataFrame,
    classifier: str = "random_forest",
    n_folds: int = 5,
) -> dict:
    """Train a classifier to predict fracture type from orientation features.

    Parameters
    ----------
    df : DataFrame with fracture data
    classifier : 'random_forest' or 'gradient_boosting'
    n_folds : Number of cross-validation folds

    Returns
    -------
    dict with model, scores, feature importances, classification report
    """
    features = engineer_features(df)
    labels = df[FRACTURE_TYPE_COL].values
    le = LabelEncoder()
    y = le.fit_transform(labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    if classifier == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
        )
    elif classifier == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=42
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # Fit on all data for feature importances and report
    model.fit(X, y)
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": features.columns.tolist(),
        "cv_scores": scores,
        "cv_mean_accuracy": scores.mean(),
        "cv_std_accuracy": scores.std(),
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "feature_importances": dict(
            zip(features.columns, model.feature_importances_)
        ),
    }


def cluster_fracture_sets(
    df: pd.DataFrame,
    n_clusters: int = None,
    max_clusters: int = 8,
) -> dict:
    """Identify fracture sets using K-Means clustering on orientation data.

    If n_clusters is None, uses silhouette score to find optimal k.
    """
    from sklearn.metrics import silhouette_score

    features = engineer_features(df)
    # Use only orientation features for clustering (not depth)
    orient_cols = [c for c in features.columns if c != "depth"]
    X = StandardScaler().fit_transform(features[orient_cols].values)

    if n_clusters is None:
        # Find optimal k using silhouette score
        best_k, best_score = 2, -1
        scores = {}
        for k in range(2, max_clusters + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            scores[k] = score
            if score > best_score:
                best_k, best_score = k, score
        n_clusters = best_k
    else:
        scores = {}

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = km.fit_predict(X)

    # Compute mean orientation per cluster
    cluster_stats = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_stats.append({
            "cluster": c,
            "count": mask.sum(),
            "mean_azimuth": circular_mean_deg(df.loc[mask, AZIMUTH_COL].values),
            "mean_dip": df.loc[mask, DIP_COL].mean(),
            "std_azimuth": circular_std_deg(df.loc[mask, AZIMUTH_COL].values),
            "std_dip": df.loc[mask, DIP_COL].std(),
        })

    return {
        "n_clusters": n_clusters,
        "labels": cluster_labels,
        "silhouette_scores": scores,
        "cluster_stats": pd.DataFrame(cluster_stats),
        "model": km,
    }


def identify_critically_stressed(
    sigma_n: np.ndarray,
    tau: np.ndarray,
    mu: float = 0.6,
    cohesion: float = 0.0,
) -> np.ndarray:
    """Identify critically stressed fractures.

    A fracture is critically stressed if:
        τ ≥ cohesion + μ · σn

    These fractures are likely to be hydraulically conductive
    (fluid can flow through them).

    Returns boolean mask (True = critically stressed).
    """
    tau_critical = cohesion + mu * sigma_n
    return tau >= tau_critical


if __name__ == "__main__":
    from data_loader import load_all_fractures

    df = load_all_fractures("../data/raw")

    # Classification
    print("=== Fracture Type Classification ===")
    result = classify_fracture_types(df, classifier="random_forest")
    print(f"  CV Accuracy: {result['cv_mean_accuracy']:.3f} +/- {result['cv_std_accuracy']:.3f}")
    print(f"  Feature importances:")
    for feat, imp in sorted(result["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.3f}")

    # Clustering
    print("\n=== Fracture Set Clustering ===")
    for well in df[WELL_COL].unique():
        df_well = df[df[WELL_COL] == well].copy()
        clust = cluster_fracture_sets(df_well)
        print(f"\n  Well {well}: {clust['n_clusters']} fracture sets identified")
        print(clust["cluster_stats"].to_string(index=False))
