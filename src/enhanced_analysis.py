"""
Enhanced Fracture Analysis Engine.

Multi-model comparison, physics-informed feature engineering,
uncertainty quantification, and stakeholder decision support.
Designed for production use in the oil & gas industry.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    cross_val_score, cross_val_predict, StratifiedKFold,
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score,
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from data_loader import (
        DEPTH_COL, AZIMUTH_COL, DIP_COL, WELL_COL, FRACTURE_TYPE_COL,
        fracture_plane_normal,
    )
except ImportError:
    from .data_loader import (
        DEPTH_COL, AZIMUTH_COL, DIP_COL, WELL_COL, FRACTURE_TYPE_COL,
        fracture_plane_normal,
    )


# ──────────────────────────────────────────────────────
# Subsurface Physics Constants
# ──────────────────────────────────────────────────────

WATER_DENSITY = 1020.0        # kg/m3 (formation water, slightly saline)
ROCK_DENSITY = 2500.0         # kg/m3 (average carbonate/sandstone)
GRAVITY = 9.81                # m/s2
GEOTHERMAL_GRADIENT = 0.030   # °C/m (typical 25-35 °C/km)
SURFACE_TEMP = 25.0           # °C


# ──────────────────────────────────────────────────────
# Enhanced Feature Engineering
# ──────────────────────────────────────────────────────

def compute_pore_pressure(depth_m: np.ndarray,
                          gradient: float = None) -> np.ndarray:
    """Estimate pore pressure from depth using hydrostatic gradient.

    Pp = ρ_water * g * depth (hydrostatic assumption).
    Returns pressure in MPa.

    In real applications, replace with measured pressure data
    or use company-specific pore pressure models (Eaton, Bowers, etc.).
    """
    if gradient is not None:
        return gradient * depth_m
    return WATER_DENSITY * GRAVITY * depth_m / 1e6


def compute_overburden(depth_m: np.ndarray,
                       density: float = ROCK_DENSITY) -> np.ndarray:
    """Estimate vertical (overburden) stress from depth.

    Sv = ρ_rock * g * depth.  Returns MPa.
    """
    return density * GRAVITY * depth_m / 1e6


def compute_temperature(depth_m: np.ndarray,
                        gradient: float = GEOTHERMAL_GRADIENT,
                        surface_temp: float = SURFACE_TEMP) -> np.ndarray:
    """Estimate formation temperature from depth."""
    return surface_temp + gradient * depth_m


def engineer_enhanced_features(df: pd.DataFrame,
                               window_m: float = 50.0) -> pd.DataFrame:
    """Create physics-informed features from fracture data.

    Adds:
    - Standard orientation features (normal vector, sin/cos encoding)
    - Estimated pore pressure, overburden stress, temperature
    - Fracture density within a depth window
    - Fracture spacing (distance to nearest neighbor in depth)
    - Orientation tensor eigenvalues (fabric strength)
    - Depth-normalized position within well
    """
    feat = pd.DataFrame(index=df.index)

    az_rad = np.radians(df[AZIMUTH_COL])
    dip_rad = np.radians(df[DIP_COL])

    # ── Standard orientation features ──
    feat["nx"] = np.sin(az_rad) * np.sin(dip_rad)
    feat["ny"] = np.cos(az_rad) * np.sin(dip_rad)
    feat["nz"] = np.cos(dip_rad)
    feat["az_sin"] = np.sin(az_rad)
    feat["az_cos"] = np.cos(az_rad)
    feat["az2_sin"] = np.sin(2 * az_rad)
    feat["az2_cos"] = np.cos(2 * az_rad)
    feat["dip"] = df[DIP_COL]

    # ── Depth and depth-derived features ──
    has_depth = DEPTH_COL in df.columns and df[DEPTH_COL].notna().any()
    if has_depth:
        depth = df[DEPTH_COL].fillna(df[DEPTH_COL].median()).values
        feat["depth"] = depth

        # Subsurface physics estimates
        feat["pore_pressure_mpa"] = compute_pore_pressure(depth)
        feat["overburden_mpa"] = compute_overburden(depth)
        feat["temperature_c"] = compute_temperature(depth)

        # Fracture density: count of fractures within ±window_m
        density = np.zeros(len(depth))
        for i, d in enumerate(depth):
            density[i] = np.sum(np.abs(depth - d) <= window_m) - 1
        feat["fracture_density"] = density

        # Fracture spacing: distance to nearest neighbor
        sorted_depths = np.sort(depth)
        spacing = np.zeros(len(depth))
        for i, d in enumerate(depth):
            idx = np.searchsorted(sorted_depths, d)
            neighbors = []
            if idx > 0:
                neighbors.append(d - sorted_depths[idx - 1])
            if idx < len(sorted_depths) - 1:
                neighbors.append(sorted_depths[idx + 1] - d)
            spacing[i] = min(neighbors) if neighbors else 0.0
        # Avoid exact zeros from duplicate depths
        spacing[spacing == 0] = 0.01
        feat["fracture_spacing"] = spacing
        feat["log_spacing"] = np.log1p(spacing)

        # Normalized depth within well (0 = shallowest, 1 = deepest)
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            feat["depth_normalized"] = (depth - d_min) / (d_max - d_min)
        else:
            feat["depth_normalized"] = 0.5

    # ── Orientation tensor (fabric strength) ──
    normals = np.column_stack([feat["nx"], feat["ny"], feat["nz"]])
    orientation_tensor = normals.T @ normals / len(normals)
    eigenvalues = np.sort(np.linalg.eigvalsh(orientation_tensor))[::-1]
    # Broadcast eigenvalues as constant features (per-dataset context)
    feat["fabric_e1"] = eigenvalues[0]
    feat["fabric_e2"] = eigenvalues[1]
    feat["fabric_e3"] = eigenvalues[2]
    # Woodcock K and C parameters
    if eigenvalues[2] > 1e-10 and eigenvalues[1] > 1e-10:
        feat["woodcock_K"] = np.log(eigenvalues[0] / eigenvalues[1]) / \
                             np.log(eigenvalues[1] / eigenvalues[2])
        feat["woodcock_C"] = np.log(eigenvalues[0] / eigenvalues[2])
    else:
        feat["woodcock_K"] = 0.0
        feat["woodcock_C"] = 0.0

    return feat


# ──────────────────────────────────────────────────────
# Multi-Model Classification
# ──────────────────────────────────────────────────────

def _get_models() -> dict:
    """Return all available classification models."""
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            random_state=42, class_weight="balanced", n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
        "svm": SVC(
            kernel="rbf", C=10.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=42,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            random_state=42,
        ),
    }

    if HAS_XGB:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="mlogloss",
        )

    if HAS_LGB:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )

    return models


def compare_models(
    df: pd.DataFrame,
    n_folds: int = 5,
    models_to_run: list = None,
) -> dict:
    """Run all models and return comparative metrics.

    Returns per-model: accuracy, F1, precision, recall, confusion matrix,
    feature importances (where available), and cross-val predictions.
    """
    features = engineer_enhanced_features(df)
    labels = df[FRACTURE_TYPE_COL].values
    le = LabelEncoder()
    y = le.fit_transform(labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_models = _get_models()

    if models_to_run:
        all_models = {k: v for k, v in all_models.items() if k in models_to_run}

    results = {}
    for name, model in all_models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Cross-validated predictions for confusion matrix
            y_pred_cv = cross_val_predict(model, X, y, cv=cv)

            # Scores per fold
            acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            f1_scores = cross_val_score(
                model, X, y, cv=cv, scoring="f1_weighted"
            )

            # Fit on full data for feature importances
            model.fit(X, y)
            y_pred_full = model.predict(X)

            # Feature importances
            feat_imp = {}
            if hasattr(model, "feature_importances_"):
                feat_imp = dict(zip(
                    features.columns, model.feature_importances_
                ))
            elif hasattr(model, "coef_"):
                # For SVM/linear models, use mean absolute coefficient
                coef = np.abs(model.coef_).mean(axis=0)
                feat_imp = dict(zip(features.columns, coef))

            results[name] = {
                "cv_accuracy_mean": float(acc_scores.mean()),
                "cv_accuracy_std": float(acc_scores.std()),
                "cv_accuracy_scores": acc_scores.tolist(),
                "cv_f1_mean": float(f1_scores.mean()),
                "cv_f1_std": float(f1_scores.std()),
                "cv_precision": float(precision_score(
                    y, y_pred_cv, average="weighted", zero_division=0
                )),
                "cv_recall": float(recall_score(
                    y, y_pred_cv, average="weighted", zero_division=0
                )),
                "confusion_matrix": confusion_matrix(y, y_pred_cv).tolist(),
                "feature_importances": {
                    k: round(float(v), 4) for k, v in feat_imp.items()
                },
                "class_names": le.classes_.tolist(),
                "train_accuracy": float(accuracy_score(y, y_pred_full)),
            }

    # Rank models
    ranked = sorted(
        results.items(), key=lambda x: x[1]["cv_accuracy_mean"], reverse=True
    )
    ranking = [
        {"rank": i + 1, "model": name, "accuracy": r["cv_accuracy_mean"],
         "f1": r["cv_f1_mean"]}
        for i, (name, r) in enumerate(ranked)
    ]

    # Model agreement analysis (uncertainty signal)
    all_preds = {}
    for name, model in all_models.items():
        model.fit(X, y)
        all_preds[name] = model.predict(X)

    pred_matrix = np.array(list(all_preds.values()))  # (n_models, n_samples)
    # For each sample, what fraction of models agree on the prediction?
    agreement = np.zeros(len(y))
    for i in range(len(y)):
        votes = pred_matrix[:, i]
        majority = np.bincount(votes).max()
        agreement[i] = majority / len(all_models)

    return {
        "models": results,
        "ranking": ranking,
        "best_model": ranked[0][0] if ranked else None,
        "feature_names": features.columns.tolist(),
        "n_samples": len(y),
        "n_features": X.shape[1],
        "model_agreement_mean": float(agreement.mean()),
        "model_agreement_min": float(agreement.min()),
        "low_confidence_count": int((agreement < 0.7).sum()),
        "low_confidence_pct": round(
            100 * float((agreement < 0.7).sum()) / len(y), 1
        ),
    }


def classify_enhanced(
    df: pd.DataFrame,
    classifier: str = "xgboost",
    n_folds: int = 5,
) -> dict:
    """Enhanced single-model classification with richer output."""
    features = engineer_enhanced_features(df)
    labels = df[FRACTURE_TYPE_COL].values
    le = LabelEncoder()
    y = le.fit_transform(labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    all_models = _get_models()
    if classifier not in all_models:
        classifier = "random_forest"  # fallback
    model = all_models[classifier]

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)

    model.fit(X, y)

    feat_imp = {}
    if hasattr(model, "feature_importances_"):
        feat_imp = dict(zip(features.columns, model.feature_importances_))
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_).mean(axis=0)
        feat_imp = dict(zip(features.columns, coef))

    conf_matrix = confusion_matrix(y, y_pred_cv)

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": features.columns.tolist(),
        "cv_scores": scores,
        "cv_mean_accuracy": float(scores.mean()),
        "cv_std_accuracy": float(scores.std()),
        "cv_f1_mean": float(f1_scores.mean()),
        "cv_f1_std": float(f1_scores.std()),
        "confusion_matrix": conf_matrix,
        "feature_importances": {
            k: round(float(v), 4) for k, v in feat_imp.items()
        },
        "class_names": le.classes_.tolist(),
    }


# ──────────────────────────────────────────────────────
# Enhanced Clustering with Multiple Methods
# ──────────────────────────────────────────────────────

def cluster_enhanced(
    df: pd.DataFrame,
    n_clusters: int = None,
    max_clusters: int = 8,
    methods: list = None,
) -> dict:
    """Compare multiple clustering methods on fracture orientation data.

    Methods: KMeans, Gaussian Mixture Model (GMM), DBSCAN.
    """
    if methods is None:
        methods = ["kmeans", "gmm"]

    features = engineer_enhanced_features(df)
    orient_cols = [c for c in features.columns
                   if c not in ("depth", "depth_normalized",
                                "pore_pressure_mpa", "overburden_mpa",
                                "temperature_c", "fracture_density",
                                "fracture_spacing", "log_spacing",
                                "fabric_e1", "fabric_e2", "fabric_e3",
                                "woodcock_K", "woodcock_C")]
    X = StandardScaler().fit_transform(features[orient_cols].values)

    results = {}

    for method in methods:
        if method == "kmeans":
            result = _cluster_kmeans(X, df, n_clusters, max_clusters)
        elif method == "gmm":
            result = _cluster_gmm(X, df, n_clusters, max_clusters)
        elif method == "dbscan":
            result = _cluster_dbscan(X, df)
        else:
            continue
        results[method] = result

    # Pick best method by silhouette score
    best_method = max(
        results.items(),
        key=lambda x: x[1].get("silhouette", -1)
    )

    return {
        "methods": results,
        "best_method": best_method[0],
        "best_result": best_method[1],
    }


def _cluster_kmeans(X, df, n_clusters, max_clusters):
    if n_clusters is None:
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
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    stats = _compute_cluster_stats(df, labels, n_clusters)

    return {
        "n_clusters": n_clusters,
        "labels": labels,
        "silhouette": float(sil),
        "calinski_harabasz": float(ch),
        "silhouette_scores": scores,
        "cluster_stats": stats,
    }


def _cluster_gmm(X, df, n_clusters, max_clusters):
    if n_clusters is None:
        best_k, best_bic = 2, np.inf
        for k in range(2, max_clusters + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_k, best_bic = k, bic
        n_clusters = best_k

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(X)
    probabilities = gmm.predict_proba(X)

    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0
    stats = _compute_cluster_stats(df, labels, n_clusters)

    # Cluster assignment confidence
    max_prob = probabilities.max(axis=1)

    return {
        "n_clusters": n_clusters,
        "labels": labels,
        "silhouette": float(sil),
        "cluster_stats": stats,
        "mean_confidence": float(max_prob.mean()),
        "low_confidence_count": int((max_prob < 0.7).sum()),
    }


def _cluster_dbscan(X, df):
    db = DBSCAN(eps=0.8, min_samples=5)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    sil = silhouette_score(X, labels) if n_clusters > 1 else 0
    stats = _compute_cluster_stats(df, labels, n_clusters)

    return {
        "n_clusters": n_clusters,
        "labels": labels,
        "silhouette": float(sil),
        "cluster_stats": stats,
        "noise_count": int(n_noise),
        "noise_pct": round(100 * n_noise / len(labels), 1),
    }


def _compute_cluster_stats(df, labels, n_clusters):
    stats = []
    for c in range(n_clusters):
        mask = labels == c
        if not mask.any():
            continue
        stats.append({
            "cluster": c,
            "count": int(mask.sum()),
            "mean_azimuth": round(float(df.loc[mask, AZIMUTH_COL].mean()), 1),
            "mean_dip": round(float(df.loc[mask, DIP_COL].mean()), 1),
            "std_azimuth": round(float(df.loc[mask, AZIMUTH_COL].std()), 1),
            "std_dip": round(float(df.loc[mask, DIP_COL].std()), 1),
        })
    return stats


# ──────────────────────────────────────────────────────
# Pore-Pressure-Corrected Critically Stressed Analysis
# ──────────────────────────────────────────────────────

def critically_stressed_enhanced(
    sigma_n: np.ndarray,
    tau: np.ndarray,
    mu: float = 0.6,
    cohesion: float = 0.0,
    pore_pressure: float = 0.0,
) -> dict:
    """Enhanced critically stressed analysis with pore pressure correction.

    Uses effective stress: σn_eff = σn - Pp
    Mohr-Coulomb: τ >= cohesion + μ * (σn - Pp)

    Returns detailed analysis with risk categories.
    """
    sigma_n_eff = sigma_n - pore_pressure
    tau_critical = cohesion + mu * sigma_n_eff
    excess_shear = tau - tau_critical  # positive = critically stressed

    is_critical = excess_shear >= 0
    slip_ratio = np.where(tau_critical > 0, tau / tau_critical, 0.0)

    # Risk categories
    HIGH_RISK = 1.0    # slip_ratio >= 1.0 (above failure line)
    MODERATE = 0.8     # 0.8 <= slip_ratio < 1.0
    LOW = 0.0          # slip_ratio < 0.8

    risk = np.full(len(sigma_n), "low", dtype=object)
    risk[slip_ratio >= MODERATE] = "moderate"
    risk[slip_ratio >= HIGH_RISK] = "high"

    return {
        "is_critical": is_critical,
        "count_critical": int(is_critical.sum()),
        "total": len(sigma_n),
        "pct_critical": round(100 * float(is_critical.sum()) / len(sigma_n), 1),
        "excess_shear": excess_shear,
        "slip_ratio": slip_ratio,
        "effective_normal_stress": sigma_n_eff,
        "risk_categories": risk,
        "high_risk_count": int((risk == "high").sum()),
        "moderate_risk_count": int((risk == "moderate").sum()),
        "low_risk_count": int((risk == "low").sum()),
        "pore_pressure_applied": pore_pressure,
        "mean_slip_ratio": float(slip_ratio.mean()),
        "max_slip_ratio": float(slip_ratio.max()),
    }


# ──────────────────────────────────────────────────────
# Stakeholder Decision Support
# ──────────────────────────────────────────────────────

def generate_interpretation(inversion_result: dict,
                            cs_result: dict,
                            well_name: str = "") -> dict:
    """Generate plain-language interpretation of results for stakeholders.

    Translates technical numbers into actionable insights that
    non-specialists can understand.
    """
    sigma1 = inversion_result["sigma1"]
    sigma3 = inversion_result["sigma3"]
    shmax = inversion_result["shmax_azimuth_deg"]
    R = inversion_result["R"]
    mu = inversion_result["mu"]
    regime = inversion_result["regime"]

    interpretations = []
    warnings_list = []
    recommendations = []

    # Stress regime interpretation
    regime_names = {
        "normal": "Normal faulting",
        "strike_slip": "Strike-slip faulting",
        "thrust": "Thrust/reverse faulting",
    }
    regime_desc = {
        "normal": "The vertical stress is the maximum principal stress. "
                  "This means the rock is being pulled apart horizontally. "
                  "Expect vertical or steeply-dipping fractures to dominate.",
        "strike_slip": "The vertical stress is intermediate. "
                       "Horizontal stresses control fracture behavior. "
                       "Expect subvertical fractures striking parallel to SHmax.",
        "thrust": "The vertical stress is the minimum principal stress. "
                  "Strong horizontal compression is present. "
                  "Expect shallow-dipping thrust faults and fractures.",
    }
    interpretations.append({
        "title": "Stress Regime",
        "value": regime_names.get(regime, regime),
        "explanation": regime_desc.get(regime, ""),
        "confidence": "Based on fracture orientation analysis",
    })

    # SHmax direction
    compass = _azimuth_to_compass(shmax)
    interpretations.append({
        "title": "Maximum Horizontal Stress Direction",
        "value": f"{shmax:.1f}° ({compass})",
        "explanation": f"The maximum horizontal stress points {compass} "
                       f"({shmax:.1f}° from North). Drilling horizontal wells "
                       f"perpendicular to this direction ({(shmax + 90) % 360:.0f}°) "
                       f"will maximize fracture intersection for better production. "
                       f"Wells drilled parallel to SHmax will have more stable boreholes.",
        "confidence": "Derived from fracture population inversion",
    })

    # Stress magnitudes
    diff_stress = sigma1 - sigma3
    interpretations.append({
        "title": "Stress Magnitudes",
        "value": f"σ1={sigma1:.1f}, σ2={inversion_result['sigma2']:.1f}, "
                 f"σ3={sigma3:.1f} MPa",
        "explanation": f"The differential stress (σ1-σ3) is {diff_stress:.1f} MPa. "
                       f"{'High' if diff_stress > 30 else 'Moderate' if diff_stress > 15 else 'Low'} "
                       f"differential stress means fractures are "
                       f"{'highly' if diff_stress > 30 else 'moderately' if diff_stress > 15 else 'less'} "
                       f"likely to be reactivated during operations.",
        "confidence": f"R ratio = {R:.3f}",
    })

    # Critically stressed assessment
    cs_pct = cs_result["pct_critical"]
    high_risk = cs_result["high_risk_count"]
    total = cs_result["total"]

    if cs_pct > 50:
        risk_level = "HIGH"
        risk_color = "danger"
        interpretations.append({
            "title": "Critically Stressed Fractures",
            "value": f"{cs_pct:.1f}% ({cs_result['count_critical']}/{total})",
            "explanation": f"Over half of the fractures are critically stressed. "
                           f"These fractures are at or above the Mohr-Coulomb "
                           f"failure line and are likely to slip or open under "
                           f"current stress conditions. They may act as fluid "
                           f"conduits, which is good for production but poses "
                           f"risk for wellbore stability.",
            "confidence": f"μ = {mu:.3f}, Pp = {cs_result['pore_pressure_applied']:.1f} MPa",
        })
        warnings_list.append(
            f"High proportion ({cs_pct:.0f}%) of critically stressed fractures. "
            f"Exercise caution with mud weight and completion design."
        )
    elif cs_pct > 25:
        risk_level = "MODERATE"
        risk_color = "warning"
        interpretations.append({
            "title": "Critically Stressed Fractures",
            "value": f"{cs_pct:.1f}% ({cs_result['count_critical']}/{total})",
            "explanation": f"About a quarter of fractures are critically stressed. "
                           f"Moderate risk of induced slip during drilling or "
                           f"stimulation operations.",
            "confidence": f"μ = {mu:.3f}, Pp = {cs_result['pore_pressure_applied']:.1f} MPa",
        })
    else:
        risk_level = "LOW"
        risk_color = "success"
        interpretations.append({
            "title": "Critically Stressed Fractures",
            "value": f"{cs_pct:.1f}% ({cs_result['count_critical']}/{total})",
            "explanation": f"Most fractures are stable under current stress. "
                           f"Low risk of induced seismicity or uncontrolled "
                           f"fluid loss through fracture reactivation.",
            "confidence": f"μ = {mu:.3f}, Pp = {cs_result['pore_pressure_applied']:.1f} MPa",
        })

    # Recommendations
    recommendations.append(
        f"Drill horizontal wells at {(shmax + 90) % 360:.0f}° azimuth "
        f"(perpendicular to SHmax) for maximum natural fracture intersection."
    )

    if cs_pct > 40:
        recommendations.append(
            "Use managed pressure drilling (MPD) to maintain wellbore "
            "stability in zones with critically stressed fractures."
        )
        recommendations.append(
            "Consider selective stimulation - critically stressed fractures "
            "may already provide natural permeability without hydraulic fracturing."
        )

    if mu < 0.5:
        warnings_list.append(
            f"Low friction coefficient (μ={mu:.3f}). Rock may have weak "
            f"fracture surfaces (clay-filled?). Higher slip risk."
        )

    if R > 0.8:
        recommendations.append(
            f"High R ratio ({R:.3f}) indicates σ2 ≈ σ1. The stress field "
            f"is nearly axially symmetric - fracture reactivation potential "
            f"is similar across many orientations."
        )

    return {
        "interpretations": interpretations,
        "warnings": warnings_list,
        "recommendations": recommendations,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "well_name": well_name,
    }


def _azimuth_to_compass(az: float) -> str:
    """Convert azimuth to compass direction."""
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    idx = round(az / 22.5) % 16
    return directions[idx]


# ──────────────────────────────────────────────────────
# SHAP Explainability
# ──────────────────────────────────────────────────────

def compute_shap_explanations(
    df: pd.DataFrame,
    classifier: str = "gradient_boosting",
    max_display: int = 10,
) -> dict:
    """Compute SHAP feature importance explanations for stakeholders.

    Uses TreeExplainer for tree-based models (fast, exact) and falls back
    to permutation-based importance when SHAP is unavailable.

    Returns:
        - global_importance: feature-level mean |SHAP| values
        - top_features: ranked list with plain-language explanations
        - sample_explanations: per-sample SHAP values for top samples
        - feature_interactions: top feature interaction pairs
    """
    features = engineer_enhanced_features(df)
    labels = df[FRACTURE_TYPE_COL].values
    le = LabelEncoder()
    y = le.fit_transform(labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    feature_names = features.columns.tolist()

    # Pick a tree-based model supported by SHAP TreeExplainer.
    # Note: GradientBoostingClassifier only supports binary in SHAP,
    # so we prefer RF / XGBoost / LightGBM for multiclass SHAP.
    shap_preferred = ["xgboost", "lightgbm", "random_forest"]
    all_models = _get_models()

    if classifier == "gradient_boosting" and len(np.unique(y)) > 2:
        # GBM not supported for multiclass SHAP - pick best alternative
        for alt in shap_preferred:
            if alt in all_models:
                classifier = alt
                break

    if classifier not in all_models:
        classifier = list(all_models.keys())[0]
    model = all_models[classifier]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    class_names = le.classes_.tolist()

    # Plain-language feature descriptions for stakeholders
    feature_descriptions = {
        "nx": "Fracture orientation (East-West component)",
        "ny": "Fracture orientation (North-South component)",
        "nz": "Fracture steepness (vertical component)",
        "az_sin": "Fracture direction (sine encoding)",
        "az_cos": "Fracture direction (cosine encoding)",
        "az2_sin": "Fracture direction periodicity (2x sine)",
        "az2_cos": "Fracture direction periodicity (2x cosine)",
        "dip": "Fracture dip angle from horizontal",
        "depth": "Depth below surface",
        "pore_pressure_mpa": "Pore fluid pressure at depth",
        "overburden_mpa": "Weight of overlying rock",
        "temperature_c": "Formation temperature at depth",
        "fracture_density": "Number of nearby fractures",
        "fracture_spacing": "Distance to nearest fracture",
        "log_spacing": "Log-transformed fracture spacing",
        "depth_normalized": "Relative position within well",
        "fabric_e1": "Primary fabric orientation strength",
        "fabric_e2": "Secondary fabric orientation strength",
        "fabric_e3": "Tertiary fabric orientation strength",
        "woodcock_K": "Fabric shape (clustered vs girdle)",
        "woodcock_C": "Overall fabric strength",
    }

    result = {
        "classifier": classifier,
        "class_names": class_names,
        "n_samples": len(y),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "has_shap": False,
    }

    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # shap_values shape varies by model/SHAP version:
            #   - list of (n_samples, n_features) arrays, one per class
            #   - 3D array (n_samples, n_features, n_classes) from XGBoost
            #   - 2D array (n_samples, n_features) for binary
            if isinstance(shap_values, list):
                abs_shap = np.mean(
                    [np.abs(sv) for sv in shap_values], axis=0
                )
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # (n_samples, n_features, n_classes) -> mean abs across classes
                abs_shap = np.abs(shap_values).mean(axis=2)
            else:
                abs_shap = np.abs(shap_values)

            # Global importance: mean |SHAP| per feature
            global_importance = abs_shap.mean(axis=0)
            sorted_idx = np.argsort(global_importance)[::-1]

            # Build ranked feature list with descriptions
            top_features = []
            for rank, idx in enumerate(sorted_idx[:max_display]):
                fname = feature_names[idx]
                top_features.append({
                    "rank": rank + 1,
                    "feature": fname,
                    "importance": round(float(global_importance[idx]), 4),
                    "description": feature_descriptions.get(
                        fname, f"Feature: {fname}"
                    ),
                })

            # Per-class importance (which features matter for each class)
            class_importance = {}
            if isinstance(shap_values, list):
                for ci, cls_name in enumerate(class_names):
                    cls_abs = np.abs(shap_values[ci]).mean(axis=0)
                    cls_sorted = np.argsort(cls_abs)[::-1][:5]
                    class_importance[cls_name] = [
                        {
                            "feature": feature_names[i],
                            "importance": round(float(cls_abs[i]), 4),
                        }
                        for i in cls_sorted
                    ]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # (n_samples, n_features, n_classes)
                for ci, cls_name in enumerate(class_names):
                    cls_abs = np.abs(shap_values[:, :, ci]).mean(axis=0)
                    cls_sorted = np.argsort(cls_abs)[::-1][:5]
                    class_importance[cls_name] = [
                        {
                            "feature": feature_names[i],
                            "importance": round(float(cls_abs[i]), 4),
                        }
                        for i in cls_sorted
                    ]

            # Sample explanations for most uncertain samples
            # (highest max SHAP spread = most interesting to review)
            sample_shap_spread = abs_shap.max(axis=1)
            top_sample_idx = np.argsort(sample_shap_spread)[::-1][:5]

            sample_explanations = []
            for si in top_sample_idx:
                sample_feats = []
                if isinstance(shap_values, list):
                    sample_abs = np.mean(
                        [np.abs(sv[si]) for sv in shap_values], axis=0
                    )
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    sample_abs = np.abs(shap_values[si]).mean(axis=1)
                else:
                    sample_abs = abs_shap[si]
                top_feat_idx = np.argsort(sample_abs)[::-1][:3]
                for fi in top_feat_idx:
                    sample_feats.append({
                        "feature": feature_names[fi],
                        "shap_value": round(float(sample_abs[fi]), 4),
                    })
                sample_explanations.append({
                    "sample_index": int(si),
                    "predicted_class": class_names[int(model.predict(X[si:si+1])[0])],
                    "top_drivers": sample_feats,
                })

            result.update({
                "has_shap": True,
                "global_importance": {
                    feature_names[i]: round(float(global_importance[i]), 4)
                    for i in sorted_idx
                },
                "top_features": top_features,
                "class_importance": class_importance,
                "sample_explanations": sample_explanations,
            })
        except Exception:
            # Fall through to permutation-based fallback
            pass

    # Fallback: use model's built-in feature importance
    if not result["has_shap"]:
        feat_imp = {}
        if hasattr(model, "feature_importances_"):
            feat_imp = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_).mean(axis=0)
            feat_imp = dict(zip(feature_names, coef))

        sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        top_features = [
            {
                "rank": i + 1,
                "feature": fname,
                "importance": round(float(imp), 4),
                "description": feature_descriptions.get(fname, f"Feature: {fname}"),
            }
            for i, (fname, imp) in enumerate(sorted_feats[:max_display])
        ]
        result.update({
            "global_importance": {k: round(float(v), 4) for k, v in feat_imp.items()},
            "top_features": top_features,
            "class_importance": {},
            "sample_explanations": [],
            "fallback_method": "model_native",
        })

    return result


# ──────────────────────────────────────────────────────
# Feedback & Validation Tracking
# ──────────────────────────────────────────────────────

class FeedbackStore:
    """In-memory store for expert feedback on analysis results.

    In production, this would be backed by a database.
    Tracks:
    - Expert ratings on result accuracy
    - Flagged fractures that need review
    - Suggested corrections
    - Data quality scores
    """

    def __init__(self):
        self.feedback_log = []
        self.flagged_fractures = []
        self.data_quality_scores = []

    def add_feedback(self, well: str, analysis_type: str,
                     rating: int, comment: str = "",
                     expert_name: str = "anonymous") -> dict:
        """Record expert feedback on an analysis result.

        rating: 1 (very inaccurate) to 5 (very accurate)
        """
        entry = {
            "well": well,
            "analysis_type": analysis_type,
            "rating": max(1, min(5, rating)),
            "comment": comment,
            "expert_name": expert_name,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        self.feedback_log.append(entry)
        return entry

    def flag_fracture(self, well: str, fracture_idx: int,
                      reason: str, suggested_type: str = "") -> dict:
        """Flag a specific fracture for expert review."""
        entry = {
            "well": well,
            "fracture_idx": fracture_idx,
            "reason": reason,
            "suggested_type": suggested_type,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        self.flagged_fractures.append(entry)
        return entry

    def get_summary(self) -> dict:
        """Return summary of all feedback collected."""
        if not self.feedback_log:
            return {
                "total_feedback": 0,
                "avg_rating": None,
                "feedback_by_type": {},
            }

        ratings = [f["rating"] for f in self.feedback_log]
        by_type = {}
        for f in self.feedback_log:
            t = f["analysis_type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(f["rating"])

        return {
            "total_feedback": len(self.feedback_log),
            "avg_rating": round(sum(ratings) / len(ratings), 2),
            "feedback_by_type": {
                k: {"count": len(v), "avg_rating": round(sum(v) / len(v), 2)}
                for k, v in by_type.items()
            },
            "flagged_fractures": len(self.flagged_fractures),
            "recent_feedback": self.feedback_log[-5:],
        }


# Global feedback store
feedback_store = FeedbackStore()
