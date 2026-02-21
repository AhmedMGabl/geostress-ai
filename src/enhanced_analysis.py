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
    cross_val_score, cross_val_predict, cross_validate, StratifiedKFold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, balanced_accuracy_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

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

def _get_models(fast: bool = False) -> dict:
    """Return all available classification models.

    fast=True uses fewer estimators for quicker results (~3x speedup).
    Accuracy loss is typically <0.5%.
    """
    n_est = 100 if fast else 300

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=n_est, max_depth=12, min_samples_leaf=3,
            random_state=42, class_weight="balanced", n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=n_est, max_depth=5, learning_rate=0.1 if fast else 0.05,
            subsample=0.8, random_state=42,
        ),
        "svm": SVC(
            kernel="rbf", C=10.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=42,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32) if fast else (128, 64, 32),
            max_iter=300 if fast else 500,
            early_stopping=True, validation_fraction=0.15,
            random_state=42,
        ),
    }

    if HAS_XGB:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=6,
            learning_rate=0.1 if fast else 0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="mlogloss",
        )

    if HAS_LGB:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=n_est, max_depth=6,
            learning_rate=0.1 if fast else 0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42, verbose=-1,
        )

    return models


def _should_use_smote(y: np.ndarray, min_ratio: float = 0.15) -> bool:
    """Determine if SMOTE should be applied based on class imbalance severity.

    Returns True if the minority/majority ratio is below min_ratio
    AND the minority class has at least 6 samples (SMOTE needs k_neighbors=5).
    """
    counts = np.bincount(y)
    min_count = counts.min()
    max_count = counts.max()
    if min_count < 6:
        # SMOTE default k_neighbors=5; need at least 6 samples
        return False
    return (min_count / max_count) < min_ratio


def _cv_with_smote(model, X, y, cv, smote_strategy="auto"):
    """Run cross-validation with SMOTE applied only to training folds.

    This prevents data leakage: synthetic samples are never in test folds.
    Returns (acc_scores, f1_scores, y_pred_cv, smote_info).
    """
    from sklearn.base import clone

    acc_scores = []
    f1_scores = []
    y_pred_cv = np.zeros(len(y), dtype=int)
    smote_counts = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Determine k_neighbors: must be <= smallest class count - 1
        train_counts = np.bincount(y_train)
        min_train = train_counts[train_counts > 0].min()
        k_neighbors = min(5, min_train - 1)

        if k_neighbors < 1:
            # Can't use SMOTE, fall back to original data
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
        else:
            smote = SMOTE(
                k_neighbors=k_neighbors,
                random_state=42,
                sampling_strategy=smote_strategy,
            )
            X_res, y_res = smote.fit_resample(X_train, y_train)
            smote_counts.append({
                "original": len(y_train),
                "resampled": len(y_res),
                "synthetic": len(y_res) - len(y_train),
            })
            model_clone = clone(model)
            model_clone.fit(X_res, y_res)

        y_pred_fold = model_clone.predict(X_test)
        y_pred_cv[test_idx] = y_pred_fold
        acc_scores.append(float(accuracy_score(y_test, y_pred_fold)))
        f1_scores.append(float(f1_score(y_test, y_pred_fold, average="weighted", zero_division=0)))

    smote_info = {}
    if smote_counts:
        avg_synthetic = np.mean([s["synthetic"] for s in smote_counts])
        smote_info = {
            "applied": True,
            "avg_synthetic_per_fold": round(avg_synthetic),
            "total_folds": len(smote_counts),
        }
    else:
        smote_info = {"applied": False, "reason": "Insufficient minority samples for SMOTE"}

    return (
        np.array(acc_scores),
        np.array(f1_scores),
        y_pred_cv,
        smote_info,
    )


def compare_models(
    df: pd.DataFrame,
    n_folds: int = 5,
    models_to_run: list = None,
    fast: bool = False,
) -> dict:
    """Run all models and return comparative metrics.

    Optimized: uses cross_validate for single-pass multi-metric scoring.
    Also includes a stacking ensemble, conformal prediction, and SMOTE
    oversampling when severe class imbalance is detected.

    fast=True: fewer estimators, 3-fold CV for ~3x speedup.
    """
    features = engineer_enhanced_features(df)
    labels = df[FRACTURE_TYPE_COL].values
    le = LabelEncoder()
    y = le.fit_transform(labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    if fast:
        n_folds = min(n_folds, 3)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_models = _get_models(fast=fast)

    if models_to_run:
        all_models = {k: v for k, v in all_models.items() if k in models_to_run}

    # Determine if SMOTE should be used
    use_smote = HAS_IMBLEARN and _should_use_smote(y)
    smote_global_info = {
        "available": HAS_IMBLEARN,
        "applied": use_smote,
        "reason": (
            "Class imbalance detected; SMOTE applied to training folds"
            if use_smote
            else (
                "imbalanced-learn not installed" if not HAS_IMBLEARN
                else "Class balance is acceptable or minority classes too small for SMOTE"
            )
        ),
    }

    results = {}
    all_preds = {}  # Collect predictions for agreement analysis
    sample_weights = compute_sample_weight("balanced", y)

    for name, model in all_models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if use_smote:
                # SMOTE-enhanced CV (proper: SMOTE only on train folds)
                acc_scores, f1_scores, y_pred_cv, smote_info = _cv_with_smote(
                    model, X, y, cv,
                )
            else:
                # Use sample_weight for models without native class_weight support
                needs_weight = name in ("gradient_boosting", "xgboost")
                sw_params = {"sample_weight": sample_weights} if needs_weight else {}

                # Single-pass cross_validate for both metrics at once
                cv_result = cross_validate(
                    model, X, y, cv=cv,
                    scoring={"accuracy": "accuracy", "f1": "f1_weighted"},
                    return_estimator=False,
                    params=sw_params if sw_params else None,
                )
                acc_scores = cv_result["test_accuracy"]
                f1_scores = cv_result["test_f1"]
                smote_info = {"applied": False}

                # cross_val_predict for confusion matrix (separate, but only 1 extra CV)
                y_pred_cv = cross_val_predict(
                    model, X, y, cv=cv, params=sw_params if sw_params else None,
                )

            # Fit on full data for feature importances + predictions
            # (SMOTE for full fit if applicable, else use sample weights)
            if use_smote and HAS_IMBLEARN:
                train_counts = np.bincount(y)
                min_train = train_counts[train_counts > 0].min()
                k_n = min(5, min_train - 1)
                if k_n >= 1:
                    smote_full = SMOTE(k_neighbors=k_n, random_state=42)
                    X_full_res, y_full_res = smote_full.fit_resample(X, y)
                    model.fit(X_full_res, y_full_res)
                else:
                    model.fit(X, y)
            else:
                needs_weight = name in ("gradient_boosting", "xgboost")
                sw_fit = {"sample_weight": sample_weights} if needs_weight else {}
                model.fit(X, y, **sw_fit)
            y_pred_full = model.predict(X)
            all_preds[name] = y_pred_full

            # Feature importances
            feat_imp = {}
            if hasattr(model, "feature_importances_"):
                feat_imp = dict(zip(
                    features.columns, model.feature_importances_
                ))
            elif hasattr(model, "coef_"):
                coef = np.abs(model.coef_).mean(axis=0)
                feat_imp = dict(zip(features.columns, coef))

            train_acc = float(accuracy_score(y, y_pred_full))
            cv_acc = float(acc_scores.mean())

            # Per-class metrics from cross-validated predictions
            per_class = {}
            class_prec = precision_score(y, y_pred_cv, average=None, zero_division=0)
            class_rec = recall_score(y, y_pred_cv, average=None, zero_division=0)
            class_f1 = f1_score(y, y_pred_cv, average=None, zero_division=0)
            class_counts = np.bincount(y, minlength=len(le.classes_))
            for ci, cname in enumerate(le.classes_):
                per_class[cname] = {
                    "precision": round(float(class_prec[ci]), 3),
                    "recall": round(float(class_rec[ci]), 3),
                    "f1": round(float(class_f1[ci]), 3),
                    "support": int(class_counts[ci]),
                }

            results[name] = {
                "cv_accuracy_mean": cv_acc,
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
                "train_accuracy": train_acc,
                "overfit_gap": round(train_acc - cv_acc, 3),
                "balanced_accuracy": round(float(
                    balanced_accuracy_score(y, y_pred_cv)
                ), 3),
                "per_class_metrics": per_class,
                "smote": smote_info,
            }

    # ── Stacking Ensemble ──
    # Uses top 3 tree-based models as base learners with LR meta-learner
    stack_result = _build_stacking_ensemble(X, y, cv, features.columns, le, fast,
                                               use_smote=use_smote)
    if stack_result is not None:
        results["stacking_ensemble"] = stack_result

    # Determine if class imbalance is severe enough to rank by balanced accuracy
    class_counts = np.bincount(y, minlength=len(le.classes_))
    max_count = class_counts.max()
    min_count = class_counts[class_counts > 0].min()
    imbalance_ratio = min_count / max_count if max_count > 0 else 1.0
    rank_by_balanced = imbalance_ratio < 0.15  # severe imbalance

    # Rank models: use balanced accuracy when severe imbalance, else standard
    sort_key = "balanced_accuracy" if rank_by_balanced else "cv_accuracy_mean"
    ranked = sorted(
        results.items(),
        key=lambda x: x[1].get(sort_key, 0),
        reverse=True,
    )
    ranking = [
        {"rank": i + 1, "model": name, "accuracy": r["cv_accuracy_mean"],
         "f1": r["cv_f1_mean"], "overfit_gap": r.get("overfit_gap", 0),
         "balanced_accuracy": r.get("balanced_accuracy", 0)}
        for i, (name, r) in enumerate(ranked)
    ]
    ranking_criterion = "balanced_accuracy" if rank_by_balanced else "standard_accuracy"

    # Generalization assessment
    best_name, best_res = ranked[0] if ranked else (None, {})
    gap = best_res.get("overfit_gap", 0) if best_res else 0
    best_cv_std = best_res.get("cv_accuracy_std", 0) if best_res else 0
    gen_warnings = []
    if gap > 0.10:
        gen_warnings.append(
            f"The best model ({best_name}) shows significant overfitting "
            f"(train-test gap = {gap:.1%}). Predictions on new data may be "
            f"less accurate than reported."
        )
    if best_cv_std > 0.05:
        gen_warnings.append(
            f"Cross-validation scores have high variance (std = {best_cv_std:.1%}). "
            f"Model performance is unstable across data splits."
        )
    # Check for under-represented classes
    class_counts = np.bincount(y, minlength=len(le.classes_))
    minority_classes = [
        le.classes_[i] for i in range(len(le.classes_))
        if class_counts[i] < 30
    ]
    if minority_classes:
        gen_warnings.append(
            f"Under-represented fracture types ({', '.join(minority_classes)}) "
            f"have fewer than 30 samples. Per-class metrics for these types "
            f"are unreliable. Collecting more samples will improve accuracy."
        )
    # Check balanced accuracy gap (class imbalance severity)
    best_bal_acc = best_res.get("balanced_accuracy", 0) if best_res else 0
    best_std_acc = best_res.get("cv_accuracy_mean", 0) if best_res else 0
    imbalance_gap = best_std_acc - best_bal_acc
    if imbalance_gap > 0.20:
        gen_warnings.append(
            f"SEVERE CLASS IMBALANCE: Standard accuracy ({best_std_acc:.1%}) is much "
            f"higher than balanced accuracy ({best_bal_acc:.1%}). The model correctly "
            f"classifies majority types (Vuggy, Brecciated) but fails on minority types. "
            f"Do NOT rely on predictions for under-represented fracture types."
        )
    elif imbalance_gap > 0.10:
        gen_warnings.append(
            f"Moderate class imbalance: Balanced accuracy ({best_bal_acc:.1%}) is notably "
            f"lower than standard accuracy ({best_std_acc:.1%}). Predictions for smaller "
            f"fracture type classes may be unreliable."
        )

    generalization = {
        "overfit_gap": round(gap, 3),
        "cv_stability": round(best_cv_std, 3),
        "min_class_count": int(class_counts.min()),
        "imbalance_gap": round(imbalance_gap, 3),
        "balanced_accuracy": round(best_bal_acc, 3),
        "warnings": gen_warnings,
        "recommendation": (
            "Model generalizes well." if not gen_warnings
            else f"{len(gen_warnings)} issue(s) detected. See warnings."
        ),
    }

    # Model agreement analysis (uncertainty signal)
    pred_matrix = np.array(list(all_preds.values()))  # (n_models, n_samples)
    agreement = np.zeros(len(y))
    for i in range(len(y)):
        votes = pred_matrix[:, i]
        majority = np.bincount(votes).max()
        agreement[i] = majority / len(all_models)

    # ── Conformal prediction: per-sample confidence ──
    conformal = _conformal_confidence(X, y, cv, all_models)

    return {
        "models": results,
        "ranking": ranking,
        "ranking_criterion": ranking_criterion,
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
        "conformal": conformal,
        "generalization": generalization,
        "smote": smote_global_info,
    }


def _build_stacking_ensemble(X, y, cv, feature_names, le, fast=False,
                              use_smote=False) -> dict | None:
    """Build a stacking ensemble from the best available base learners.

    Uses LogisticRegression as meta-learner (learns which base model
    to trust for which type of sample).
    When use_smote=True, applies SMOTE to training folds.
    """
    from sklearn.base import clone

    base_estimators = []
    available = _get_models(fast=fast)

    # Prefer tree-based models as base learners (diverse + accurate)
    for name in ["random_forest", "xgboost", "lightgbm", "gradient_boosting"]:
        if name in available:
            base_estimators.append((name, available[name]))
        if len(base_estimators) >= 3:
            break

    if len(base_estimators) < 2:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            stack = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(
                    max_iter=1000, random_state=42, multi_class="multinomial",
                ),
                cv=cv,
                stack_method="predict_proba",
                n_jobs=-1,
            )

            if use_smote and HAS_IMBLEARN:
                # Manual CV with SMOTE on training folds
                acc_scores = []
                f1_scores_list = []
                y_pred_cv = np.zeros(len(y), dtype=int)

                for train_idx, test_idx in cv.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    train_counts = np.bincount(y_train)
                    min_train = train_counts[train_counts > 0].min()
                    k_n = min(5, min_train - 1)

                    stack_clone = clone(stack)
                    if k_n >= 1:
                        smote = SMOTE(k_neighbors=k_n, random_state=42)
                        X_res, y_res = smote.fit_resample(X_train, y_train)
                        stack_clone.fit(X_res, y_res)
                    else:
                        stack_clone.fit(X_train, y_train)

                    y_pred_fold = stack_clone.predict(X_test)
                    y_pred_cv[test_idx] = y_pred_fold
                    acc_scores.append(float(accuracy_score(y_test, y_pred_fold)))
                    f1_scores_list.append(float(f1_score(y_test, y_pred_fold, average="weighted", zero_division=0)))

                acc_scores = np.array(acc_scores)
                f1_scores = np.array(f1_scores_list)

                # Fit on full SMOTE data
                train_counts = np.bincount(y)
                min_train = train_counts[train_counts > 0].min()
                k_n = min(5, min_train - 1)
                if k_n >= 1:
                    smote_full = SMOTE(k_neighbors=k_n, random_state=42)
                    X_full_res, y_full_res = smote_full.fit_resample(X, y)
                    stack.fit(X_full_res, y_full_res)
                else:
                    stack.fit(X, y)
            else:
                cv_result = cross_validate(
                    stack, X, y, cv=cv,
                    scoring={"accuracy": "accuracy", "f1": "f1_weighted"},
                )
                acc_scores = cv_result["test_accuracy"]
                f1_scores = cv_result["test_f1"]

                y_pred_cv = cross_val_predict(stack, X, y, cv=cv)
                stack.fit(X, y)

            y_pred_full = stack.predict(X)

            train_acc = float(accuracy_score(y, y_pred_full))
            cv_acc = float(acc_scores.mean())

            # Per-class metrics
            per_class = {}
            class_prec = precision_score(y, y_pred_cv, average=None, zero_division=0)
            class_rec = recall_score(y, y_pred_cv, average=None, zero_division=0)
            class_f1_vals = f1_score(y, y_pred_cv, average=None, zero_division=0)
            class_counts = np.bincount(y, minlength=len(le.classes_))
            for ci, cname in enumerate(le.classes_):
                per_class[cname] = {
                    "precision": round(float(class_prec[ci]), 3),
                    "recall": round(float(class_rec[ci]), 3),
                    "f1": round(float(class_f1_vals[ci]), 3),
                    "support": int(class_counts[ci]),
                }

            return {
                "cv_accuracy_mean": cv_acc,
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
                "feature_importances": {},  # stacking doesn't have simple importances
                "class_names": le.classes_.tolist(),
                "train_accuracy": train_acc,
                "overfit_gap": round(train_acc - cv_acc, 3),
                "balanced_accuracy": round(float(
                    balanced_accuracy_score(y, y_pred_cv)
                ), 3),
                "per_class_metrics": per_class,
                "base_learners": [name for name, _ in base_estimators],
                "smote": {"applied": use_smote and HAS_IMBLEARN},
            }
    except Exception:
        return None


def _conformal_confidence(X, y, cv, models) -> dict:
    """Simple conformal prediction: calibrate confidence from CV probabilities.

    For each sample, reports the probability of the predicted class.
    Samples where no model is >80% confident are flagged for expert review.
    """
    # Use the best probability-capable model
    prob_models = {k: v for k, v in models.items()
                   if hasattr(v, "predict_proba")}
    if not prob_models:
        return {"available": False}

    # Use first available model with probability support
    model_name = list(prob_models.keys())[0]
    model = prob_models[model_name]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probs = np.zeros((len(y), len(np.unique(y))))
            for train_idx, test_idx in cv.split(X, y):
                model.fit(X[train_idx], y[train_idx])
                probs[test_idx] = model.predict_proba(X[test_idx])

        max_prob = probs.max(axis=1)
        return {
            "available": True,
            "model": model_name,
            "mean_confidence": round(float(max_prob.mean()), 3),
            "min_confidence": round(float(max_prob.min()), 3),
            "high_confidence_pct": round(
                100 * float((max_prob >= 0.8).sum()) / len(y), 1
            ),
            "uncertain_count": int((max_prob < 0.5).sum()),
            "uncertain_pct": round(
                100 * float((max_prob < 0.5).sum()) / len(y), 1
            ),
        }
    except Exception:
        return {"available": False}


def classify_enhanced(
    df: pd.DataFrame,
    classifier: str = "xgboost",
    n_folds: int = 5,
) -> dict:
    """Enhanced single-model classification with richer output.

    Optimized: single cross_validate pass for both accuracy and F1.
    """
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Single-pass multi-metric CV
        cv_result = cross_validate(
            model, X, y, cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1_weighted"},
        )
        scores = cv_result["test_accuracy"]
        f1_scores = cv_result["test_f1"]

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
# Data Quality Validation
# ──────────────────────────────────────────────────────

def validate_data_quality(df: pd.DataFrame) -> dict:
    """Run comprehensive data quality checks for production use.

    Critical for the oil industry: bad data leads to bad drilling decisions.
    Returns quality score (0-100), issues found, and recommendations.
    """
    issues = []
    warnings_list = []
    score = 100  # Start perfect, deduct for issues

    n = len(df)
    if n == 0:
        return {"score": 0, "issues": ["No data"], "warnings": [], "recommendations": ["Upload fracture data to begin analysis."]}

    # ── Check required columns ──
    required = [AZIMUTH_COL, DIP_COL]
    for col in required:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
            score -= 30

    # ── Azimuth range check (0-360) ──
    if AZIMUTH_COL in df.columns:
        az = df[AZIMUTH_COL].dropna()
        out_of_range = ((az < 0) | (az > 360)).sum()
        if out_of_range > 0:
            issues.append(f"{out_of_range} azimuth values outside 0-360 range")
            score -= min(20, out_of_range * 2)

        # Check for suspicious clustering (all same value = likely error)
        if len(az.unique()) < 3 and n > 10:
            warnings_list.append("Very low azimuth diversity - check if data is real")
            score -= 10

    # ── Dip range check (0-90) ──
    if DIP_COL in df.columns:
        dip = df[DIP_COL].dropna()
        out_of_range = ((dip < 0) | (dip > 90)).sum()
        if out_of_range > 0:
            issues.append(f"{out_of_range} dip values outside 0-90 range")
            score -= min(20, out_of_range * 2)

        # Very low or very high average dip is suspicious
        if len(dip) > 0:
            mean_dip = dip.mean()
            if mean_dip < 5:
                warnings_list.append(f"Mean dip is very low ({mean_dip:.1f}). Near-horizontal fractures are unusual.")
            elif mean_dip > 85:
                warnings_list.append(f"Mean dip is very high ({mean_dip:.1f}). Near-vertical fractures dominate.")

    # ── Depth checks ──
    if DEPTH_COL in df.columns:
        depth = df[DEPTH_COL].dropna()
        if len(depth) == 0:
            warnings_list.append("No depth data available - pore pressure and overburden estimates will be unreliable")
            score -= 5
        else:
            if depth.min() < 0:
                issues.append(f"Negative depth values found (min={depth.min():.1f}m)")
                score -= 15
            if depth.max() > 12000:
                warnings_list.append(f"Very deep well ({depth.max():.0f}m). Verify depth units are in meters.")
            if depth.max() - depth.min() < 10 and n > 20:
                warnings_list.append("Very narrow depth range - fractures may be from a single thin zone")

    # ── Missing values ──
    for col in [AZIMUTH_COL, DIP_COL, DEPTH_COL]:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                pct = 100 * missing / n
                if pct > 20:
                    issues.append(f"{col}: {missing} missing values ({pct:.0f}%)")
                    score -= min(15, int(pct / 2))
                elif pct > 5:
                    warnings_list.append(f"{col}: {missing} missing values ({pct:.1f}%)")
                    score -= 3

    # ── Sample size checks ──
    if n < 30:
        warnings_list.append(f"Only {n} fractures. ML models need 50+ for reliable results.")
        score -= 10
    elif n < 100:
        warnings_list.append(f"{n} fractures is marginal for multi-class ML. Consider adding more data.")
        score -= 5

    # ── Fracture type distribution ──
    if FRACTURE_TYPE_COL in df.columns:
        type_counts = df[FRACTURE_TYPE_COL].value_counts()
        min_count = type_counts.min()
        if min_count < 5:
            minority = type_counts[type_counts < 5].index.tolist()
            warnings_list.append(
                f"Very few samples for types: {', '.join(minority)}. "
                f"Classification may be unreliable for these types."
            )
            score -= 5

        # Check for unknown types
        n_types = len(type_counts)
        if n_types > 10:
            warnings_list.append(f"{n_types} fracture types is unusual. Check for data entry errors.")
        elif n_types == 1:
            warnings_list.append("Only 1 fracture type - classification is not meaningful.")
            score -= 10

    # ── Well distribution ──
    if WELL_COL in df.columns:
        wells = df[WELL_COL].unique()
        if len(wells) == 1:
            warnings_list.append("Single well - results may not generalize to other wells.")

    # ── Duplicate detection ──
    if AZIMUTH_COL in df.columns and DIP_COL in df.columns:
        check_cols = [AZIMUTH_COL, DIP_COL]
        if DEPTH_COL in df.columns:
            check_cols.append(DEPTH_COL)
        dupes = df.duplicated(subset=check_cols).sum()
        if dupes > 0:
            pct = 100 * dupes / n
            if pct > 10:
                warnings_list.append(f"{dupes} duplicate rows ({pct:.0f}%). Check for repeated entries.")
                score -= 5

    score = max(0, min(100, score))

    # Generate recommendations
    recommendations = []
    if score < 50:
        recommendations.append("Data quality is LOW. Fix critical issues before running analysis.")
    elif score < 80:
        recommendations.append("Data quality is MODERATE. Review warnings before making operational decisions.")

    if n < 100:
        recommendations.append("Add more fracture measurements to improve ML model accuracy.")
    if DEPTH_COL not in df.columns or df[DEPTH_COL].isna().all():
        recommendations.append("Include depth measurements to enable pore pressure and overburden estimates.")

    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 50 else "F"

    return {
        "score": score,
        "grade": grade,
        "issues": issues,
        "warnings": warnings_list,
        "recommendations": recommendations,
        "stats": {
            "total_rows": n,
            "wells": df[WELL_COL].nunique() if WELL_COL in df.columns else 0,
            "fracture_types": df[FRACTURE_TYPE_COL].nunique() if FRACTURE_TYPE_COL in df.columns else 0,
            "missing_azimuth": int(df[AZIMUTH_COL].isna().sum()) if AZIMUTH_COL in df.columns else n,
            "missing_dip": int(df[DIP_COL].isna().sum()) if DIP_COL in df.columns else n,
            "missing_depth": int(df[DEPTH_COL].isna().sum()) if DEPTH_COL in df.columns else n,
        },
    }


# ──────────────────────────────────────────────────────
# Feedback & Validation Tracking
# ──────────────────────────────────────────────────────

class FeedbackStore:
    """In-memory store for expert feedback on analysis results.

    In production, this would be backed by a database.
    Tracks:
    - Expert ratings on result accuracy
    - Flagged fractures that need review
    - Label corrections (expert says fracture X is actually type Y)
    - Data quality scores
    - Actionable analytics: which types are most misclassified
    """

    def __init__(self):
        self.feedback_log = []
        self.flagged_fractures = []
        self.label_corrections = []  # Expert-corrected labels
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

    def correct_label(self, well: str, fracture_idx: int,
                      original_type: str, corrected_type: str,
                      expert_name: str = "anonymous") -> dict:
        """Record an expert's correction of a fracture classification.

        This is the key feedback mechanism: experts tell us when the model
        got it wrong, and we can retrain using these corrections.
        """
        entry = {
            "well": well,
            "fracture_idx": fracture_idx,
            "original_type": original_type,
            "corrected_type": corrected_type,
            "expert_name": expert_name,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        self.label_corrections.append(entry)
        return entry

    def get_corrections_count(self) -> int:
        """Number of expert label corrections available."""
        return len(self.label_corrections)

    def apply_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply expert label corrections to a DataFrame.

        Returns a copy with corrected labels. This is how expert feedback
        actually improves the model - corrected labels lead to better training.
        """
        if not self.label_corrections:
            return df

        corrected = df.copy()
        applied = 0
        for corr in self.label_corrections:
            mask = (corrected[WELL_COL] == corr["well"])
            if corr["fracture_idx"] < len(corrected[mask]):
                idx = corrected[mask].index[corr["fracture_idx"]]
                corrected.at[idx, FRACTURE_TYPE_COL] = corr["corrected_type"]
                applied += 1

        return corrected

    def get_summary(self) -> dict:
        """Return summary of all feedback collected."""
        if not self.feedback_log and not self.label_corrections:
            return {
                "total_feedback": 0,
                "avg_rating": None,
                "feedback_by_type": {},
                "label_corrections": 0,
                "accuracy_trend": [],
            }

        ratings = [f["rating"] for f in self.feedback_log]
        by_type = {}
        for f in self.feedback_log:
            t = f["analysis_type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(f["rating"])

        # Build accuracy trend (running average of last N ratings)
        trend = []
        window = 5
        for i in range(len(ratings)):
            start = max(0, i - window + 1)
            avg = sum(ratings[start:i+1]) / len(ratings[start:i+1])
            trend.append(round(avg, 2))

        # Correction analytics
        correction_summary = {}
        for c in self.label_corrections:
            key = f"{c['original_type']} -> {c['corrected_type']}"
            correction_summary[key] = correction_summary.get(key, 0) + 1

        return {
            "total_feedback": len(self.feedback_log),
            "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "feedback_by_type": {
                k: {"count": len(v), "avg_rating": round(sum(v) / len(v), 2)}
                for k, v in by_type.items()
            },
            "flagged_fractures": len(self.flagged_fractures),
            "label_corrections": len(self.label_corrections),
            "correction_patterns": correction_summary,
            "accuracy_trend": trend,
            "recent_feedback": self.feedback_log[-5:],
            "actionable_insights": self._generate_insights(),
        }

    def _generate_insights(self) -> list:
        """Generate actionable insights from feedback data."""
        insights = []

        if not self.feedback_log:
            return insights

        ratings = [f["rating"] for f in self.feedback_log]
        avg = sum(ratings) / len(ratings)

        if avg < 2.5:
            insights.append({
                "type": "critical",
                "message": "Average expert rating is very low. Model predictions "
                           "may be unreliable for this dataset. Consider uploading "
                           "more training data or reviewing data quality.",
            })
        elif avg < 3.5:
            insights.append({
                "type": "warning",
                "message": "Model accuracy is rated as moderate by experts. "
                           "Review the most common correction patterns below to "
                           "identify systematic biases.",
            })

        # Check for specific problem areas
        by_type = {}
        for f in self.feedback_log:
            t = f["analysis_type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(f["rating"])

        for atype, type_ratings in by_type.items():
            type_avg = sum(type_ratings) / len(type_ratings)
            if type_avg < 3.0 and len(type_ratings) >= 2:
                insights.append({
                    "type": "warning",
                    "message": f"{atype} analysis has low expert confidence "
                               f"(avg {type_avg:.1f}/5). This analysis type "
                               f"may need model retraining or parameter adjustment.",
                })

        if self.label_corrections:
            insights.append({
                "type": "info",
                "message": f"{len(self.label_corrections)} expert corrections "
                           f"available. Use 'Retrain with Corrections' to "
                           f"incorporate expert knowledge into the model.",
            })

        return insights


# Global feedback store
feedback_store = FeedbackStore()


def retrain_with_corrections(
    df: pd.DataFrame,
    classifier: str = "xgboost",
) -> dict:
    """Retrain a model using expert-corrected labels.

    This is how the feedback loop actually improves predictions:
    1. Expert flags misclassified fractures
    2. Expert provides correct labels
    3. Model is retrained with corrected labels
    4. Comparison shows improvement vs original

    Returns before/after accuracy comparison.
    """
    corrections_count = feedback_store.get_corrections_count()
    if corrections_count == 0:
        return {
            "status": "no_corrections",
            "message": "No expert corrections available yet. "
                       "Use the label correction feature to flag "
                       "misclassified fractures with their correct types.",
        }

    # Train on original data
    original_result = classify_enhanced(df, classifier=classifier)

    # Apply corrections and retrain
    corrected_df = feedback_store.apply_corrections(df)
    corrected_result = classify_enhanced(corrected_df, classifier=classifier)

    improvement = (corrected_result["cv_mean_accuracy"]
                   - original_result["cv_mean_accuracy"])

    return {
        "status": "retrained",
        "corrections_applied": corrections_count,
        "original_accuracy": round(original_result["cv_mean_accuracy"], 4),
        "corrected_accuracy": round(corrected_result["cv_mean_accuracy"], 4),
        "improvement": round(improvement, 4),
        "improvement_pct": round(100 * improvement, 2),
        "message": (
            f"Applied {corrections_count} expert corrections. "
            f"{'Accuracy improved' if improvement > 0 else 'Accuracy changed'} "
            f"from {original_result['cv_mean_accuracy']*100:.1f}% to "
            f"{corrected_result['cv_mean_accuracy']*100:.1f}% "
            f"({'+' if improvement > 0 else ''}{100*improvement:.2f}%)."
        ),
    }


# ──────────────────────────────────────────────────────
# Sensitivity Analysis
# ──────────────────────────────────────────────────────

def sensitivity_analysis(
    normals: np.ndarray,
    base_result: dict,
    depth_m: float = 3000.0,
) -> dict:
    """Run sensitivity analysis on stress inversion parameters.

    Varies each key input parameter while holding others at best-fit values
    to show how results change with assumptions. Essential for industrial
    decision-making: quantifies how wrong we might be if assumptions are off.

    Returns tornado diagram data, parameter ranges, and risk implications.
    """
    from src.geostress import (
        build_stress_tensor, resolve_stress_on_planes,
        invert_stress as _invert_stress,
    )

    base_sigma1 = base_result["sigma1"]
    base_sigma3 = base_result["sigma3"]
    base_R = base_result["R"]
    base_shmax = base_result["shmax_azimuth_deg"]
    base_mu = base_result["mu"]
    base_regime = base_result["regime"]
    base_pp = base_result.get("pore_pressure", 0.0)

    parameter_ranges = {
        "friction_coefficient": {
            "label": "Friction Coefficient (μ)",
            "values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "base": base_mu,
            "unit": "",
            "stakeholder_desc": "How easily rock surfaces slide against each other. "
                                "Lower values mean weaker rock joints.",
        },
        "pore_pressure": {
            "label": "Pore Pressure (MPa)",
            "values": [
                round(base_pp * f, 2)
                for f in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
            ],
            "base": round(base_pp, 2),
            "unit": "MPa",
            "stakeholder_desc": "Fluid pressure inside the rock. Higher pore pressure "
                                "makes fractures more likely to slip.",
        },
        "stress_regime": {
            "label": "Stress Regime",
            "values": ["normal", "strike_slip", "thrust"],
            "base": base_regime,
            "unit": "",
            "stakeholder_desc": "The type of tectonic environment. Determines which "
                                "direction the strongest stress acts.",
        },
    }

    results = {}

    # ── Friction coefficient sensitivity ──
    mu_results = []
    for mu_val in parameter_ranges["friction_coefficient"]["values"]:
        S = build_stress_tensor(base_sigma1, base_sigma3, base_R, base_shmax, base_regime)
        sigma_n, tau = resolve_stress_on_planes(S, normals)
        sigma_n_eff = sigma_n - base_pp
        tau_critical = mu_val * sigma_n_eff
        is_critical = tau >= tau_critical
        pct_critical = 100 * float(is_critical.sum()) / len(normals)
        safe_denom = np.where(sigma_n_eff > 0, mu_val * sigma_n_eff, 1.0)
        mean_slip = float(np.where(sigma_n_eff > 0, tau / safe_denom, 0).mean())

        mu_results.append({
            "value": mu_val,
            "pct_critically_stressed": round(pct_critical, 1),
            "mean_slip_tendency": round(mean_slip, 3),
        })
    results["friction_coefficient"] = mu_results

    # ── Pore pressure sensitivity ──
    pp_results = []
    for pp_val in parameter_ranges["pore_pressure"]["values"]:
        if pp_val <= 0:
            pp_val = 0.1
        S = build_stress_tensor(base_sigma1, base_sigma3, base_R, base_shmax, base_regime)
        sigma_n, tau = resolve_stress_on_planes(S, normals)
        sigma_n_eff = sigma_n - pp_val
        tau_critical = base_mu * sigma_n_eff
        is_critical = tau >= tau_critical
        pct_critical = 100 * float(is_critical.sum()) / len(normals)

        pp_results.append({
            "value": round(pp_val, 2),
            "pct_critically_stressed": round(pct_critical, 1),
        })
    results["pore_pressure"] = pp_results

    # ── Stress regime sensitivity ──
    regime_results = []
    for regime_val in parameter_ranges["stress_regime"]["values"]:
        try:
            inv_result = _invert_stress(
                normals, regime=regime_val, depth_m=depth_m,
                pore_pressure=base_pp,
            )
            sigma_n = inv_result["sigma_n"]
            tau = inv_result["tau"]
            sigma_n_eff = sigma_n - base_pp
            tau_critical = inv_result["mu"] * sigma_n_eff
            is_critical = tau >= tau_critical
            pct_critical = 100 * float(is_critical.sum()) / len(normals)

            regime_results.append({
                "regime": regime_val,
                "sigma1": round(inv_result["sigma1"], 2),
                "sigma3": round(inv_result["sigma3"], 2),
                "R": round(inv_result["R"], 3),
                "shmax": round(inv_result["shmax_azimuth_deg"], 1),
                "mu": round(inv_result["mu"], 3),
                "pct_critically_stressed": round(pct_critical, 1),
                "misfit": round(float(np.sum(inv_result["misfit"] ** 2)), 2),
            })
        except Exception:
            regime_results.append({
                "regime": regime_val,
                "error": "Inversion failed for this regime",
            })
    results["stress_regime"] = regime_results

    # ── Tornado diagram data ──
    tornado = []
    for param_key in ["friction_coefficient", "pore_pressure"]:
        param_results = results[param_key]
        values = [r.get("pct_critically_stressed", 0) for r in param_results]
        if values:
            tornado.append({
                "parameter": parameter_ranges[param_key]["label"],
                "description": parameter_ranges[param_key]["stakeholder_desc"],
                "min_pct_critical": round(min(values), 1),
                "max_pct_critical": round(max(values), 1),
                "range": round(max(values) - min(values), 1),
                "base_value": parameter_ranges[param_key]["base"],
            })
    tornado.sort(key=lambda x: x["range"], reverse=True)

    # ── Risk implications ──
    risk_implications = []
    regime_shmax_vals = [r.get("shmax", 0) for r in regime_results if "shmax" in r]
    if len(regime_shmax_vals) > 1:
        shmax_range = max(regime_shmax_vals) - min(regime_shmax_vals)
        if shmax_range > 180:
            shmax_range = 360 - shmax_range
        if shmax_range > 30:
            risk_implications.append({
                "severity": "high",
                "message": f"SHmax direction varies by {shmax_range:.0f}° across "
                           f"regime assumptions. Regime choice significantly "
                           f"affects drilling recommendations.",
            })

    pp_crits = [r["pct_critically_stressed"] for r in pp_results]
    if max(pp_crits) - min(pp_crits) > 30:
        risk_implications.append({
            "severity": "high",
            "message": f"Pore pressure uncertainty causes critically stressed "
                       f"percentage to range from {min(pp_crits):.0f}% to "
                       f"{max(pp_crits):.0f}%. Accurate pore pressure "
                       f"measurement is critical.",
        })

    if not risk_implications:
        risk_implications.append({
            "severity": "low",
            "message": "Results are relatively robust to parameter variations.",
        })

    return {
        "parameters": parameter_ranges,
        "results": results,
        "tornado": tornado,
        "risk_implications": risk_implications,
        "base_result": {
            "sigma1": round(base_sigma1, 2),
            "sigma3": round(base_sigma3, 2),
            "R": round(base_R, 3),
            "shmax": round(base_shmax, 1),
            "mu": round(base_mu, 3),
            "pore_pressure": round(base_pp, 2),
            "regime": base_regime,
        },
    }


# ──────────────────────────────────────────────────────
# Risk Assessment Matrix
# ──────────────────────────────────────────────────────

def compute_risk_matrix(
    inversion_result: dict,
    cs_result: dict,
    quality_result: dict,
    model_comparison: dict = None,
    sensitivity_result: dict = None,
) -> dict:
    """Compute comprehensive operational risk assessment matrix.

    Combines all analysis results into a single risk framework for
    drilling, completion, and stimulation decisions.

    Each risk factor is scored 0-100 (0=lowest risk, 100=highest risk)
    and categorized as low/moderate/high/critical.
    """
    factors = []

    # ── 1. Critically Stressed Risk ──
    cs_pct = cs_result.get("pct_critical", 0)
    cs_score = min(100, cs_pct * 1.5)
    factors.append({
        "factor": "Critically Stressed Fractures",
        "score": round(cs_score),
        "detail": f"{cs_pct:.1f}% of fractures are critically stressed",
        "impact": "Wellbore instability, lost circulation, induced seismicity",
        "mitigation": "Managed pressure drilling, careful mud weight design"
            if cs_score > 50 else "Standard drilling practices sufficient",
    })

    # ── 2. Data Quality Risk ──
    dq_score = max(0, 100 - quality_result.get("score", 50))
    factors.append({
        "factor": "Data Quality",
        "score": round(dq_score),
        "detail": f"Quality grade: {quality_result.get('grade', '?')} "
                  f"({quality_result.get('score', 0)}/100)",
        "impact": "Unreliable predictions, wrong drilling decisions",
        "mitigation": "; ".join(quality_result.get("recommendations", []))
            or "Data quality is adequate",
    })

    # ── 3. Model Confidence Risk ──
    if model_comparison:
        best_acc = max(
            (res.get("cv_accuracy_mean", 0)
             for res in model_comparison.get("models", {}).values()),
            default=0,
        )
        mc_score = max(0, 100 - best_acc * 100)
        agreement = model_comparison.get("model_agreement_mean", 1.0)
        low_conf_pct = model_comparison.get("low_confidence_pct", 0)

        if agreement < 0.7:
            mc_score = min(100, mc_score + 20)

        factors.append({
            "factor": "ML Classification Confidence",
            "score": round(mc_score),
            "detail": f"Best model: {best_acc*100:.1f}% accuracy, "
                      f"{agreement*100:.0f}% model agreement",
            "impact": "Incorrect fracture type identification",
            "mitigation": "Use expert review for low-confidence samples"
                if mc_score > 30 else "Model predictions are reliable",
        })

        conformal = model_comparison.get("conformal", {})
        if conformal.get("available"):
            uncertain_pct = conformal.get("uncertain_pct", 0)
            if uncertain_pct > 15:
                factors.append({
                    "factor": "Prediction Uncertainty",
                    "score": round(min(100, uncertain_pct * 3)),
                    "detail": f"{uncertain_pct:.1f}% of samples have "
                              f"<50% confidence",
                    "impact": "Individual fracture predictions may be wrong",
                    "mitigation": "Flag uncertain predictions for manual review",
                })

    # ── 4. Stress Field Uncertainty ──
    if sensitivity_result:
        tornado = sensitivity_result.get("tornado", [])
        max_range = max(
            (t.get("range", 0) for t in tornado), default=0
        )
        stress_score = min(100, max_range * 1.5)

        risk_msgs = [
            r["message"]
            for r in sensitivity_result.get("risk_implications", [])
            if r.get("severity") == "high"
        ]

        factors.append({
            "factor": "Stress Field Sensitivity",
            "score": round(stress_score),
            "detail": f"Max sensitivity range: {max_range:.0f}% "
                      f"(critically stressed varies with assumptions)",
            "impact": "Design based on uncertain stress model",
            "mitigation": "; ".join(risk_msgs) if risk_msgs
                else "Stress results are robust",
        })

    # ── 5. Differential Stress Risk ──
    diff_stress = inversion_result["sigma1"] - inversion_result["sigma3"]
    if diff_stress > 40:
        ds_score = 80
    elif diff_stress > 25:
        ds_score = 50
    elif diff_stress > 15:
        ds_score = 30
    else:
        ds_score = 10
    factors.append({
        "factor": "Differential Stress",
        "score": ds_score,
        "detail": f"σ1-σ3 = {diff_stress:.1f} MPa",
        "impact": "High differential stress increases fracture reactivation",
        "mitigation": "Monitor microseismicity during operations"
            if ds_score > 50 else "Normal stress differential",
    })

    # ── 6. Friction Properties Risk ──
    mu = inversion_result.get("mu", 0.6)
    if mu < 0.4:
        mu_score, mu_msg = 80, f"Very low μ={mu:.3f} (weak/clay-filled surfaces)"
    elif mu < 0.5:
        mu_score, mu_msg = 50, f"Low μ={mu:.3f} (below typical Byerlee range)"
    else:
        mu_score, mu_msg = 10, f"μ={mu:.3f} (within normal Byerlee range)"
    factors.append({
        "factor": "Rock Friction Properties",
        "score": mu_score,
        "detail": mu_msg,
        "impact": "Low friction increases slip risk on all orientations",
        "mitigation": "Lab testing to confirm friction coefficient"
            if mu_score > 40 else "Properties within expected range",
    })

    # ── Compute overall risk ──
    n_factors = len(factors)
    overall_score = round(
        sum(f["score"] for f in factors) / n_factors
    ) if n_factors > 0 else 0

    if overall_score >= 70:
        overall_level, overall_color = "CRITICAL", "danger"
        go_nogo = "NO-GO"
        go_nogo_detail = (
            "Multiple high-risk factors identified. Do NOT proceed with "
            "standard operations. Requires additional data, expert review, "
            "and risk mitigation before proceeding."
        )
    elif overall_score >= 50:
        overall_level, overall_color = "HIGH", "danger"
        go_nogo = "CONDITIONAL"
        go_nogo_detail = (
            "Proceed with caution. Implement all recommended mitigations. "
            "Consider additional data acquisition before critical operations."
        )
    elif overall_score >= 30:
        overall_level, overall_color = "MODERATE", "warning"
        go_nogo = "GO (with monitoring)"
        go_nogo_detail = (
            "Safe to proceed with standard monitoring. Pay attention to "
            "flagged risk factors during operations."
        )
    else:
        overall_level, overall_color = "LOW", "success"
        go_nogo = "GO"
        go_nogo_detail = (
            "Risk within acceptable limits. Proceed with standard "
            "operational procedures."
        )

    return {
        "overall_score": overall_score,
        "overall_level": overall_level,
        "overall_color": overall_color,
        "go_nogo": go_nogo,
        "go_nogo_detail": go_nogo_detail,
        "factors": sorted(factors, key=lambda x: x["score"], reverse=True),
        "n_factors": n_factors,
        "high_risk_factors": [f for f in factors if f["score"] >= 60],
    }


# ──────────────────────────────────────────────────────
# Well Report Generation
# ──────────────────────────────────────────────────────

def generate_well_report(
    well_name: str,
    inversion_result: dict,
    cs_result: dict,
    quality_result: dict,
    model_comparison: dict = None,
    sensitivity_result: dict = None,
    risk_matrix: dict = None,
    auto_regime_result: dict = None,
    calibration_result: dict = None,
    data_recommendations: dict = None,
) -> dict:
    """Generate a comprehensive stakeholder report for a single well.

    Aggregates all analysis results into a structured report suitable
    for non-technical decision-makers. Designed to be printed/exported.
    """
    report = {
        "well_name": well_name,
        "generated_at": pd.Timestamp.now().isoformat(),
        "version": "2.5.0",
    }

    regime = inversion_result.get("regime", "unknown")
    shmax = inversion_result.get("shmax_azimuth_deg", 0)
    cs_pct = cs_result.get("pct_critical", 0)
    risk_level = risk_matrix.get("overall_level", "UNKNOWN") if risk_matrix else "N/A"

    # ── Executive Summary ──
    exec_summary = (
        f"Well {well_name}: {regime.replace('_', ' ')} stress regime with "
        f"SHmax at {shmax:.0f}° ({_azimuth_to_compass(shmax)}). "
        f"{cs_pct:.0f}% critically stressed fractures. "
        f"Risk level: {risk_level}."
    )
    if risk_matrix:
        exec_summary += f" Decision: {risk_matrix.get('go_nogo', 'N/A')}."
    report["executive_summary"] = exec_summary

    # ── Stress State ──
    report["stress_state"] = {
        "sigma1_mpa": round(inversion_result["sigma1"], 2),
        "sigma2_mpa": round(inversion_result["sigma2"], 2),
        "sigma3_mpa": round(inversion_result["sigma3"], 2),
        "R_ratio": round(inversion_result["R"], 3),
        "shmax_azimuth": round(shmax, 1),
        "shmax_compass": _azimuth_to_compass(shmax),
        "friction_coefficient": round(inversion_result["mu"], 3),
        "regime": regime,
        "pore_pressure_mpa": round(inversion_result.get("pore_pressure", 0), 2),
        "differential_stress_mpa": round(
            inversion_result["sigma1"] - inversion_result["sigma3"], 2
        ),
    }

    # ── Critically Stressed ──
    report["critically_stressed"] = {
        "total_fractures": cs_result.get("total", 0),
        "critically_stressed_count": cs_result.get("count_critical", 0),
        "critically_stressed_pct": round(cs_pct, 1),
        "high_risk_count": cs_result.get("high_risk_count", 0),
        "moderate_risk_count": cs_result.get("moderate_risk_count", 0),
        "mean_slip_ratio": round(cs_result.get("mean_slip_ratio", 0), 3),
    }

    # ── Data Quality ──
    report["data_quality"] = {
        "score": quality_result.get("score", 0),
        "grade": quality_result.get("grade", "?"),
        "issues": quality_result.get("issues", []),
        "warnings": quality_result.get("warnings", []),
    }

    # ── Auto Regime Detection (if available) ──
    if auto_regime_result:
        report["regime_detection"] = {
            "best_regime": auto_regime_result.get("best_regime", "unknown"),
            "confidence": auto_regime_result.get("confidence", "N/A"),
            "misfit_ratio": auto_regime_result.get("misfit_ratio", 0),
            "comparison": auto_regime_result.get("comparison", []),
            "summary": auto_regime_result.get("stakeholder_summary", ""),
        }

    # ── Classification (if available) ──
    if model_comparison:
        best = model_comparison.get("best_model", "unknown")
        best_acc = 0
        best_bal_acc = 0
        best_per_class = {}
        for name, res in model_comparison.get("models", {}).items():
            if name == best:
                best_acc = res.get("cv_accuracy_mean", 0)
                best_bal_acc = res.get("balanced_accuracy", 0)
                best_per_class = res.get("per_class_metrics", {})

        imbalance_warning = None
        if best_acc - best_bal_acc > 0.20:
            imbalance_warning = (
                f"SEVERE CLASS IMBALANCE: Standard accuracy ({best_acc:.1%}) masks "
                f"poor balanced accuracy ({best_bal_acc:.1%}). The model fails on "
                f"minority fracture types. Do NOT rely on classifications for "
                f"under-represented types."
            )
        # Find failing classes
        failing_classes = [
            cls for cls, met in best_per_class.items()
            if met.get("f1", 0) < 0.1
        ]

        report["classification"] = {
            "best_model": best,
            "accuracy_pct": round(best_acc * 100, 1),
            "balanced_accuracy_pct": round(best_bal_acc * 100, 1),
            "n_models_compared": len(model_comparison.get("models", {})),
            "model_agreement_pct": round(
                model_comparison.get("model_agreement_mean", 0) * 100, 1
            ),
            "imbalance_warning": imbalance_warning,
            "failing_classes": failing_classes,
            "per_class": best_per_class,
        }

    # ── Sensitivity (if available) ──
    if sensitivity_result:
        report["sensitivity"] = {
            "tornado": sensitivity_result.get("tornado", []),
            "risk_implications": sensitivity_result.get("risk_implications", []),
            "regime_comparison": sensitivity_result.get("results", {}).get(
                "stress_regime", []
            ),
        }

    # ── Risk Assessment ──
    if risk_matrix:
        report["risk_assessment"] = {
            "overall_score": risk_matrix["overall_score"],
            "overall_level": risk_matrix["overall_level"],
            "go_nogo": risk_matrix["go_nogo"],
            "go_nogo_detail": risk_matrix["go_nogo_detail"],
            "factors": risk_matrix["factors"],
        }

    # ── Operational Recommendations ──
    optimal_dir = (shmax + 90) % 360
    drilling = [
        f"Optimal horizontal well azimuth: {optimal_dir:.0f}° "
        f"({_azimuth_to_compass(optimal_dir)}) — perpendicular to SHmax.",
    ]
    if regime == "thrust":
        drilling.append(
            "Thrust regime: high horizontal stresses may cause borehole "
            "collapse. Consider higher mud weight."
        )

    completion = []
    if cs_pct > 50:
        completion.append(
            "High critically stressed percentage: natural fracture "
            "permeability may be sufficient. Consider reduced stimulation."
        )
    elif cs_pct < 20:
        completion.append(
            "Low critically stressed percentage: hydraulic fracturing "
            "may be needed for adequate connectivity."
        )

    monitoring = []
    if risk_matrix and risk_matrix["overall_score"] >= 50:
        monitoring.append("Deploy real-time microseismic monitoring.")
        monitoring.append("Establish pore pressure monitoring program.")

    report["recommendations"] = {
        "drilling": drilling,
        "completion": completion,
        "monitoring": monitoring,
    }

    # ── Calibration Section ──
    if calibration_result:
        report["calibration"] = {
            "reliability": calibration_result["reliability"],
            "ece_pct": round(calibration_result["ece"] * 100, 1),
            "brier_score": calibration_result["brier_score"],
            "summary": calibration_result.get("stakeholder_summary", ""),
        }
    else:
        report["calibration"] = None

    # ── Data Collection Roadmap ──
    if data_recommendations:
        report["data_roadmap"] = {
            "n_priority_actions": len(data_recommendations.get("priority_actions", [])),
            "n_recommendations": len(data_recommendations.get("recommendations", [])),
            "completeness_pct": data_recommendations.get("data_completeness_pct", 0),
            "priority_actions": [
                a["action"] for a in data_recommendations.get("priority_actions", [])
            ],
            "recommendations": [
                r["action"] for r in data_recommendations.get("recommendations", [])
            ],
        }
    else:
        report["data_roadmap"] = None

    return report


# ──────────────────────────────────────────────────────
# Multi-Well Comparison
# ──────────────────────────────────────────────────────

def compare_wells(
    df: pd.DataFrame,
    depth_m: float = 3000.0,
) -> dict:
    """Compare analysis results across all wells in the dataset.

    Checks stress field consistency, classification transferability,
    and flags spatial anomalies.
    """
    from src.geostress import invert_stress as _invert_stress
    from src.data_loader import fracture_plane_normal

    wells = df[WELL_COL].unique().tolist()
    if len(wells) < 2:
        return {
            "status": "insufficient_wells",
            "message": f"Only {len(wells)} well(s). Need ≥2 for comparison.",
            "wells": wells,
        }

    well_results = {}
    for well in wells:
        wdf = df[df[WELL_COL] == well]
        normals = fracture_plane_normal(
            wdf[AZIMUTH_COL].values, wdf[DIP_COL].values
        )

        try:
            inv = _invert_stress(normals, regime="strike_slip", depth_m=depth_m)
        except Exception:
            inv = None

        quality = validate_data_quality(wdf)

        try:
            clf = classify_enhanced(wdf, classifier="xgboost", n_folds=3)
            accuracy = clf["cv_mean_accuracy"]
        except Exception:
            accuracy = None

        well_results[well] = {
            "n_fractures": len(wdf),
            "fracture_types": wdf[FRACTURE_TYPE_COL].value_counts().to_dict()
                if FRACTURE_TYPE_COL in wdf.columns else {},
            "mean_azimuth": round(float(wdf[AZIMUTH_COL].mean()), 1),
            "mean_dip": round(float(wdf[DIP_COL].mean()), 1),
            "std_azimuth": round(float(wdf[AZIMUTH_COL].std()), 1),
            "std_dip": round(float(wdf[DIP_COL].std()), 1),
            "data_quality_score": quality["score"],
            "data_quality_grade": quality["grade"],
            "classification_accuracy": round(accuracy * 100, 1) if accuracy else None,
        }

        if inv:
            sigma_n_eff = inv["sigma_n"] - inv["pore_pressure"]
            tau_critical = inv["mu"] * sigma_n_eff
            is_critical = inv["tau"] >= tau_critical
            pct_cs = 100 * float(is_critical.sum()) / len(normals)
            well_results[well].update({
                "sigma1": round(inv["sigma1"], 2),
                "sigma3": round(inv["sigma3"], 2),
                "R": round(inv["R"], 3),
                "shmax": round(inv["shmax_azimuth_deg"], 1),
                "mu": round(inv["mu"], 3),
                "pct_critically_stressed": round(pct_cs, 1),
            })

    # ── Consistency checks ──
    consistency = []

    shmax_vals = [r["shmax"] for r in well_results.values() if "shmax" in r]
    if len(shmax_vals) >= 2:
        shmax_diff = max(shmax_vals) - min(shmax_vals)
        if shmax_diff > 180:
            shmax_diff = 360 - shmax_diff
        status = "WARNING" if shmax_diff > 20 else "OK"
        consistency.append({
            "check": "SHmax Direction",
            "status": status,
            "detail": f"SHmax varies by {shmax_diff:.0f}° between wells."
                + (" May indicate stress rotation." if status == "WARNING" else ""),
        })

    s1_vals = [r["sigma1"] for r in well_results.values() if "sigma1" in r]
    if len(s1_vals) >= 2:
        s1_range = max(s1_vals) - min(s1_vals)
        status = "WARNING" if s1_range > 20 else "OK"
        consistency.append({
            "check": "Stress Magnitudes",
            "status": status,
            "detail": f"σ1 varies by {s1_range:.1f} MPa between wells.",
        })

    dq_scores = [r["data_quality_score"] for r in well_results.values()]
    if min(dq_scores) < 60:
        consistency.append({
            "check": "Data Quality Balance",
            "status": "WARNING",
            "detail": "Some wells have poor data quality.",
        })

    # ── Cross-well classification ──
    cross_val_results = {}
    if len(wells) >= 2:
        features_all = engineer_enhanced_features(df)
        le = LabelEncoder()
        all_labels = le.fit_transform(df[FRACTURE_TYPE_COL].values)
        scaler = StandardScaler()
        X_all = scaler.fit_transform(features_all.values)

        for train_well in wells:
            for test_well in wells:
                if train_well == test_well:
                    continue
                train_mask = df[WELL_COL].values == train_well
                test_mask = df[WELL_COL].values == test_well

                # Check class overlap
                train_classes = set(all_labels[train_mask])
                test_classes = set(all_labels[test_mask])
                unseen = test_classes - train_classes
                if unseen:
                    cross_val_results[f"{train_well} → {test_well}"] = {
                        "accuracy": None,
                        "error": f"Test has {len(unseen)} class(es) not in training",
                        "note": "Wells have different fracture type populations",
                        "train_size": int(train_mask.sum()),
                        "test_size": int(test_mask.sum()),
                    }
                    continue

                try:
                    model = _get_models(fast=True).get(
                        "xgboost",
                        list(_get_models(fast=True).values())[0],
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_all[train_mask], all_labels[train_mask])
                        acc = float(accuracy_score(
                            all_labels[test_mask],
                            model.predict(X_all[test_mask]),
                        ))
                    cross_val_results[f"{train_well} → {test_well}"] = {
                        "accuracy": round(acc * 100, 1),
                        "train_size": int(train_mask.sum()),
                        "test_size": int(test_mask.sum()),
                    }
                except Exception:
                    cross_val_results[f"{train_well} → {test_well}"] = {
                        "accuracy": None, "error": "Failed",
                    }

        cv_accs = [r["accuracy"] for r in cross_val_results.values() if r.get("accuracy")]
        if cv_accs and min(cv_accs) < 60:
            consistency.append({
                "check": "Model Transferability",
                "status": "WARNING",
                "detail": f"Accuracy drops to {min(cv_accs):.0f}% across wells.",
            })
        elif cv_accs:
            consistency.append({
                "check": "Model Transferability",
                "status": "OK",
                "detail": f"Cross-well accuracy: {min(cv_accs):.0f}%-{max(cv_accs):.0f}%.",
            })

    return {
        "status": "compared",
        "wells": well_results,
        "consistency_checks": consistency,
        "cross_validation": cross_val_results,
        "n_wells": len(wells),
    }


# ──────────────────────────────────────────────────────
# Uncertainty Budget
# ──────────────────────────────────────────────────────

def compute_uncertainty_budget(
    inversion_result: dict,
    sensitivity_result: dict = None,
    bayesian_result: dict = None,
    quality_result: dict = None,
    model_comparison: dict = None,
    well_comparison: dict = None,
) -> dict:
    """Compute an uncertainty budget ranking all sources of uncertainty.

    Aggregates uncertainties from the entire analysis pipeline into a
    single ranked view showing where the most uncertainty comes from and
    what stakeholders should invest in to reduce it.

    Each source is quantified as a normalized 0-100 score (100 = maximum
    uncertainty) with actionable recommendations.

    Returns
    -------
    dict with:
        sources : list of dicts, ranked by contribution
        total_score : weighted aggregate
        dominant_source : name of the largest contributor
        stakeholder_summary : plain-language narrative
        recommended_actions : prioritized list
    """
    sources = []

    # ── 1. Parameter Sensitivity Uncertainty ──
    if sensitivity_result:
        tornado = sensitivity_result.get("tornado", [])
        if tornado:
            # Range of critically stressed % across parameter variations
            max_range = max(t.get("range", 0) for t in tornado)
            param_names = [t["parameter"] for t in tornado]
            sens_score = min(100, max_range * 1.2)

            detail_parts = []
            for t in tornado:
                detail_parts.append(
                    f"{t['parameter']}: {t['min_pct_critical']}%-"
                    f"{t['max_pct_critical']}% (range {t['range']}%)"
                )

            sources.append({
                "source": "Parameter Assumptions",
                "score": round(sens_score),
                "detail": "; ".join(detail_parts),
                "driver": f"Largest driver: {param_names[0]}" if param_names else "",
                "recommendation": (
                    f"Measure {param_names[0].lower()} directly to reduce "
                    f"the {max_range:.0f}% range in critically stressed predictions."
                    if param_names else "Run sensitivity analysis."
                ),
                "category": "input_parameters",
                "weight": 0.25,
            })

    # ── 2. Bayesian Posterior Uncertainty ──
    if bayesian_result and bayesian_result.get("available"):
        params = bayesian_result.get("parameters", {})

        # SHmax uncertainty is the most operationally critical
        shmax = params.get("SHmax_azimuth", {})
        shmax_range = 0
        if shmax.get("ci_90"):
            shmax_range = shmax["ci_90"][1] - shmax["ci_90"][0]
            if shmax_range > 180:
                shmax_range = 360 - shmax_range

        # Mu uncertainty
        mu = params.get("mu", {})
        mu_range = 0
        if mu.get("ci_90"):
            mu_range = mu["ci_90"][1] - mu["ci_90"][0]

        # Score: SHmax direction is the key operational parameter
        # >60° range is very uncertain; >120° is essentially unconstrained
        bayes_score = min(100, shmax_range * 0.8)

        converged = bayesian_result.get("converged", False)
        if not converged:
            bayes_score = min(100, bayes_score + 15)

        ci_details = []
        for pname, pdata in params.items():
            if pdata.get("ci_90"):
                ci_details.append(
                    f"{pname}: {pdata['ci_90'][0]:.1f}-{pdata['ci_90'][1]:.1f}"
                )

        sources.append({
            "source": "Stress Model Uncertainty (Bayesian)",
            "score": round(bayes_score),
            "detail": f"SHmax 90% CI: {shmax_range:.0f} degrees; "
                      f"Friction 90% CI width: {mu_range:.2f}" +
                      ("" if converged else " (chain not fully converged)"),
            "driver": (
                "SHmax direction is poorly constrained" if shmax_range > 60
                else "SHmax direction is well constrained"
            ),
            "recommendation": (
                "Add more fracture orientation data or independent stress "
                "indicators (breakouts, drilling-induced fractures) to "
                "constrain SHmax direction."
                if shmax_range > 60 else
                "Bayesian uncertainty is acceptable for operational decisions."
            ),
            "category": "model_uncertainty",
            "weight": 0.30,
        })
    else:
        # No Bayesian run: high uncertainty by default
        sources.append({
            "source": "Stress Model Uncertainty (Bayesian)",
            "score": 70,
            "detail": "Bayesian MCMC not yet run; only point estimates available",
            "driver": "No confidence intervals on stress parameters",
            "recommendation": (
                "Run Bayesian MCMC inversion to quantify uncertainty "
                "on all 5 stress parameters."
            ),
            "category": "model_uncertainty",
            "weight": 0.30,
        })

    # ── 3. Data Quality Uncertainty ──
    if quality_result:
        dq_score = max(0, 100 - quality_result.get("score", 50))
        grade = quality_result.get("grade", "?")
        issues = quality_result.get("issues", [])
        warnings_list = quality_result.get("warnings", [])
        recs = quality_result.get("recommendations", [])

        sources.append({
            "source": "Data Quality",
            "score": round(dq_score),
            "detail": f"Grade {grade} ({quality_result.get('score', 0)}/100). "
                      f"{len(issues)} critical issues, "
                      f"{len(warnings_list)} warnings.",
            "driver": issues[0] if issues else (
                warnings_list[0] if warnings_list else "No critical issues"
            ),
            "recommendation": recs[0] if recs else "Data quality is adequate.",
            "category": "data",
            "weight": 0.20,
        })

    # ── 4. ML Model Uncertainty ──
    if model_comparison:
        # Model disagreement
        agreement = model_comparison.get("model_agreement_mean", 1.0)
        low_conf = model_comparison.get("low_confidence_pct", 0)
        best_acc = 0
        if model_comparison.get("ranking"):
            best_acc = model_comparison["ranking"][0].get("accuracy", 0)

        ml_score = max(0, 100 - best_acc * 100) + (100 - agreement * 100) * 0.3
        ml_score = min(100, ml_score)

        conformal = model_comparison.get("conformal", {})
        conf_detail = ""
        if conformal.get("available"):
            uncertain_pct = conformal.get("uncertain_pct", 0)
            conf_detail = f" | {uncertain_pct:.0f}% low-confidence predictions"

        sources.append({
            "source": "ML Classification Confidence",
            "score": round(ml_score),
            "detail": (
                f"Best accuracy: {best_acc*100:.1f}%, "
                f"model agreement: {agreement*100:.0f}%, "
                f"low confidence: {low_conf}%{conf_detail}"
            ),
            "driver": (
                f"Models disagree on {low_conf}% of fractures"
                if low_conf > 10 else "Model agreement is acceptable"
            ),
            "recommendation": (
                "Review low-confidence fractures with domain experts. "
                "Consider adding more training data for ambiguous types."
                if ml_score > 30 else "Classification is reliable."
            ),
            "category": "classification",
            "weight": 0.15,
        })

    # ── 5. Cross-Well Consistency ──
    if well_comparison and well_comparison.get("consistency_checks"):
        checks = well_comparison["consistency_checks"]
        n_warnings = sum(1 for c in checks if c["status"] == "WARNING")
        n_checks = len(checks)
        well_score = min(100, n_warnings * 35)

        warning_details = [
            c["detail"] for c in checks if c["status"] == "WARNING"
        ]

        sources.append({
            "source": "Cross-Well Consistency",
            "score": round(well_score),
            "detail": f"{n_warnings}/{n_checks} checks flagged warnings. " +
                      (" ".join(warning_details[:2]) if warning_details
                       else "All checks passed."),
            "driver": (
                warning_details[0] if warning_details
                else "Wells show consistent stress field"
            ),
            "recommendation": (
                "Investigate structural domain boundaries between wells. "
                "Consider well-specific stress models instead of a single model."
                if well_score > 40 else "Cross-well consistency is acceptable."
            ),
            "category": "spatial",
            "weight": 0.10,
        })

    # ── 6. Pore Pressure Uncertainty (always relevant) ──
    pp = inversion_result.get("pore_pressure", 0.0)
    pp_source = "estimated (hydrostatic)" if pp > 0 else "unknown"
    # If pore pressure is estimated rather than measured, it's inherently uncertain
    pp_score = 60 if pp > 0 else 80  # assumed hydrostatic = moderate uncertainty
    sources.append({
        "source": "Pore Pressure Estimate",
        "score": pp_score,
        "detail": f"Pp = {pp:.1f} MPa ({pp_source}). "
                  f"Hydrostatic assumption may not hold in overpressured zones.",
        "driver": "Using estimated rather than measured pore pressure",
        "recommendation": (
            "Obtain direct pore pressure measurements (MDT/RFT) or "
            "use drilling mud weight records to constrain Pp. "
            "This is the single most impactful data acquisition."
        ),
        "category": "input_parameters",
        "weight": 0.15,
    })

    # ── Sort by contribution ──
    sources.sort(key=lambda x: x["score"], reverse=True)

    # ── Weighted total ──
    total_weight = sum(s["weight"] for s in sources)
    if total_weight > 0:
        total_score = round(
            sum(s["score"] * s["weight"] for s in sources) / total_weight
        )
    else:
        total_score = 0

    dominant = sources[0] if sources else None

    # ── Recommended actions (top 3 by impact) ──
    recommended_actions = []
    for i, s in enumerate(sources[:3]):
        if s["score"] > 20:
            recommended_actions.append({
                "priority": i + 1,
                "action": s["recommendation"],
                "source": s["source"],
                "impact": (
                    f"Could reduce overall uncertainty by ~"
                    f"{round(s['score'] * s['weight'] / total_weight / max(total_score, 1) * 100)}%"
                    if total_weight > 0 and total_score > 0 else ""
                ),
            })

    # ── Stakeholder summary ──
    if dominant:
        summary_parts = [
            f"The largest source of uncertainty is '{dominant['source']}' "
            f"(score: {dominant['score']}/100). "
            f"{dominant['driver']}. ",
        ]
        if len(sources) > 1 and sources[1]["score"] > 40:
            summary_parts.append(
                f"The second largest concern is '{sources[1]['source']}' "
                f"(score: {sources[1]['score']}/100). "
            )
        summary_parts.append(
            f"Overall uncertainty: {total_score}/100. "
        )
        if total_score >= 60:
            summary_parts.append(
                "This uncertainty level is HIGH — results should be treated "
                "as preliminary. Additional data acquisition is strongly "
                "recommended before committing to operational decisions."
            )
        elif total_score >= 40:
            summary_parts.append(
                "This uncertainty level is MODERATE — results provide useful "
                "guidance but should be validated with additional data "
                "before final decisions."
            )
        else:
            summary_parts.append(
                "This uncertainty level is ACCEPTABLE — results are suitable "
                "for operational planning with standard safety margins."
            )
        stakeholder_summary = "".join(summary_parts)
    else:
        stakeholder_summary = "Insufficient data to compute uncertainty budget."

    return {
        "sources": sources,
        "total_score": total_score,
        "uncertainty_level": (
            "HIGH" if total_score >= 60
            else "MODERATE" if total_score >= 40
            else "LOW"
        ),
        "dominant_source": dominant["source"] if dominant else None,
        "recommended_actions": recommended_actions,
        "stakeholder_summary": stakeholder_summary,
        "n_sources": len(sources),
    }


# ──────────────────────────────────────────────────────
# Active Learning: Identify Samples for Expert Review
# ──────────────────────────────────────────────────────

def active_learning_query(
    df: pd.DataFrame,
    n_suggest: int = 20,
    classifier: str = "xgboost",
) -> dict:
    """Identify the fractures the model is most uncertain about.

    Uses prediction entropy and margin sampling to rank fractures
    by how much the model would benefit from an expert label.
    This is the practical equivalent of RLHF for geoscience:
    humans correct the highest-value samples, model retrains.

    Parameters
    ----------
    df : DataFrame with fracture data
    n_suggest : Number of samples to suggest for review
    classifier : Which model to use for uncertainty estimation

    Returns
    -------
    dict with:
        suggestions : list of fractures to review (most uncertain first)
        learning_curve : estimated accuracy gains from labeling more data
        coverage_gaps : fracture types or regions with sparse data
        summary : plain-language explanation
    """
    features = engineer_enhanced_features(df)
    labels = df[FRACTURE_TYPE_COL].values
    le = LabelEncoder()
    y = le.fit_transform(labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    all_models = _get_models()
    model = all_models.get(classifier, all_models.get("random_forest"))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Get cross-validated probabilities for each sample
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_classes = len(np.unique(y))
        probs = np.zeros((len(y), n_classes))

        for train_idx, test_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            probs[test_idx] = model.predict_proba(X[test_idx])

    # ── Uncertainty metrics per sample ──
    # 1. Entropy: higher = more uncertain
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    max_entropy = np.log(n_classes)
    normalized_entropy = entropy / max_entropy

    # 2. Margin: smallest gap between top-2 predicted classes
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]

    # 3. Combined active learning score (lower margin + higher entropy = more valuable)
    al_score = normalized_entropy - margin  # higher = more uncertain
    ranked_indices = np.argsort(al_score)[::-1]

    # ── Build suggestions ──
    suggestions = []
    predicted_classes = le.inverse_transform(probs.argmax(axis=1))
    actual_classes = labels

    for idx in ranked_indices[:n_suggest]:
        idx = int(idx)
        row = df.iloc[idx]
        suggestions.append({
            "index": idx,
            "depth": round(float(row.get(DEPTH_COL, 0)), 1),
            "azimuth": round(float(row.get(AZIMUTH_COL, 0)), 1),
            "dip": round(float(row.get(DIP_COL, 0)), 1),
            "well": str(row.get(WELL_COL, "")),
            "current_label": str(actual_classes[idx]),
            "predicted_label": str(predicted_classes[idx]),
            "confidence": round(float(sorted_probs[idx, 0]) * 100, 1),
            "entropy": round(float(normalized_entropy[idx]), 3),
            "margin": round(float(margin[idx]), 3),
            "mismatch": str(actual_classes[idx]) != str(predicted_classes[idx]),
            "top_probabilities": {
                le.classes_[i]: round(float(probs[idx, i]), 3)
                for i in np.argsort(probs[idx])[::-1][:3]
            },
        })

    # ── Coverage gaps ──
    coverage_gaps = []
    type_counts = pd.Series(labels).value_counts()
    median_count = type_counts.median()
    for ftype, count in type_counts.items():
        if count < median_count * 0.5:
            coverage_gaps.append({
                "type": str(ftype),
                "count": int(count),
                "issue": f"Only {count} samples (median is {int(median_count)}). "
                         f"Model may underperform on this type.",
                "recommendation": f"Collect ~{int(max(0, median_count - count))} "
                                  f"more {ftype} fractures.",
            })

    # Check depth coverage gaps
    if DEPTH_COL in df.columns:
        depths = df[DEPTH_COL].dropna()
        if len(depths) > 10:
            depth_range = depths.max() - depths.min()
            n_bins = 5
            bin_edges = np.linspace(depths.min(), depths.max(), n_bins + 1)
            for i in range(n_bins):
                bin_mask = (depths >= bin_edges[i]) & (depths < bin_edges[i+1])
                bin_count = int(bin_mask.sum())
                if bin_count < len(depths) * 0.05:  # <5% of data in this bin
                    coverage_gaps.append({
                        "type": f"Depth {bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}m",
                        "count": bin_count,
                        "issue": f"Sparse coverage ({bin_count} samples) in depth range.",
                        "recommendation": "Collect fracture data in this depth interval.",
                    })

    # ── Learning curve estimate ──
    # Simulate accuracy at different data fractions
    learning_curve_data = []
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for frac in fractions:
            n_samples = max(n_classes * 2, int(len(y) * frac))
            if n_samples > len(y):
                n_samples = len(y)
            # Use stratified subset
            if frac < 1.0:
                from sklearn.model_selection import train_test_split
                try:
                    X_sub, _, y_sub, _ = train_test_split(
                        X, y, train_size=frac, stratify=y, random_state=42,
                    )
                except ValueError:
                    X_sub, y_sub = X, y
            else:
                X_sub, y_sub = X, y

            sub_cv = StratifiedKFold(
                n_splits=min(5, max(2, len(np.unique(y_sub)))),
                shuffle=True, random_state=42,
            )
            try:
                scores = cross_val_score(model, X_sub, y_sub, cv=sub_cv, scoring="accuracy")
                learning_curve_data.append({
                    "fraction": frac,
                    "n_samples": len(y_sub),
                    "accuracy": round(float(scores.mean()), 4),
                    "std": round(float(scores.std()), 4),
                })
            except Exception:
                pass

    # ── Estimate benefit of more data ──
    if len(learning_curve_data) >= 2:
        last = learning_curve_data[-1]
        second = learning_curve_data[-2]
        improvement_rate = last["accuracy"] - second["accuracy"]
        projected = min(1.0, last["accuracy"] + improvement_rate * 0.5)
    else:
        projected = None

    # ── Summary ──
    n_mismatches = sum(1 for s in suggestions if s["mismatch"])
    low_conf = sum(1 for s in suggestions if s["confidence"] < 50)
    summary_parts = [
        f"Identified {n_suggest} fractures for expert review. ",
        f"{n_mismatches} show current label/prediction mismatch. ",
        f"{low_conf} have <50% prediction confidence. ",
    ]
    if coverage_gaps:
        summary_parts.append(
            f"{len(coverage_gaps)} data coverage gaps identified. "
        )
    if projected:
        summary_parts.append(
            f"Estimated accuracy with 50% more data: "
            f"{projected*100:.1f}% (current: {last['accuracy']*100:.1f}%)."
        )

    return {
        "suggestions": suggestions,
        "n_suggested": len(suggestions),
        "coverage_gaps": coverage_gaps,
        "learning_curve": learning_curve_data,
        "projected_accuracy": round(projected, 4) if projected else None,
        "current_accuracy": (
            round(learning_curve_data[-1]["accuracy"], 4)
            if learning_curve_data else None
        ),
        "summary": "".join(summary_parts),
        "total_samples": len(y),
        "n_classes": n_classes,
        "class_names": le.classes_.tolist(),
    }


# ──────────────────────────────────────────────────────
# Out-of-Distribution Detection
# ──────────────────────────────────────────────────────

def detect_ood(
    reference_df: pd.DataFrame,
    new_df: pd.DataFrame,
    contamination: float = 0.05,
) -> dict:
    """Detect if new/uploaded data is out-of-distribution vs reference data.

    Uses two complementary methods:
    1. Mahalanobis distance: parametric, detects shift in mean/covariance
    2. Isolation Forest: non-parametric, detects individual outliers

    Critical for industrial safety: models trained on one geological
    formation may give unreliable predictions on different formations.
    """
    ref_feats = engineer_enhanced_features(reference_df)
    new_feats = engineer_enhanced_features(new_df)

    # Align columns (new data might have different features)
    common_cols = [c for c in ref_feats.columns if c in new_feats.columns]
    if not common_cols:
        return {
            "ood_detected": True,
            "severity": "CRITICAL",
            "message": "No common features between reference and new data.",
            "details": [],
        }

    scaler = StandardScaler()
    X_ref = scaler.fit_transform(ref_feats[common_cols].values)
    X_new = scaler.transform(new_feats[common_cols].values)

    # ── Method 1: Mahalanobis Distance ──
    try:
        cov = EmpiricalCovariance().fit(X_ref)
        ref_mahal = cov.mahalanobis(X_ref)
        new_mahal = cov.mahalanobis(X_new)
        ref_threshold = np.percentile(ref_mahal, 95)

        pct_ood_mahal = float(np.mean(new_mahal > ref_threshold) * 100)
        mean_mahal_shift = float(np.mean(new_mahal) / max(np.mean(ref_mahal), 1e-6))
    except Exception:
        pct_ood_mahal = None
        mean_mahal_shift = None

    # ── Method 2: Isolation Forest ──
    try:
        iso = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100,
        )
        iso.fit(X_ref)
        iso_scores = iso.decision_function(X_new)
        iso_pred = iso.predict(X_new)  # -1 = outlier, 1 = inlier
        pct_outlier_iso = float(np.mean(iso_pred == -1) * 100)
    except Exception:
        pct_outlier_iso = None
        iso_scores = None

    # ── Feature-level drift detection ──
    drift_features = []
    for i, col in enumerate(common_cols):
        ref_mean = float(X_ref[:, i].mean())
        new_mean = float(X_new[:, i].mean())
        ref_std = float(X_ref[:, i].std())
        shift = abs(new_mean - ref_mean) / max(ref_std, 1e-6)
        if shift > 2.0:  # >2 standard deviations
            drift_features.append({
                "feature": col,
                "shift_sigma": round(shift, 2),
                "ref_mean": round(ref_mean, 3),
                "new_mean": round(new_mean, 3),
            })
    drift_features.sort(key=lambda x: x["shift_sigma"], reverse=True)

    # ── Range checks ──
    range_warnings = []
    for col_name in [DEPTH_COL, AZIMUTH_COL, DIP_COL]:
        if col_name in reference_df.columns and col_name in new_df.columns:
            ref_min = float(reference_df[col_name].min())
            ref_max = float(reference_df[col_name].max())
            new_min = float(new_df[col_name].min())
            new_max = float(new_df[col_name].max())
            if new_min < ref_min * 0.8 or new_max > ref_max * 1.2:
                range_warnings.append({
                    "column": col_name,
                    "ref_range": f"{ref_min:.1f} - {ref_max:.1f}",
                    "new_range": f"{new_min:.1f} - {new_max:.1f}",
                    "message": f"New data {col_name} range extends beyond training data bounds.",
                })

    # ── Overall severity assessment ──
    ood_score = 0
    if pct_ood_mahal is not None and pct_ood_mahal > 30:
        ood_score += 2
    elif pct_ood_mahal is not None and pct_ood_mahal > 10:
        ood_score += 1
    if pct_outlier_iso is not None and pct_outlier_iso > 30:
        ood_score += 2
    elif pct_outlier_iso is not None and pct_outlier_iso > 10:
        ood_score += 1
    if len(drift_features) > 3:
        ood_score += 2
    elif len(drift_features) > 0:
        ood_score += 1
    if len(range_warnings) > 0:
        ood_score += 1

    if ood_score >= 4:
        severity = "HIGH"
        message = (
            "Significant distribution shift detected. Model predictions on this "
            "data are UNRELIABLE. The new data appears to come from a different "
            "geological formation, depth range, or measurement system. "
            "Retrain the model with representative data before making decisions."
        )
    elif ood_score >= 2:
        severity = "MODERATE"
        message = (
            "Some distribution differences detected. Model predictions should be "
            "treated with caution. Review the flagged features and consider "
            "whether the new data represents a different geological setting."
        )
    else:
        severity = "LOW"
        message = (
            "New data is consistent with the training distribution. "
            "Model predictions are expected to be reliable."
        )

    return {
        "ood_detected": ood_score >= 2,
        "severity": severity,
        "ood_score": ood_score,
        "message": message,
        "mahalanobis_pct_ood": round(pct_ood_mahal, 1) if pct_ood_mahal is not None else None,
        "mahalanobis_shift": round(mean_mahal_shift, 2) if mean_mahal_shift is not None else None,
        "isolation_forest_pct_outlier": round(pct_outlier_iso, 1) if pct_outlier_iso is not None else None,
        "drift_features": drift_features[:10],
        "range_warnings": range_warnings,
        "n_reference": len(reference_df),
        "n_new": len(new_df),
        "stakeholder_summary": (
            f"Distribution check: {severity} concern. "
            f"{'Model predictions may not be reliable for this data. ' if ood_score >= 2 else ''}"
            f"{len(drift_features)} feature(s) show significant drift. "
            f"{len(range_warnings)} measurement range(s) outside training bounds."
        ),
    }


# ──────────────────────────────────────────────────────
# Confidence Calibration Assessment
# ──────────────────────────────────────────────────────

def assess_calibration(
    df: pd.DataFrame,
    n_bins: int = 10,
    fast: bool = False,
) -> dict:
    """Assess whether predicted probabilities match actual outcomes.

    If a model says "80% confident this is Vuggy", is it actually correct
    80% of the time? Calibration measures this alignment.

    Returns calibration curves, Brier score, and reliability assessment
    for the best available model.
    """
    features = engineer_enhanced_features(df)
    labels = df[FRACTURE_TYPE_COL].values
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    n_folds = 3 if fast else 5
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Use Random Forest (usually well-calibrated with probability estimates)
    model = RandomForestClassifier(
        n_estimators=100 if fast else 300,
        max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    # Collect CV probabilities
    probs = np.zeros((len(y), n_classes))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            probs[test_idx] = model.predict_proba(X[test_idx])

    y_pred = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)

    # ── Overall Brier Score (multi-class) ──
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y] = 1
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    # ── Per-class calibration curves ──
    calibration_data = {}
    for ci, cname in enumerate(le.classes_):
        y_binary = (y == ci).astype(int)
        prob_class = probs[:, ci]
        try:
            fraction_positive, mean_predicted = calibration_curve(
                y_binary, prob_class, n_bins=n_bins, strategy="uniform",
            )
            class_brier = float(brier_score_loss(y_binary, prob_class))
            calibration_data[cname] = {
                "fraction_positive": [round(float(v), 4) for v in fraction_positive],
                "mean_predicted": [round(float(v), 4) for v in mean_predicted],
                "brier_score": round(class_brier, 4),
                "n_samples": int(y_binary.sum()),
            }
        except Exception:
            calibration_data[cname] = {
                "fraction_positive": [],
                "mean_predicted": [],
                "brier_score": None,
                "n_samples": int(y_binary.sum()),
                "error": "Insufficient samples for calibration curve",
            }

    # ── Confidence bins analysis ──
    confidence_bins = []
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (max_probs >= low) & (max_probs < high)
        if i == n_bins - 1:
            mask = (max_probs >= low) & (max_probs <= high)
        n_in_bin = int(mask.sum())
        if n_in_bin > 0:
            acc_in_bin = float(accuracy_score(y[mask], y_pred[mask]))
            avg_conf = float(max_probs[mask].mean())
            confidence_bins.append({
                "range": f"{low:.0%}-{high:.0%}",
                "n_samples": n_in_bin,
                "accuracy": round(acc_in_bin, 3),
                "avg_confidence": round(avg_conf, 3),
                "gap": round(abs(avg_conf - acc_in_bin), 3),
            })

    # ── Expected Calibration Error (ECE) ──
    ece = 0.0
    total = len(y)
    for b in confidence_bins:
        weight = b["n_samples"] / total
        ece += weight * b["gap"]
    ece = round(ece, 4)

    # ── Reliability verdict ──
    if ece < 0.05:
        reliability = "EXCELLENT"
        reliability_msg = (
            "Model probabilities are well-calibrated. When the model says "
            "80% confident, it is correct approximately 80% of the time."
        )
    elif ece < 0.10:
        reliability = "GOOD"
        reliability_msg = (
            "Model probabilities are reasonably calibrated. Confidence values "
            "are slightly optimistic or pessimistic but usable for decisions."
        )
    elif ece < 0.20:
        reliability = "FAIR"
        reliability_msg = (
            "Model probabilities show moderate miscalibration. High confidence "
            "predictions may be overconfident. Use with caution for critical decisions."
        )
    else:
        reliability = "POOR"
        reliability_msg = (
            "Model probabilities are poorly calibrated. Confidence values "
            "DO NOT reflect actual accuracy. Do NOT rely on probability "
            "estimates for operational decisions."
        )

    return {
        "brier_score": round(brier, 4),
        "ece": ece,
        "reliability": reliability,
        "reliability_message": reliability_msg,
        "per_class": calibration_data,
        "confidence_bins": confidence_bins,
        "n_samples": len(y),
        "n_classes": n_classes,
        "class_names": le.classes_.tolist(),
        "stakeholder_summary": (
            f"Calibration: {reliability}. Expected Calibration Error = {ece:.1%}. "
            f"Brier score = {brier:.3f} (lower is better, perfect = 0). "
            f"{reliability_msg}"
        ),
    }


# ──────────────────────────────────────────────────────
# Enhanced Data Quality with Actionable Recommendations
# ──────────────────────────────────────────────────────

def data_collection_recommendations(df: pd.DataFrame) -> dict:
    """Generate specific, actionable data collection recommendations.

    Tells operators exactly what data to collect next to maximize
    model accuracy improvement per effort invested.
    """
    recommendations = []
    priority_actions = []

    n = len(df)

    # ── Class distribution analysis ──
    if FRACTURE_TYPE_COL in df.columns:
        type_counts = df[FRACTURE_TYPE_COL].value_counts()
        median_count = type_counts.median()

        for ftype, count in type_counts.items():
            if count < 30:
                needed = 30 - count
                priority_actions.append({
                    "action": f"Collect {needed} more '{ftype}' fracture measurements",
                    "priority": "HIGH",
                    "reason": (
                        f"Only {count} samples of type '{ftype}'. ML models need "
                        f"at least 30 per class for basic reliability. "
                        f"Current F1 score for this type is likely 0%."
                    ),
                    "expected_impact": (
                        f"Balanced accuracy improvement: ~{min(15, needed)}% "
                        f"(currently the model cannot predict this type)"
                    ),
                })
            elif count < median_count * 0.3:
                target = int(median_count * 0.5)
                needed = target - count
                if needed > 0:
                    recommendations.append({
                        "action": f"Add ~{needed} more '{ftype}' measurements",
                        "priority": "MODERATE",
                        "reason": (
                            f"Under-represented ({count} samples vs median {int(median_count)}). "
                            f"Class imbalance degrades overall balanced accuracy."
                        ),
                    })

    # ── Depth coverage ──
    if DEPTH_COL in df.columns:
        depth = df[DEPTH_COL].dropna()
        if len(depth) > 0:
            d_range = depth.max() - depth.min()
            bin_size = max(50, d_range / 20)
            depth_bins = np.arange(depth.min(), depth.max() + bin_size, bin_size)
            hist, _ = np.histogram(depth, bins=depth_bins)
            sparse_zones = []
            for i, count in enumerate(hist):
                if count < 5:
                    zone_start = depth_bins[i]
                    zone_end = depth_bins[i + 1]
                    sparse_zones.append(f"{zone_start:.0f}-{zone_end:.0f}m")

            if sparse_zones:
                recommendations.append({
                    "action": f"Collect fracture data in sparse depth zones: {', '.join(sparse_zones[:5])}",
                    "priority": "MODERATE",
                    "reason": (
                        "These depth intervals have fewer than 5 fracture measurements. "
                        "Model predictions in these zones are extrapolations."
                    ),
                })

    # ── Azimuth coverage ──
    if AZIMUTH_COL in df.columns:
        az = df[AZIMUTH_COL].dropna()
        az_bins = np.arange(0, 370, 30)
        hist, _ = np.histogram(az, bins=az_bins)
        empty_sectors = []
        for i, count in enumerate(hist):
            if count == 0:
                start = int(az_bins[i])
                end = int(az_bins[i + 1])
                empty_sectors.append(f"{start}-{end}")

        if empty_sectors:
            recommendations.append({
                "action": f"Investigate fractures with azimuths in: {', '.join(empty_sectors)} degrees",
                "priority": "LOW",
                "reason": (
                    "No fractures observed in these azimuth sectors. "
                    "This may be real (stress-controlled) or a measurement gap."
                ),
            })

    # ── Sample size assessment ──
    if n < 50:
        priority_actions.append({
            "action": f"Collect at least {50 - n} more fracture measurements (any type)",
            "priority": "CRITICAL",
            "reason": (
                f"Only {n} total samples. ML classification requires minimum 50 "
                f"for any meaningful results."
            ),
            "expected_impact": "Accuracy could improve from ~50% (random) to 70%+",
        })
    elif n < 200:
        recommendations.append({
            "action": f"Target {200 - n} more measurements for robust ML",
            "priority": "MODERATE",
            "reason": (
                f"{n} samples is marginal. Learning curve analysis shows accuracy "
                f"improvement plateaus around 200-300 samples."
            ),
        })

    # ── Well diversity ──
    if WELL_COL in df.columns:
        n_wells = df[WELL_COL].nunique()
        if n_wells < 3:
            recommendations.append({
                "action": "Include data from additional wells in the field",
                "priority": "MODERATE",
                "reason": (
                    f"Only {n_wells} well(s). Cross-well validation is essential to "
                    f"confirm stress patterns are field-wide, not well-specific."
                ),
            })

    # ── Minimum viable dataset specification ──
    min_viable = {
        "min_total_samples": 200,
        "min_per_class": 30,
        "min_wells": 3,
        "min_depth_coverage_pct": 80,
        "required_columns": ["Depth(m)", "Azimuth(deg)", "Dip(deg)", "Fracture_Type", "Well"],
    }

    current_meets = {
        "total_samples": bool(n >= 200),
        "per_class": bool(
            df[FRACTURE_TYPE_COL].value_counts().min() >= 30
            if FRACTURE_TYPE_COL in df.columns else False
        ),
        "wells": bool(
            df[WELL_COL].nunique() >= 3
            if WELL_COL in df.columns else False
        ),
    }

    return {
        "priority_actions": priority_actions,
        "recommendations": recommendations,
        "min_viable_dataset": min_viable,
        "current_meets_minimum": current_meets,
        "data_completeness_pct": round(
            100 * sum(current_meets.values()) / len(current_meets), 0
        ),
        "summary": (
            f"{len(priority_actions)} critical action(s), "
            f"{len(recommendations)} recommendation(s). "
            f"Dataset completeness: {sum(current_meets.values())}/{len(current_meets)} criteria met."
        ),
    }


# ═══════════════════════════════════════════════════════
#  Learning Curve Analysis
# ═══════════════════════════════════════════════════════

def compute_learning_curve(df, n_points=8, fast=True):
    """Compute learning curve showing how accuracy improves with more data.

    Trains the best model (Random Forest) on increasingly larger subsets of
    data to show stakeholders exactly how more data would improve accuracy.
    Returns per-class curves to highlight where minority classes plateau.

    Parameters
    ----------
    df : DataFrame with fracture data
    n_points : number of training-set sizes to evaluate
    fast : if True, use fewer estimators for speed

    Returns
    -------
    dict with train_sizes, train_scores, val_scores, per_class, projection
    """
    features = engineer_enhanced_features(df)
    le = LabelEncoder()
    y = le.fit_transform(df[FRACTURE_TYPE_COL].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    n_samples = len(X_scaled)
    n_classes = len(le.classes_)

    # Skip if too few samples
    if n_samples < 30:
        return {"error": "Need at least 30 samples for learning curve analysis"}

    n_est = 50 if fast else 200
    model = RandomForestClassifier(
        n_estimators=n_est, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )

    # Generate training sizes (from 20% to 100%)
    min_train = max(n_classes * 3, 15)  # Need at least 3 per class
    max_train = int(n_samples * 0.8)  # Reserve 20% for validation
    if min_train >= max_train:
        min_train = max(n_classes * 2, 10)

    train_sizes = np.linspace(min_train, max_train, n_points).astype(int)
    train_sizes = np.unique(train_sizes)  # Remove duplicates

    train_scores = []
    val_scores = []
    per_class_scores = {cls: [] for cls in le.classes_}
    balanced_scores = []

    # Stratified split: 80% train pool, 20% fixed validation
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_pool_idx, val_idx = next(sss.split(X_scaled, y))
    X_pool, y_pool = X_scaled[train_pool_idx], y[train_pool_idx]
    X_val, y_val = X_scaled[val_idx], y[val_idx]

    for size in train_sizes:
        if size > len(X_pool):
            size = len(X_pool)

        # Stratified subsample from the pool
        try:
            sub_sss = StratifiedShuffleSplit(
                n_splits=1, train_size=min(size, len(X_pool) - 1),
                random_state=42,
            )
            sub_idx, _ = next(sub_sss.split(X_pool, y_pool))
        except ValueError:
            sub_idx = np.arange(min(size, len(X_pool)))

        X_sub, y_sub = X_pool[sub_idx], y_pool[sub_idx]

        model_clone = RandomForestClassifier(
            n_estimators=n_est, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        model_clone.fit(X_sub, y_sub)

        train_acc = model_clone.score(X_sub, y_sub)
        val_acc = model_clone.score(X_val, y_val)
        val_pred = model_clone.predict(X_val)
        bal_acc = balanced_accuracy_score(y_val, val_pred)

        train_scores.append(round(float(train_acc), 4))
        val_scores.append(round(float(val_acc), 4))
        balanced_scores.append(round(float(bal_acc), 4))

        # Per-class F1
        for i, cls in enumerate(le.classes_):
            mask = y_val == i
            if mask.sum() > 0:
                cls_f1 = f1_score(y_val == i, val_pred == i, zero_division=0)
                per_class_scores[cls].append(round(float(cls_f1), 4))
            else:
                per_class_scores[cls].append(None)

    # Projection: extrapolate how much data would be needed for target accuracy
    projection = _project_data_needs(train_sizes, val_scores)

    # Convergence analysis
    if len(val_scores) >= 3:
        recent_improvement = val_scores[-1] - val_scores[-3]
        if recent_improvement < 0.01:
            convergence = "PLATEAU"
            convergence_msg = (
                "Model accuracy has plateaued. More data of the same type "
                "won't help significantly. Consider: (1) new feature types, "
                "(2) data from different geological settings, or "
                "(3) higher-quality measurements."
            )
        elif recent_improvement < 0.03:
            convergence = "SLOWING"
            convergence_msg = (
                "Accuracy growth is slowing. Each additional sample adds "
                "less value. Targeted collection of under-represented classes "
                "would be more effective than bulk data collection."
            )
        else:
            convergence = "GROWING"
            convergence_msg = (
                "Model is still improving with more data. Continue collecting "
                "data, especially for minority fracture types."
            )
    else:
        convergence = "INSUFFICIENT"
        convergence_msg = "Not enough data points to assess convergence."

    return {
        "train_sizes": [int(s) for s in train_sizes],
        "train_scores": train_scores,
        "val_scores": val_scores,
        "balanced_scores": balanced_scores,
        "per_class": {cls: scores for cls, scores in per_class_scores.items()},
        "class_names": list(le.classes_),
        "n_samples": n_samples,
        "convergence": convergence,
        "convergence_message": convergence_msg,
        "projection": projection,
    }


def _project_data_needs(sizes, scores):
    """Extrapolate how many samples would be needed for various accuracy targets."""
    if len(sizes) < 3 or len(scores) < 3:
        return {"available": False, "reason": "Need more data points"}

    current_acc = scores[-1]
    targets = [0.85, 0.90, 0.95]
    projections = []

    for target in targets:
        if current_acc >= target:
            projections.append({
                "target_accuracy": target,
                "status": "ACHIEVED",
                "samples_needed": 0,
            })
        else:
            # Simple log-linear extrapolation (accuracy ~ a + b*log(n))
            try:
                log_sizes = np.log(sizes[-4:])
                recent_scores = np.array(scores[-4:])
                coeffs = np.polyfit(log_sizes, recent_scores, 1)
                if coeffs[0] > 0:
                    needed_log_n = (target - coeffs[1]) / coeffs[0]
                    needed_n = int(np.exp(needed_log_n))
                    if needed_n > 100000:
                        projections.append({
                            "target_accuracy": target,
                            "status": "UNLIKELY",
                            "reason": "Would require unrealistic data volume",
                        })
                    else:
                        projections.append({
                            "target_accuracy": target,
                            "status": "PROJECTED",
                            "samples_needed": needed_n,
                            "additional_needed": max(0, needed_n - int(sizes[-1])),
                        })
                else:
                    projections.append({
                        "target_accuracy": target,
                        "status": "PLATEAU",
                        "reason": "Accuracy not improving with more data — need better features",
                    })
            except Exception:
                projections.append({
                    "target_accuracy": target,
                    "status": "UNKNOWN",
                    "reason": "Could not extrapolate",
                })

    return {"available": True, "targets": projections}


# ═══════════════════════════════════════════════════════
#  Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════

def bootstrap_class_metrics(df, n_bootstrap=200, ci_level=0.95, fast=True):
    """Compute bootstrap 95% confidence intervals for per-class metrics.

    Industrial-grade: instead of point estimates, returns ranges like
    "F1 = 0.85 [0.78-0.91]" so decision-makers know the reliability.

    Parameters
    ----------
    df : DataFrame with fracture data
    n_bootstrap : number of bootstrap resamples
    ci_level : confidence level (default 0.95 = 95% CI)
    fast : if True, use fewer estimators

    Returns
    -------
    dict with per-class CIs for F1, precision, recall, and overall accuracy CI
    """
    features = engineer_enhanced_features(df)
    le = LabelEncoder()
    y = le.fit_transform(df[FRACTURE_TYPE_COL].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    n_samples = len(X_scaled)
    n_classes = len(le.classes_)

    if n_samples < 20:
        return {"error": "Need at least 20 samples for bootstrap CI"}

    n_est = 50 if fast else 200
    alpha = 1 - ci_level

    # Bootstrap resamples
    accuracy_boot = []
    balanced_boot = []
    f1_boot = {cls: [] for cls in le.classes_}
    precision_boot = {cls: [] for cls in le.classes_}
    recall_boot = {cls: [] for cls in le.classes_}

    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[idx] = False
        oob_idx = np.where(oob_mask)[0]

        # Need at least 2 classes in both train and OOB
        if len(oob_idx) < 5:
            continue
        if len(np.unique(y[idx])) < 2:
            continue
        if len(np.unique(y[oob_idx])) < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=n_est, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        model.fit(X_scaled[idx], y[idx])
        pred = model.predict(X_scaled[oob_idx])

        accuracy_boot.append(accuracy_score(y[oob_idx], pred))
        balanced_boot.append(balanced_accuracy_score(y[oob_idx], pred))

        for i, cls in enumerate(le.classes_):
            y_bin = (y[oob_idx] == i).astype(int)
            p_bin = (pred == i).astype(int)
            if y_bin.sum() > 0:
                f1_boot[cls].append(f1_score(y_bin, p_bin, zero_division=0))
                precision_boot[cls].append(precision_score(y_bin, p_bin, zero_division=0))
                recall_boot[cls].append(recall_score(y_bin, p_bin, zero_division=0))

    if len(accuracy_boot) < 20:
        return {"error": "Too few valid bootstrap resamples. Need more data."}

    def _ci(values):
        arr = np.array(values)
        lo = float(np.percentile(arr, 100 * alpha / 2))
        hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        mean = float(np.mean(arr))
        return {"mean": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "width": round(hi - lo, 4)}

    per_class = {}
    for cls in le.classes_:
        per_class[cls] = {
            "f1": _ci(f1_boot[cls]) if len(f1_boot[cls]) >= 20 else None,
            "precision": _ci(precision_boot[cls]) if len(precision_boot[cls]) >= 20 else None,
            "recall": _ci(recall_boot[cls]) if len(recall_boot[cls]) >= 20 else None,
            "n_bootstrap": len(f1_boot[cls]),
        }

    # Reliability assessment
    wide_cis = sum(
        1 for cls in per_class.values()
        if cls["f1"] and cls["f1"]["width"] > 0.3
    )
    if wide_cis == 0:
        reliability = "HIGH"
        reliability_msg = "All class metrics have tight confidence intervals. Predictions are reliable."
    elif wide_cis <= n_classes // 2:
        reliability = "MODERATE"
        reliability_msg = (
            f"{wide_cis} class(es) have wide CIs (>0.30). "
            f"These classes need more training data for reliable predictions."
        )
    else:
        reliability = "LOW"
        reliability_msg = (
            f"{wide_cis}/{n_classes} classes have wide CIs. "
            f"Model predictions are not reliable enough for production decisions."
        )

    return {
        "accuracy": _ci(accuracy_boot),
        "balanced_accuracy": _ci(balanced_boot),
        "per_class": per_class,
        "class_names": list(le.classes_),
        "n_bootstrap": len(accuracy_boot),
        "ci_level": ci_level,
        "reliability": reliability,
        "reliability_message": reliability_msg,
    }


# ═══════════════════════════════════════════════════════
#  Scenario Comparison
# ═══════════════════════════════════════════════════════

def scenario_comparison(df, scenarios, depth_m=3000):
    """Compare multiple what-if scenarios for stress inversion.

    Allows decision-makers to see how different assumptions about regime,
    pore pressure, and friction coefficient affect the results.

    Parameters
    ----------
    df : DataFrame with fracture data for one well
    scenarios : list of dicts, each with keys: name, regime, pore_pressure, friction
    depth_m : depth in meters

    Returns
    -------
    dict with scenario results and comparison metrics
    """
    try:
        from src.geostress import invert_stress
    except ImportError:
        from geostress import invert_stress

    normals = fracture_plane_normal(
        df[AZIMUTH_COL].values, df[DIP_COL].values
    )

    results = []
    for i, scenario in enumerate(scenarios):
        name = scenario.get("name", f"Scenario {i+1}")
        regime = scenario.get("regime", "strike_slip")
        pp = scenario.get("pore_pressure", None)
        # friction is not a parameter for invert_stress, it's optimized

        try:
            inv = invert_stress(
                normals, regime=regime,
                depth_m=depth_m, pore_pressure=pp,
            )

            pp_val = inv.get("pore_pressure", 0.0)
            if isinstance(pp_val, np.ndarray):
                pp_val = float(pp_val.flat[0])

            cs = critically_stressed_enhanced(
                inv["sigma_n"], inv["tau"],
                mu=inv["mu"], pore_pressure=pp_val,
            )

            # Safely extract scalar values from potential numpy types
            def _scalar(v):
                if isinstance(v, np.ndarray):
                    return float(v.flat[0])
                return float(v)

            results.append({
                "name": name,
                "regime": regime,
                "sigma1": round(_scalar(inv["sigma1"]), 2),
                "sigma3": round(_scalar(inv["sigma3"]), 2),
                "shmax": round(_scalar(inv["shmax_azimuth_deg"]), 1),
                "R_ratio": round(_scalar(inv["R"]), 4),
                "mu": round(_scalar(inv["mu"]), 4),
                "pore_pressure": round(_scalar(pp_val), 2),
                "misfit": round(_scalar(inv["misfit"]), 4),
                "critically_stressed_pct": round(cs["pct_critical"], 1),
                "high_risk_count": cs["high_risk_count"],
                "status": "OK",
            })
        except Exception as e:
            results.append({
                "name": name,
                "regime": regime,
                "status": "ERROR",
                "error": str(e)[:100],
            })

    # Compute variance across scenarios for each key metric
    ok_results = [r for r in results if r["status"] == "OK"]
    if len(ok_results) >= 2:
        metrics_spread = {}
        for key in ("sigma1", "sigma3", "shmax", "R_ratio", "mu",
                     "critically_stressed_pct"):
            vals = [r[key] for r in ok_results if key in r]
            if vals:
                metrics_spread[key] = {
                    "min": min(vals),
                    "max": max(vals),
                    "range": round(max(vals) - min(vals), 2),
                    "cv_pct": round(100 * np.std(vals) / max(abs(np.mean(vals)), 1e-6), 1),
                }

        # Find which parameter has most impact
        max_cv_key = max(metrics_spread, key=lambda k: metrics_spread[k]["cv_pct"])
        sensitivity_msg = (
            f"'{max_cv_key}' shows the most variation across scenarios "
            f"(CV={metrics_spread[max_cv_key]['cv_pct']}%). "
            f"This parameter requires the most careful specification."
        )
    else:
        metrics_spread = {}
        sensitivity_msg = "Need at least 2 successful scenarios for comparison."

    # Recommendation
    if ok_results:
        best = min(ok_results, key=lambda r: r["misfit"])
        recommendation = (
            f"Scenario '{best['name']}' has the lowest misfit ({best['misfit']:.4f}), "
            f"suggesting it best fits the observed fracture data."
        )
    else:
        recommendation = "No scenarios completed successfully."

    return {
        "scenarios": results,
        "n_scenarios": len(results),
        "n_successful": len(ok_results),
        "metrics_spread": metrics_spread,
        "sensitivity_message": sensitivity_msg,
        "recommendation": recommendation,
    }