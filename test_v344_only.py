"""Focused test for v3.44.0 endpoints: PCA + Fracture Intensity + CV Stability + Geomech Summary + Correlation Network."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8131"


def api(method, path, body=None, timeout=300):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {label}" + (f" ({detail})" if detail else ""))
        passed += 1
    else:
        print(f"  FAIL: {label}" + (f" ({detail})" if detail else ""))
        failed += 1


# ── [155] PCA Analysis ────────────────────────────────
print("\n[155] PCA Analysis")
pc = api("POST", "/api/analysis/pca", {"source": "demo", "well": "3P", "n_components": 5})
check("Status 200", pc is not None)
check("Has well", pc.get("well") == "3P")
check("Has n_features", isinstance(pc.get("n_features"), int) and pc["n_features"] >= 5)
check("Has n_components", isinstance(pc.get("n_components"), int) and pc["n_components"] >= 2)
check("Has variance_explained", isinstance(pc.get("variance_explained"), list) and len(pc["variance_explained"]) >= 2)
check("Has cumulative_variance", isinstance(pc.get("cumulative_variance"), list))
check("Has n_for_80pct", isinstance(pc.get("n_for_80pct"), int))
check("Has n_for_95pct", isinstance(pc.get("n_for_95pct"), int))
check("Has components", isinstance(pc.get("components"), list) and len(pc["components"]) >= 2)
c0 = pc["components"][0]
check("Component has component", isinstance(c0.get("component"), int))
check("Component has variance_pct", isinstance(c0.get("variance_pct"), (int, float)))
check("Component has cumulative_pct", isinstance(c0.get("cumulative_pct"), (int, float)))
check("Component has top_features", isinstance(c0.get("top_features"), list) and len(c0["top_features"]) >= 3)
tf0 = c0["top_features"][0]
check("TopFeat has feature", isinstance(tf0.get("feature"), str))
check("TopFeat has loading", isinstance(tf0.get("loading"), (int, float)))
check("Has recommendations", isinstance(pc.get("recommendations"), list) and len(pc["recommendations"]) > 0)
check("Has plot", isinstance(pc.get("plot"), str) and len(pc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(pc.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(pc["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(pc.get("elapsed_s"), (int, float)))

# 6P test
pc2 = api("POST", "/api/analysis/pca", {"source": "demo", "well": "6P"})
check("6P works", pc2 is not None and pc2.get("well") == "6P")


# ── [156] Fracture Intensity Profile ──────────────────
print("\n[156] Fracture Intensity Profile")
fi = api("POST", "/api/analysis/fracture-intensity", {"source": "demo", "well": "3P", "bin_size_m": 10})
check("Status 200", fi is not None)
check("Has well", fi.get("well") == "3P")
check("Has n_fractures", isinstance(fi.get("n_fractures"), int) and fi["n_fractures"] > 0)
check("Has depth_range_m", isinstance(fi.get("depth_range_m"), list) and len(fi["depth_range_m"]) == 2)
check("Has bin_size_m", isinstance(fi.get("bin_size_m"), (int, float)))
check("Has n_intervals", isinstance(fi.get("n_intervals"), int) and fi["n_intervals"] >= 2)
check("Has overall_P10", isinstance(fi.get("overall_P10"), (int, float)))
check("Has max_P10", isinstance(fi.get("max_P10"), (int, float)))
check("Has min_P10", isinstance(fi.get("min_P10"), (int, float)))
check("Has intensity_class", fi.get("intensity_class") in ("VERY_HIGH", "HIGH", "MODERATE", "LOW"))
check("Has intervals", isinstance(fi.get("intervals"), list) and len(fi["intervals"]) >= 2)
iv0 = fi["intervals"][0]
check("Interval has depth_from", isinstance(iv0.get("depth_from"), (int, float)))
check("Interval has depth_to", isinstance(iv0.get("depth_to"), (int, float)))
check("Interval has n_fractures", isinstance(iv0.get("n_fractures"), int))
check("Interval has P10", isinstance(iv0.get("P10"), (int, float)))
check("Has recommendations", isinstance(fi.get("recommendations"), list) and len(fi["recommendations"]) > 0)
check("Has plot", isinstance(fi.get("plot"), str) and len(fi["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fi.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(fi["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(fi.get("elapsed_s"), (int, float)))

# 6P test
try:
    fi2 = api("POST", "/api/analysis/fracture-intensity", {"source": "demo", "well": "6P"})
    check("6P works or graceful error", fi2 is not None)
except Exception as e:
    check("6P works or graceful error", "400" in str(e) or "404" in str(e), f"Expected: {e}")


# ── [157] CV Stability ────────────────────────────────
print("\n[157] CV Stability")
cv = api("POST", "/api/analysis/cv-stability", {"source": "demo", "well": "3P", "n_folds": 5})
check("Status 200", cv is not None)
check("Has well", cv.get("well") == "3P")
check("Has n_samples", isinstance(cv.get("n_samples"), int) and cv["n_samples"] > 0)
check("Has n_folds", isinstance(cv.get("n_folds"), int) and cv["n_folds"] >= 2)
check("Has n_models", isinstance(cv.get("n_models"), int) and cv["n_models"] >= 3)
check("Has models", isinstance(cv.get("models"), list) and len(cv["models"]) >= 3)
m0 = cv["models"][0]
check("Model has model", isinstance(m0.get("model"), str))
check("Model has mean_accuracy", isinstance(m0.get("mean_accuracy"), (int, float)))
check("Model has std_accuracy", isinstance(m0.get("std_accuracy"), (int, float)))
check("Model has cv_ratio", isinstance(m0.get("cv_ratio"), (int, float)))
check("Model has train_accuracy", isinstance(m0.get("train_accuracy"), (int, float)))
check("Model has overfit_gap", isinstance(m0.get("overfit_gap"), (int, float)))
check("Model has stability", m0.get("stability") in ("STABLE", "MODERATE", "UNSTABLE"))
check("Model has overfitting", m0.get("overfitting") in ("YES", "MILD", "NO"))
check("Model has fold_scores", isinstance(m0.get("fold_scores"), list) and len(m0["fold_scores"]) >= 2)
check("Has best_model", isinstance(cv.get("best_model"), str))
check("Has best_accuracy", isinstance(cv.get("best_accuracy"), (int, float)))
check("Has most_stable_model", isinstance(cv.get("most_stable_model"), str))
check("Has recommendations", isinstance(cv.get("recommendations"), list) and len(cv["recommendations"]) > 0)
check("Has plot", isinstance(cv.get("plot"), str) and len(cv["plot"]) > 100)
check("Has stakeholder_brief", isinstance(cv.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(cv["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(cv.get("elapsed_s"), (int, float)))

# 6P test
cv2 = api("POST", "/api/analysis/cv-stability", {"source": "demo", "well": "6P"})
check("6P works", cv2 is not None and cv2.get("well") == "6P")


# ── [158] Geomechanical Summary ───────────────────────
print("\n[158] Geomechanical Summary")
gs = api("POST", "/api/analysis/geomech-summary", {"source": "demo", "well": "3P"})
check("Status 200", gs is not None)
check("Has well", gs.get("well") == "3P")
check("Has n_fractures", isinstance(gs.get("n_fractures"), int) and gs["n_fractures"] > 0)
check("Has n_types", isinstance(gs.get("n_types"), int) and gs["n_types"] >= 2)
check("Has fracture_types", isinstance(gs.get("fracture_types"), dict))
check("Has mean_azimuth", isinstance(gs.get("mean_azimuth"), (int, float)))
check("Has mean_dip", isinstance(gs.get("mean_dip"), (int, float)))
check("Has resultant_length", isinstance(gs.get("resultant_length"), (int, float)))
check("Has depth_range_m", isinstance(gs.get("depth_range_m"), list))
check("Has mean_depth_m", isinstance(gs.get("mean_depth_m"), (int, float)))
check("Has depth_completeness_pct", isinstance(gs.get("depth_completeness_pct"), (int, float)))
check("Has estimated_Sv_MPa", isinstance(gs.get("estimated_Sv_MPa"), (int, float)))
check("Has estimated_Pp_MPa", isinstance(gs.get("estimated_Pp_MPa"), (int, float)))
check("Has quality_score", isinstance(gs.get("quality_score"), int))
check("Has quality_grade", gs.get("quality_grade") in ("A", "B", "C", "D"))
check("Has recommendations", isinstance(gs.get("recommendations"), list) and len(gs["recommendations"]) > 0)
check("Has plot", isinstance(gs.get("plot"), str) and len(gs["plot"]) > 100)
check("Has stakeholder_brief", isinstance(gs.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(gs["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(gs.get("elapsed_s"), (int, float)))

# 6P test
gs2 = api("POST", "/api/analysis/geomech-summary", {"source": "demo", "well": "6P"})
check("6P works", gs2 is not None and gs2.get("well") == "6P")


# ── [159] Correlation Network ─────────────────────────
print("\n[159] Correlation Network")
cn = api("POST", "/api/analysis/correlation-network", {"source": "demo", "well": "3P", "threshold": 0.5})
check("Status 200", cn is not None)
check("Has well", cn.get("well") == "3P")
check("Has n_features", isinstance(cn.get("n_features"), int) and cn["n_features"] >= 5)
check("Has mean_abs_correlation", isinstance(cn.get("mean_abs_correlation"), (int, float)))
check("Has n_strong_links", isinstance(cn.get("n_strong_links"), int))
check("Has n_redundant_features", isinstance(cn.get("n_redundant_features"), int))
check("Has redundant_features", isinstance(cn.get("redundant_features"), list))
check("Has strong_links", isinstance(cn.get("strong_links"), list))
if cn["strong_links"]:
    sl0 = cn["strong_links"][0]
    check("Link has feature_a", isinstance(sl0.get("feature_a"), str))
    check("Link has feature_b", isinstance(sl0.get("feature_b"), str))
    check("Link has correlation", isinstance(sl0.get("correlation"), (int, float)))
    check("Link has strength", sl0.get("strength") in ("STRONG_POSITIVE", "STRONG_NEGATIVE"))
else:
    check("Link has feature_a", True, "no strong links")
    check("Link has feature_b", True, "no strong links")
    check("Link has correlation", True, "no strong links")
    check("Link has strength", True, "no strong links")
check("Has top_pairs", isinstance(cn.get("top_pairs"), list) and len(cn["top_pairs"]) > 0)
tp0 = cn["top_pairs"][0]
check("Pair has feature_a", isinstance(tp0.get("feature_a"), str))
check("Pair has abs_correlation", isinstance(tp0.get("abs_correlation"), (int, float)))
check("Has recommendations", isinstance(cn.get("recommendations"), list) and len(cn["recommendations"]) > 0)
check("Has plot", isinstance(cn.get("plot"), str) and len(cn["plot"]) > 100)
check("Has stakeholder_brief", isinstance(cn.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(cn["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(cn.get("elapsed_s"), (int, float)))

# 6P test
cn2 = api("POST", "/api/analysis/correlation-network", {"source": "demo", "well": "6P"})
check("6P works", cn2 is not None and cn2.get("well") == "6P")


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.44.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
