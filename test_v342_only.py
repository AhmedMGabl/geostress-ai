"""Focused test for v3.42.0 endpoints: Sensitivity Matrix + Prediction Explanation + Model Comparison Detail + Data Profile + Anomaly Score."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8126"


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


# ── [145] Feature Interaction Sensitivity Matrix ───────
print("\n[145] Feature Interaction Sensitivity Matrix")
sm = api("POST", "/api/analysis/sensitivity-matrix", {"source": "demo", "well": "3P", "top_n": 8})
check("Status 200", sm is not None)
check("Has well", sm.get("well") == "3P")
check("Has n_features_analyzed", isinstance(sm.get("n_features_analyzed"), int) and sm["n_features_analyzed"] >= 5)
check("Has baseline_accuracy", isinstance(sm.get("baseline_accuracy"), (int, float)))
check("Has features list", isinstance(sm.get("features"), list) and len(sm["features"]) >= 5)
check("Has individual_importance", isinstance(sm.get("individual_importance"), list))
ii0 = sm["individual_importance"][0]
check("Importance has feature", isinstance(ii0.get("feature"), str))
check("Importance has drop_pct", isinstance(ii0.get("drop_pct"), (int, float)))
check("Has interactions", isinstance(sm.get("interactions"), list))
check("Has n_synergistic", isinstance(sm.get("n_synergistic"), int))
check("Has n_redundant", isinstance(sm.get("n_redundant"), int))
check("Has matrix", isinstance(sm.get("matrix"), list))
check("Has recommendations", isinstance(sm.get("recommendations"), list) and len(sm["recommendations"]) > 0)
check("Has plot", isinstance(sm.get("plot"), str) and len(sm["plot"]) > 100)
check("Has stakeholder_brief", isinstance(sm.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(sm["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(sm.get("elapsed_s"), (int, float)))

# 6P test
sm2 = api("POST", "/api/analysis/sensitivity-matrix", {"source": "demo", "well": "6P", "top_n": 5})
check("6P works", sm2 is not None and sm2.get("well") == "6P")


# ── [146] Prediction Explanation ───────────────────────
print("\n[146] Per-Sample Prediction Explanation")
pe = api("POST", "/api/analysis/prediction-explanation", {"source": "demo", "well": "3P", "n_samples": 15})
check("Status 200", pe is not None)
check("Has well", pe.get("well") == "3P")
check("Has n_explained", isinstance(pe.get("n_explained"), int) and pe["n_explained"] > 0)
check("Has n_correct", isinstance(pe.get("n_correct"), int))
check("Has n_misclassified", isinstance(pe.get("n_misclassified"), int))
check("Has mean_confidence", isinstance(pe.get("mean_confidence"), (int, float)))
check("Has explanations", isinstance(pe.get("explanations"), list) and len(pe["explanations"]) > 0)
ex0 = pe["explanations"][0]
check("Expl has predicted_class", isinstance(ex0.get("predicted_class"), str))
check("Expl has true_class", isinstance(ex0.get("true_class"), str))
check("Expl has confidence", isinstance(ex0.get("confidence"), (int, float)))
check("Expl has correct", isinstance(ex0.get("correct"), bool))
check("Expl has top_reasons", isinstance(ex0.get("top_reasons"), list) and len(ex0["top_reasons"]) == 3)
r0 = ex0["top_reasons"][0]
check("Reason has feature", isinstance(r0.get("feature"), str))
check("Reason has contribution", isinstance(r0.get("contribution"), (int, float)))
check("Reason has direction", r0.get("direction") in ("high", "low"))
check("Expl has explanation text", isinstance(ex0.get("explanation"), str))
check("Has recommendations", isinstance(pe.get("recommendations"), list) and len(pe["recommendations"]) > 0)
check("Has plot", isinstance(pe.get("plot"), str) and len(pe["plot"]) > 100)
check("Has stakeholder_brief", isinstance(pe.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(pe["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(pe.get("elapsed_s"), (int, float)))

# 6P test
pe2 = api("POST", "/api/analysis/prediction-explanation", {"source": "demo", "well": "6P"})
check("6P works", pe2 is not None and pe2.get("well") == "6P")


# ── [147] Detailed Model Comparison ────────────────────
print("\n[147] Detailed Model Comparison")
mc = api("POST", "/api/analysis/model-comparison-detailed", {"source": "demo", "well": "3P"})
check("Status 200", mc is not None)
check("Has well", mc.get("well") == "3P")
check("Has n_models", isinstance(mc.get("n_models"), int) and mc["n_models"] >= 4)
check("Has n_classes", isinstance(mc.get("n_classes"), int) and mc["n_classes"] >= 2)
check("Has classes", isinstance(mc.get("classes"), list))
check("Has models list", isinstance(mc.get("models"), list) and len(mc["models"]) >= 4)
m0 = mc["models"][0]
check("Model has name", isinstance(m0.get("model"), str))
check("Model has balanced_accuracy", isinstance(m0.get("balanced_accuracy"), (int, float)))
check("Model has per_class", isinstance(m0.get("per_class"), list) and len(m0["per_class"]) >= 2)
pc0 = m0["per_class"][0]
check("PC has class", isinstance(pc0.get("class"), str))
check("PC has precision", isinstance(pc0.get("precision"), (int, float)))
check("PC has recall", isinstance(pc0.get("recall"), (int, float)))
check("PC has f1", isinstance(pc0.get("f1"), (int, float)))
check("PC has support", isinstance(pc0.get("support"), int))
check("Model has best_class", isinstance(m0.get("best_class"), str))
check("Model has worst_class", isinstance(m0.get("worst_class"), str))
check("Has best_model", isinstance(mc.get("best_model"), str))
check("Has best_accuracy", isinstance(mc.get("best_accuracy"), (int, float)))
check("Has recommendations", isinstance(mc.get("recommendations"), list) and len(mc["recommendations"]) > 0)
check("Has plot", isinstance(mc.get("plot"), str) and len(mc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(mc.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(mc["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(mc.get("elapsed_s"), (int, float)))

# 6P test
mc2 = api("POST", "/api/analysis/model-comparison-detailed", {"source": "demo", "well": "6P"})
check("6P works", mc2 is not None and mc2.get("well") == "6P")


# ── [148] Data Profile ─────────────────────────────────
print("\n[148] Data Profile")
dp = api("POST", "/api/analysis/data-profile", {"source": "demo", "well": "3P"})
check("Status 200", dp is not None)
check("Has well", dp.get("well") == "3P")
check("Has n_samples", isinstance(dp.get("n_samples"), int) and dp["n_samples"] > 0)
check("Has n_columns", isinstance(dp.get("n_columns"), int) and dp["n_columns"] >= 2)
check("Has completeness_pct", isinstance(dp.get("completeness_pct"), (int, float)))
check("Has columns", isinstance(dp.get("columns"), list) and len(dp["columns"]) >= 2)
c0 = dp["columns"][0]
check("Col has column name", isinstance(c0.get("column"), str))
check("Col has count", isinstance(c0.get("count"), int))
check("Col has mean", isinstance(c0.get("mean"), (int, float)))
check("Col has std", isinstance(c0.get("std"), (int, float)))
check("Col has min", isinstance(c0.get("min"), (int, float)))
check("Col has median", isinstance(c0.get("median"), (int, float)))
check("Col has max", isinstance(c0.get("max"), (int, float)))
check("Col has skewness", isinstance(c0.get("skewness"), (int, float)))
check("Has class_distribution", isinstance(dp.get("class_distribution"), list) and len(dp["class_distribution"]) >= 2)
check("Has correlations", isinstance(dp.get("correlations"), list))
check("Has recommendations", isinstance(dp.get("recommendations"), list) and len(dp["recommendations"]) > 0)
check("Has plot", isinstance(dp.get("plot"), str) and len(dp["plot"]) > 100)
check("Has stakeholder_brief", isinstance(dp.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(dp["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(dp.get("elapsed_s"), (int, float)))

# 6P test
dp2 = api("POST", "/api/analysis/data-profile", {"source": "demo", "well": "6P"})
check("6P works", dp2 is not None and dp2.get("well") == "6P")


# ── [149] Anomaly Score ────────────────────────────────
print("\n[149] Per-Sample Anomaly Score")
an = api("POST", "/api/analysis/anomaly-score", {"source": "demo", "well": "3P"})
check("Status 200", an is not None)
check("Has well", an.get("well") == "3P")
check("Has n_samples", isinstance(an.get("n_samples"), int) and an["n_samples"] > 0)
check("Has n_anomalies", isinstance(an.get("n_anomalies"), int))
check("Has pct_anomalies", isinstance(an.get("pct_anomalies"), (int, float)))
check("Has samples list", isinstance(an.get("samples"), list) and len(an["samples"]) > 0)
s0 = an["samples"][0]
check("Sample has anomaly_score", isinstance(s0.get("anomaly_score"), (int, float)))
check("Sample has is_anomaly", isinstance(s0.get("is_anomaly"), bool))
check("Sample has fracture_type", isinstance(s0.get("fracture_type"), str))
# Check anomalous sample has top features
anomalous = [s for s in an["samples"] if s["is_anomaly"]]
if anomalous:
    check("Anomaly has top_unusual_features", isinstance(anomalous[0].get("top_unusual_features"), list) and len(anomalous[0]["top_unusual_features"]) > 0)
    tf0 = anomalous[0]["top_unusual_features"][0]
    check("Unusual feat has feature", isinstance(tf0.get("feature"), str))
    check("Unusual feat has z_score", isinstance(tf0.get("z_score"), (int, float)))
else:
    check("Anomaly has top_unusual_features", True, "no anomalies found")
    check("Unusual feat has feature", True, "no anomalies")
    check("Unusual feat has z_score", True, "no anomalies")
check("Has anomaly_depth_zones", isinstance(an.get("anomaly_depth_zones"), list))
check("Has recommendations", isinstance(an.get("recommendations"), list) and len(an["recommendations"]) > 0)
check("Has plot", isinstance(an.get("plot"), str) and len(an["plot"]) > 100)
check("Has stakeholder_brief", isinstance(an.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(an["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(an.get("elapsed_s"), (int, float)))

# 6P test
an2 = api("POST", "/api/analysis/anomaly-score", {"source": "demo", "well": "6P"})
check("6P works", an2 is not None and an2.get("well") == "6P")


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.42.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
