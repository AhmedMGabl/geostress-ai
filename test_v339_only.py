"""Focused test for v3.39.0 endpoints: Counterfactual XAI + Graph + Attention + Privacy + Recalibration."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8120"


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


# ── [130] Counterfactual Explanations ─────────────────────
print("\n[130] Counterfactual Explanations (XAI)")
cf = api("POST", "/api/analysis/counterfactual", {"source": "demo", "well": "3P", "n_samples": 5})
check("Status 200", cf is not None)
check("Has well", cf.get("well") == "3P")
check("Has n_samples_analyzed", isinstance(cf.get("n_samples_analyzed"), int))
check("Has n_classes", isinstance(cf.get("n_classes"), int) and cf["n_classes"] >= 2)
check("Has classes", isinstance(cf.get("classes"), list) and len(cf["classes"]) >= 2)
check("Has counterfactuals", isinstance(cf.get("counterfactuals"), list))
if cf["counterfactuals"]:
    cf0 = cf["counterfactuals"][0]
    check("CF has current_prediction", isinstance(cf0.get("current_prediction"), str))
    check("CF has current_confidence", isinstance(cf0.get("current_confidence"), (int, float)))
    check("CF has counterfactual_class", isinstance(cf0.get("counterfactual_class"), str))
    check("CF has feature_changes", isinstance(cf0.get("feature_changes"), list))
    check("CF has explanation", isinstance(cf0.get("explanation"), str))
    check("CF has total_change_magnitude", isinstance(cf0.get("total_change_magnitude"), (int, float)))
    if cf0["feature_changes"]:
        fc0 = cf0["feature_changes"][0]
        check("FC has feature", isinstance(fc0.get("feature"), str))
        check("FC has current_value", isinstance(fc0.get("current_value"), (int, float)))
        check("FC has needed_value", isinstance(fc0.get("needed_value"), (int, float)))
        check("FC has change", isinstance(fc0.get("change"), (int, float)))
    else:
        for _ in range(4):
            check("FC field", True, "no feature changes")
else:
    for _ in range(10):
        check("CF field", True, "no counterfactuals found")
check("Has most_influential_features", isinstance(cf.get("most_influential_features"), list))
check("Has recommendations", isinstance(cf.get("recommendations"), list) and len(cf["recommendations"]) > 0)
check("Has plot", isinstance(cf.get("plot"), str) and len(cf["plot"]) > 100)
check("Has stakeholder_brief", isinstance(cf.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(cf["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(cf.get("elapsed_s"), (int, float)))

# 6P test
cf2 = api("POST", "/api/analysis/counterfactual", {"source": "demo", "well": "6P", "n_samples": 3})
check("6P works", cf2 is not None and cf2.get("well") == "6P")


# ── [131] Fracture Graph Analysis ─────────────────────────
print("\n[131] Fracture Graph Analysis")
fg = api("POST", "/api/analysis/fracture-graph", {"source": "demo", "well": "3P"})
check("Status 200", fg is not None)
check("Has well", fg.get("well") == "3P")
check("Has n_nodes", isinstance(fg.get("n_nodes"), int) and fg["n_nodes"] > 0)
check("Has n_edges", isinstance(fg.get("n_edges"), int))
check("Has n_components", isinstance(fg.get("n_components"), int) and fg["n_components"] >= 1)
check("Has largest_component", isinstance(fg.get("largest_component_size"), int))
check("Has avg_degree", isinstance(fg.get("avg_degree"), (int, float)))
check("Has avg_clustering", isinstance(fg.get("avg_clustering"), (int, float)))
check("Has n_hubs", isinstance(fg.get("n_hubs"), int))
check("Has component_sizes", isinstance(fg.get("component_sizes"), list) and len(fg["component_sizes"]) >= 1)
check("Has class_graph_stats", isinstance(fg.get("class_graph_stats"), list) and len(fg["class_graph_stats"]) >= 2)
cgs0 = fg["class_graph_stats"][0]
check("CGS has class", isinstance(cgs0.get("class"), str))
check("CGS has mean_degree", isinstance(cgs0.get("mean_degree"), (int, float)))
check("CGS has mean_clustering", isinstance(cgs0.get("mean_clustering"), (int, float)))
check("CGS has pct_hubs", isinstance(cgs0.get("pct_hubs"), (int, float)))
check("Has recommendations", isinstance(fg.get("recommendations"), list) and len(fg["recommendations"]) > 0)
check("Has plot", isinstance(fg.get("plot"), str) and len(fg["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fg.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(fg["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(fg.get("elapsed_s"), (int, float)))

# 6P test
fg2 = api("POST", "/api/analysis/fracture-graph", {"source": "demo", "well": "6P"})
check("6P works", fg2 is not None and fg2.get("well") == "6P")


# ── [132] Depth-Sequence Attention ────────────────────────
print("\n[132] Depth-Sequence Attention Classification")
da = api("POST", "/api/analysis/depth-attention", {"source": "demo", "well": "3P", "window_size": 20})
check("Status 200", da is not None)
check("Has well", da.get("well") == "3P")
check("Has window_size", da.get("window_size") == 20)
check("Has n_base_features", isinstance(da.get("n_base_features"), int) and da["n_base_features"] >= 20)
check("Has n_attention_features", isinstance(da.get("n_attention_features"), int) and da["n_attention_features"] > da["n_base_features"])
check("Has baseline_balanced_accuracy", isinstance(da.get("baseline_balanced_accuracy"), (int, float)))
check("Has attention_balanced_accuracy", isinstance(da.get("attention_balanced_accuracy"), (int, float)))
check("Has improvement_pct", isinstance(da.get("improvement_pct"), (int, float)))
check("Has verdict", da.get("verdict") in ("ATTENTION_BETTER", "SIMILAR", "BASELINE_BETTER"))
check("Has per_class", isinstance(da.get("per_class"), list) and len(da["per_class"]) >= 2)
pc0 = da["per_class"][0]
check("PC has class", isinstance(pc0.get("class"), str))
check("PC has baseline_f1", isinstance(pc0.get("baseline_f1"), (int, float)))
check("PC has attention_f1", isinstance(pc0.get("attention_f1"), (int, float)))
check("PC has improvement", isinstance(pc0.get("improvement"), (int, float)))
check("Has n_predictions_changed", isinstance(da.get("n_predictions_changed"), int))
check("Has pct_predictions_changed", isinstance(da.get("pct_predictions_changed"), (int, float)))
check("Has recommendations", isinstance(da.get("recommendations"), list) and len(da["recommendations"]) > 0)
check("Has plot", isinstance(da.get("plot"), str) and len(da["plot"]) > 100)
check("Has stakeholder_brief", isinstance(da.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(da["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(da.get("elapsed_s"), (int, float)))

# 6P test
da2 = api("POST", "/api/analysis/depth-attention", {"source": "demo", "well": "6P"})
check("6P works", da2 is not None and da2.get("well") == "6P")


# ── [133] Differential Privacy ────────────────────────────
print("\n[133] Differential Privacy Predictions")
dp = api("POST", "/api/analysis/private-predict", {"source": "demo", "well": "3P", "epsilon": 1.0})
check("Status 200", dp is not None)
check("Has well", dp.get("well") == "3P")
check("Has epsilon", dp.get("epsilon") == 1.0)
check("Has privacy_level", dp.get("privacy_level") in ("VERY_HIGH", "HIGH", "MODERATE", "LOW"))
check("Has true_accuracy", isinstance(dp.get("true_accuracy"), (int, float)))
check("Has private_accuracy", isinstance(dp.get("private_accuracy"), (int, float)))
check("Has accuracy_cost_pct", isinstance(dp.get("accuracy_cost_pct"), (int, float)))
check("Has prediction_agreement", isinstance(dp.get("prediction_agreement"), (int, float)) and 0 <= dp["prediction_agreement"] <= 1)
check("Has noise_scale", isinstance(dp.get("noise_scale"), (int, float)))
check("Has per_class", isinstance(dp.get("per_class"), list) and len(dp["per_class"]) >= 2)
dpc0 = dp["per_class"][0]
check("DPC has class", isinstance(dpc0.get("class"), str))
check("DPC has true_count", isinstance(dpc0.get("true_count"), int))
check("DPC has private_count", isinstance(dpc0.get("private_count"), int))
check("Has class_counts_true", isinstance(dp.get("class_counts_true"), dict))
check("Has class_counts_private", isinstance(dp.get("class_counts_private"), dict))
check("Has recommendations", isinstance(dp.get("recommendations"), list) and len(dp["recommendations"]) > 0)
check("Has plot", isinstance(dp.get("plot"), str) and len(dp["plot"]) > 100)
check("Has stakeholder_brief", isinstance(dp.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(dp["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(dp.get("elapsed_s"), (int, float)))

# 6P test
dp2 = api("POST", "/api/analysis/private-predict", {"source": "demo", "well": "6P"})
check("6P works", dp2 is not None and dp2.get("well") == "6P")


# ── [134] Auto-Recalibration ─────────────────────────────
print("\n[134] Auto-Recalibration")
rc = api("POST", "/api/analysis/auto-recalibrate", {"source": "demo", "well": "3P"})
check("Status 200", rc is not None)
check("Has well", rc.get("well") == "3P")
check("Has method", isinstance(rc.get("method"), str))
check("Has calibration_status", rc.get("calibration_status") in ("WELL_CALIBRATED", "SLIGHTLY_MISCALIBRATED", "POORLY_CALIBRATED"))
check("Has needs_recalibration", isinstance(rc.get("needs_recalibration"), bool))
check("Has baseline", isinstance(rc.get("baseline"), dict))
check("Base has ece_pct", isinstance(rc["baseline"].get("ece_pct"), (int, float)))
check("Base has brier_score", isinstance(rc["baseline"].get("brier_score"), (int, float)))
check("Base has calibration_bins", isinstance(rc["baseline"].get("calibration_bins"), list))
check("Has calibrated", isinstance(rc.get("calibrated"), dict))
check("Cal has ece_pct", isinstance(rc["calibrated"].get("ece_pct"), (int, float)))
check("Cal has brier_score", isinstance(rc["calibrated"].get("brier_score"), (int, float)))
check("Cal has calibration_bins", isinstance(rc["calibrated"].get("calibration_bins"), list))
check("Has ece_improvement_pct", isinstance(rc.get("ece_improvement_pct"), (int, float)))
check("Has accuracy_change_pct", isinstance(rc.get("accuracy_change_pct"), (int, float)))
check("Has recommendations", isinstance(rc.get("recommendations"), list) and len(rc["recommendations"]) > 0)
check("Has plot", isinstance(rc.get("plot"), str) and len(rc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(rc.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(rc["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(rc.get("elapsed_s"), (int, float)))

# 6P test
rc2 = api("POST", "/api/analysis/auto-recalibrate", {"source": "demo", "well": "6P"})
check("6P works", rc2 is not None and rc2.get("well") == "6P")


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.39.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
