"""Focused test for v3.37.0 endpoints: Terzaghi + Effective Stress + Decision Intelligence + Feedback + Failure-Aware."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8115"


def api(method, path, body=None, timeout=120):
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


# ── [120] Terzaghi Sampling Bias Correction ─────────────────
print("\n[120] Terzaghi Sampling Bias Correction")
tz = api("POST", "/api/analysis/terzaghi-corrected", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", tz is not None)
check("Has well", tz.get("well") == "3P")
check("Has n_fractures", isinstance(tz.get("n_fractures"), int) and tz["n_fractures"] > 0)
check("Has bias_severity", tz.get("bias_severity") in ("LOW", "MODERATE", "HIGH"))
check("Has effective_sample_size", isinstance(tz.get("effective_sample_size"), (int, float)) and tz["effective_sample_size"] > 0)
check("Has n_blind_zone", isinstance(tz.get("n_blind_zone"), int))
check("Has pct_blind_zone", isinstance(tz.get("pct_blind_zone"), (int, float)))
check("Has uncorrected", isinstance(tz.get("uncorrected"), dict))
check("Uncorrected has mean_azimuth", isinstance(tz["uncorrected"].get("mean_azimuth_deg"), (int, float)))
check("Uncorrected has mean_dip", isinstance(tz["uncorrected"].get("mean_dip_deg"), (int, float)))
check("Has corrected", isinstance(tz.get("corrected"), dict))
check("Corrected has mean_azimuth", isinstance(tz["corrected"].get("mean_azimuth_deg"), (int, float)))
check("Corrected has mean_dip", isinstance(tz["corrected"].get("mean_dip_deg"), (int, float)))
check("Has azimuth_shift_deg", isinstance(tz.get("azimuth_shift_deg"), (int, float)))
check("Has dip_shift_deg", isinstance(tz.get("dip_shift_deg"), (int, float)))
check("Has max_weight", isinstance(tz.get("max_weight"), (int, float)) and tz["max_weight"] >= 1.0)
check("Has type_stats", isinstance(tz.get("type_stats"), list) and len(tz["type_stats"]) >= 2)
ts0 = tz["type_stats"][0]
check("TS has fracture_type", isinstance(ts0.get("fracture_type"), str))
check("TS has n_raw", isinstance(ts0.get("n_raw"), int))
check("TS has pct_raw", isinstance(ts0.get("pct_raw"), (int, float)))
check("TS has pct_corrected", isinstance(ts0.get("pct_corrected"), (int, float)))
check("TS has mean_weight", isinstance(ts0.get("mean_weight"), (int, float)))
check("Has fracture_data", isinstance(tz.get("fracture_data"), list) and len(tz["fracture_data"]) > 0)
fd0 = tz["fracture_data"][0]
check("FD has alpha_deg", isinstance(fd0.get("alpha_deg"), (int, float)))
check("FD has terzaghi_weight", isinstance(fd0.get("terzaghi_weight"), (int, float)))
check("FD has in_blind_zone", isinstance(fd0.get("in_blind_zone"), bool))
check("Has recommendations", isinstance(tz.get("recommendations"), list) and len(tz["recommendations"]) > 0)
check("Has plot", isinstance(tz.get("plot"), str) and len(tz["plot"]) > 100)
check("Has stakeholder_brief", isinstance(tz.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(tz["stakeholder_brief"].get("for_non_experts"), str))

# 6P test
tz2 = api("POST", "/api/analysis/terzaghi-corrected", {"source": "demo", "well": "6P"}, timeout=300)
check("6P works", tz2 is not None and tz2.get("well") == "6P")


# ── [121] Effective Stress Profile ─────────────────────────
print("\n[121] Effective Stress Profile (Biot)")
es = api("POST", "/api/analysis/effective-stress-profile", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", es is not None)
check("Has well", es.get("well") == "3P")
check("Has biot_coefficient", isinstance(es.get("biot_coefficient"), (int, float)))
check("Has depth_range_m", isinstance(es.get("depth_range_m"), dict))
check("Has n_profile_points", isinstance(es.get("n_profile_points"), int) and es["n_profile_points"] > 0)
check("Has n_narrow_zones", isinstance(es.get("n_narrow_zones"), int))
check("Has n_critical_zones", isinstance(es.get("n_critical_zones"), int))
check("Has profile", isinstance(es.get("profile"), list) and len(es["profile"]) > 0)
p0 = es["profile"][0]
check("P has depth_m", isinstance(p0.get("depth_m"), (int, float)))
check("P has Sv_MPa", isinstance(p0.get("Sv_MPa"), (int, float)))
check("P has SHmax_MPa", isinstance(p0.get("SHmax_MPa"), (int, float)))
check("P has Shmin_MPa", isinstance(p0.get("Shmin_MPa"), (int, float)))
check("P has Pp_MPa", isinstance(p0.get("Pp_MPa"), (int, float)))
check("P has Sv_eff_MPa", isinstance(p0.get("Sv_eff_MPa"), (int, float)))
check("P has SHmax_eff_MPa", isinstance(p0.get("SHmax_eff_MPa"), (int, float)))
check("P has Shmin_eff_MPa", isinstance(p0.get("Shmin_eff_MPa"), (int, float)))
check("P has mud_weight_window_MPa", isinstance(p0.get("mud_weight_window_MPa"), (int, float)))
check("Effective < Total", p0["Sv_eff_MPa"] < p0["Sv_MPa"])
check("Has recommendations", isinstance(es.get("recommendations"), list) and len(es["recommendations"]) > 0)
check("Has plot", isinstance(es.get("plot"), str) and len(es["plot"]) > 100)
check("Has stakeholder_brief", isinstance(es.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(es["stakeholder_brief"].get("for_non_experts"), str))

# 6P test
es2 = api("POST", "/api/analysis/effective-stress-profile", {"source": "demo", "well": "6P"}, timeout=300)
check("6P works", es2 is not None and es2.get("well") == "6P")


# ── [122] Decision Intelligence ─────────────────────────────
print("\n[122] Decision Intelligence Dashboard")
di = api("POST", "/api/analysis/decision-intelligence", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", di is not None)
check("Has well", di.get("well") == "3P")
check("Has overall_risk", di.get("overall_risk") in ("GREEN", "AMBER", "RED"))
check("Has n_red", isinstance(di.get("n_red"), int))
check("Has n_amber", isinstance(di.get("n_amber"), int))
check("Has n_green", isinstance(di.get("n_green"), int))
check("Risk counts sum to matrix", di["n_red"] + di["n_amber"] + di["n_green"] == len(di.get("risk_matrix", [])))
check("Has risk_matrix", isinstance(di.get("risk_matrix"), list) and len(di["risk_matrix"]) >= 3)
rm0 = di["risk_matrix"][0]
check("RM has category", isinstance(rm0.get("category"), str))
check("RM has risk_level", rm0.get("risk_level") in ("GREEN", "AMBER", "RED"))
check("RM has plain_english", isinstance(rm0.get("plain_english"), str))
check("RM has what_if_ignored", isinstance(rm0.get("what_if_ignored"), str))
check("RM has recommended_action", isinstance(rm0.get("recommended_action"), str))
check("RM has confidence", rm0.get("confidence") in ("HIGH", "MEDIUM", "LOW"))
check("Has model_balanced_accuracy", isinstance(di.get("model_balanced_accuracy"), (int, float)))
check("Has data_quality", di.get("data_quality") in ("GOOD", "FAIR", "POOR"))
check("Has prioritized_actions", isinstance(di.get("prioritized_actions"), list) and len(di["prioritized_actions"]) > 0)
check("Has glossary", isinstance(di.get("glossary"), list) and len(di["glossary"]) >= 5)
g0 = di["glossary"][0]
check("Glossary has term", isinstance(g0.get("term"), str))
check("Glossary has definition", isinstance(g0.get("definition"), str))
check("Has recommendations", isinstance(di.get("recommendations"), list) and len(di["recommendations"]) > 0)
check("Has plot", isinstance(di.get("plot"), str) and len(di["plot"]) > 100)
check("Has stakeholder_brief", isinstance(di.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(di["stakeholder_brief"].get("for_non_experts"), str))


# ── [123] Accuracy Feedback Loop ────────────────────────────
print("\n[123] Accuracy Feedback Loop")

# First check empty state
trend0 = api("GET", "/api/feedback/accuracy-trend", timeout=30)
check("Empty trend", trend0.get("n_outcomes") == 0)

# Submit some outcomes
out1 = api("POST", "/api/feedback/submit-outcome", {
    "well": "3P", "depth_m": 2500.0, "predicted_type": "Continuous", "actual_type": "Continuous", "confidence": 0.9
})
check("Submit correct", out1.get("is_correct") is True)
check("Rolling accuracy 100%", out1.get("rolling_accuracy") == 1.0)

out2 = api("POST", "/api/feedback/submit-outcome", {
    "well": "3P", "depth_m": 2600.0, "predicted_type": "Continuous", "actual_type": "Vuggy", "confidence": 0.6
})
check("Submit incorrect", out2.get("is_correct") is False)
check("Rolling accuracy 50%", out2.get("rolling_accuracy") == 0.5)

out3 = api("POST", "/api/feedback/submit-outcome", {
    "well": "3P", "depth_m": 2700.0, "predicted_type": "Boundary", "actual_type": "Boundary"
})
check("3rd outcome", out3.get("n_total_outcomes") == 3)

# Check accuracy trend
trend1 = api("GET", "/api/feedback/accuracy-trend", timeout=30)
check("Has 3 outcomes", trend1.get("n_outcomes") == 3)
check("Has rolling_accuracy", isinstance(trend1.get("rolling_accuracy"), (int, float)))
check("Has per_type_accuracy", isinstance(trend1.get("per_type_accuracy"), list) and len(trend1["per_type_accuracy"]) >= 2)
check("Has confusion_data", isinstance(trend1.get("confusion_data"), list) and len(trend1["confusion_data"]) > 0)
check("Has accuracy_trend", isinstance(trend1.get("accuracy_trend"), list) and len(trend1["accuracy_trend"]) > 0)
check("Has recommendations", isinstance(trend1.get("recommendations"), list) and len(trend1["recommendations"]) > 0)
check("Has stakeholder_brief", isinstance(trend1.get("stakeholder_brief"), dict))

# Trigger retrain
rt = api("POST", "/api/feedback/retrain-trigger", {"source": "demo", "well": "3P"}, timeout=300)
check("Retrain status", rt is not None)
check("Has baseline_accuracy", isinstance(rt.get("baseline_accuracy"), (int, float)))
check("Has retrained_accuracy", isinstance(rt.get("retrained_accuracy"), (int, float)))
check("Has improvement", isinstance(rt.get("improvement"), (int, float)))
check("Has stakeholder_brief", isinstance(rt.get("stakeholder_brief"), dict))


# ── [124] Failure-Aware Classification ──────────────────────
print("\n[124] Failure-Aware Classification")
fa = api("POST", "/api/analysis/failure-aware-classification", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", fa is not None)
check("Has well", fa.get("well") == "3P")
check("Has high_cost_types", isinstance(fa.get("high_cost_types"), list))
check("Has cost_ratio", isinstance(fa.get("cost_ratio"), (int, float)))
check("Has standard_model", isinstance(fa.get("standard_model"), dict))
check("Std has balanced_accuracy", isinstance(fa["standard_model"].get("balanced_accuracy"), (int, float)))
check("Std has n_high_cost_errors", isinstance(fa["standard_model"].get("n_high_cost_errors"), int))
check("Std has total_cost", isinstance(fa["standard_model"].get("total_cost"), (int, float)))
check("Std has confusion_matrix", isinstance(fa["standard_model"].get("confusion_matrix"), list))
check("Has failure_aware_model", isinstance(fa.get("failure_aware_model"), dict))
check("FA has balanced_accuracy", isinstance(fa["failure_aware_model"].get("balanced_accuracy"), (int, float)))
check("FA has n_high_cost_errors", isinstance(fa["failure_aware_model"].get("n_high_cost_errors"), int))
check("FA has total_cost", isinstance(fa["failure_aware_model"].get("total_cost"), (int, float)))
check("Has cost_reduction_pct", isinstance(fa.get("cost_reduction_pct"), (int, float)))
check("Has type_comparison", isinstance(fa.get("type_comparison"), list) and len(fa["type_comparison"]) >= 2)
tc0 = fa["type_comparison"][0]
check("TC has fracture_type", isinstance(tc0.get("fracture_type"), str))
check("TC has standard_accuracy", isinstance(tc0.get("standard_accuracy"), (int, float)))
check("TC has failure_aware_accuracy", isinstance(tc0.get("failure_aware_accuracy"), (int, float)))
check("TC has is_high_cost", isinstance(tc0.get("is_high_cost"), bool))
check("Has recommendations", isinstance(fa.get("recommendations"), list) and len(fa["recommendations"]) > 0)
check("Has plot", isinstance(fa.get("plot"), str) and len(fa["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fa.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(fa["stakeholder_brief"].get("for_non_experts"), str))

# 6P test
fa2 = api("POST", "/api/analysis/failure-aware-classification", {"source": "demo", "well": "6P"}, timeout=300)
check("6P works", fa2 is not None and fa2.get("well") == "6P")


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.37.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
