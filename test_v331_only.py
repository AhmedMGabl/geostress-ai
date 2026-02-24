"""Focused test for v3.31.0 endpoints: BMA Ensemble + Misclassification + Expert Feedback + Wellbore Stability."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8108"


def api(method, path, body=None, timeout=120):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def api_expect_error(method, path, body=None, expected_status=400):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return False
    except urllib.error.HTTPError as e:
        return e.code == expected_status


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


# ── [102] Bayesian Model Averaging ─────────────────────
print("\n[102] Bayesian Model Averaging")
bma = api("POST", "/api/analysis/bma-ensemble", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", bma is not None)
check("Has well", bma.get("well") == "3P")
check("Has n_samples", isinstance(bma.get("n_samples"), int) and bma["n_samples"] > 0)
check("Has n_models", isinstance(bma.get("n_models"), int) and bma["n_models"] >= 4)
check("Has n_classes", isinstance(bma.get("n_classes"), int) and bma["n_classes"] >= 2)
check("Has class_names", isinstance(bma.get("class_names"), list))
check("Has bma_accuracy", isinstance(bma.get("bma_accuracy"), (int, float)))
check("Has bma_balanced_accuracy", isinstance(bma.get("bma_balanced_accuracy"), (int, float)) and bma["bma_balanced_accuracy"] > 0)
check("Has best_single_model", isinstance(bma.get("best_single_model"), str))
check("Has best_single_balanced_accuracy", isinstance(bma.get("best_single_balanced_accuracy"), (int, float)))
check("Has bma_improvement", isinstance(bma.get("bma_improvement"), (int, float)))
check("Has model_weights", isinstance(bma.get("model_weights"), list) and len(bma["model_weights"]) >= 4)

mw0 = bma["model_weights"][0]
check("MW has model", isinstance(mw0.get("model"), str))
check("MW has weight", isinstance(mw0.get("weight"), (int, float)) and 0 <= mw0["weight"] <= 1)
check("MW has accuracy", isinstance(mw0.get("balanced_accuracy"), (int, float)))
check("MW has contribution_pct", isinstance(mw0.get("contribution_pct"), (int, float)))

check("Weights sum ~1", abs(sum(mw["weight"] for mw in bma["model_weights"]) - 1.0) < 0.01)

check("Has calibration", isinstance(bma.get("calibration"), list))
check("Has expected_calibration_error", isinstance(bma.get("expected_calibration_error"), (int, float)))

check("Has uncertainty_decomposition", isinstance(bma.get("uncertainty_decomposition"), dict))
ud = bma["uncertainty_decomposition"]
check("UD has epistemic", isinstance(ud.get("epistemic"), (int, float)))
check("UD has aleatoric", isinstance(ud.get("aleatoric"), (int, float)))
check("UD has dominant_source", ud.get("dominant_source") in ("epistemic", "aleatoric"))

check("Has sample_predictions", isinstance(bma.get("sample_predictions"), list) and len(bma["sample_predictions"]) > 0)
sp0 = bma["sample_predictions"][0]
check("SP has bma_prediction", isinstance(sp0.get("bma_prediction"), str))
check("SP has bma_confidence", isinstance(sp0.get("bma_confidence"), (int, float)))
check("SP has correct", isinstance(sp0.get("correct"), bool))
check("SP has model_agreement", isinstance(sp0.get("model_agreement"), (int, float)))

check("Has recommendations", isinstance(bma.get("recommendations"), list) and len(bma["recommendations"]) > 0)
check("Has plot", isinstance(bma.get("plot"), str) and len(bma["plot"]) > 100)
check("Has stakeholder_brief", isinstance(bma.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(bma["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", bma["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(bma["stakeholder_brief"].get("what_this_means"), str))


# ── [103] Misclassification Analysis ─────────────────────
print("\n[103] Misclassification Analysis")
mc = api("POST", "/api/analysis/misclassification-analysis", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", mc is not None)
check("Has well", mc.get("well") == "3P")
check("Has n_samples", isinstance(mc.get("n_samples"), int) and mc["n_samples"] > 0)
check("Has n_classes", isinstance(mc.get("n_classes"), int))
check("Has model_used", isinstance(mc.get("model_used"), str))
check("Has n_misclassified", isinstance(mc.get("n_misclassified"), int))
check("Has misclass_rate", isinstance(mc.get("misclass_rate"), (int, float)))

check("Has confusion_pairs", isinstance(mc.get("confusion_pairs"), list))
if mc["confusion_pairs"]:
    cp0 = mc["confusion_pairs"][0]
    check("CP has true_class", isinstance(cp0.get("true_class"), str))
    check("CP has predicted_as", isinstance(cp0.get("predicted_as"), str))
    check("CP has count", isinstance(cp0.get("count"), int))
    check("CP has rate", isinstance(cp0.get("rate"), (int, float)))
    check("CP has severity", cp0.get("severity") in ("CRITICAL", "HIGH", "MEDIUM", "LOW"))

check("Has per_class_errors", isinstance(mc.get("per_class_errors"), list) and len(mc["per_class_errors"]) >= 2)
pe0 = mc["per_class_errors"][0]
check("PE has class", isinstance(pe0.get("class"), str))
check("PE has n_errors", isinstance(pe0.get("n_errors"), int))
check("PE has error_rate", isinstance(pe0.get("error_rate"), (int, float)))
check("PE has status", pe0.get("status") in ("CRITICAL", "HIGH", "MEDIUM", "LOW"))
check("PE has geological_reason", isinstance(pe0.get("geological_reason"), str) and len(pe0["geological_reason"]) > 10)

check("Has confidence_analysis", isinstance(mc.get("confidence_analysis"), dict))
ca = mc["confidence_analysis"]
check("CA has overconfidence_risk", ca.get("overconfidence_risk") in ("HIGH", "MEDIUM", "LOW"))
check("CA has n_confident_errors", isinstance(ca.get("n_confident_errors"), int))

check("Has cross_model_errors", isinstance(mc.get("cross_model_errors"), list))
if mc["cross_model_errors"]:
    cme0 = mc["cross_model_errors"][0]
    check("CME has true_class", isinstance(cme0.get("true_class"), str))
    check("CME has hardness", cme0.get("hardness") in ("HARD", "AMBIGUOUS", "BORDERLINE"))

check("Has recommendations", isinstance(mc.get("recommendations"), list) and len(mc["recommendations"]) > 0)
r0 = mc["recommendations"][0]
check("Rec has priority", r0.get("priority") in ("CRITICAL", "HIGH", "MEDIUM", "LOW"))
check("Rec has category", isinstance(r0.get("category"), str))
check("Has Negative Examples rec", any("Negative" in r["category"] for r in mc["recommendations"]))

check("Has plot", isinstance(mc.get("plot"), str) and len(mc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(mc.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(mc["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", mc["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))


# ── [104] Expert Feedback Loop ─────────────────────────
print("\n[104] Expert Feedback Loop")
# Step 1: Submit corrections
fb_submit = api("POST", "/api/analysis/expert-feedback-submit", {
    "source": "demo", "well": "3P",
    "corrections": [
        {"index": 0, "corrected_class": "Continuous", "reason": "Expert review from image log"},
        {"index": 5, "corrected_class": "Boundary", "reason": "Clear bed boundary visible"},
        {"index": 10, "corrected_class": "Discontinuous", "reason": "Short fracture trace"},
    ]
})
check("Submit status", fb_submit.get("status") == "accepted")
check("Corrections accepted", fb_submit.get("n_corrections_accepted") == 3)
check("Total stored", fb_submit.get("total_corrections_stored") >= 3)

# Step 2: Retrain with corrections
fb = api("POST", "/api/analysis/expert-feedback-retrain", {"source": "demo", "well": "3P"}, timeout=300)
check("Retrain status 200", fb is not None)
check("Has well", fb.get("well") == "3P")
check("Has n_corrections_submitted", isinstance(fb.get("n_corrections_submitted"), int) and fb["n_corrections_submitted"] >= 3)
check("Has n_corrections_applied", isinstance(fb.get("n_corrections_applied"), int))
check("Has accuracy_original", isinstance(fb.get("accuracy_original"), (int, float)))
check("Has balanced_accuracy_original", isinstance(fb.get("balanced_accuracy_original"), (int, float)))
check("Has accuracy_corrected", isinstance(fb.get("accuracy_corrected"), (int, float)))
check("Has balanced_accuracy_corrected", isinstance(fb.get("balanced_accuracy_corrected"), (int, float)))
check("Has improvement", isinstance(fb.get("improvement"), (int, float)))
check("Has per_class_comparison", isinstance(fb.get("per_class_comparison"), list))
check("Has recommendations", isinstance(fb.get("recommendations"), list) and len(fb["recommendations"]) > 0)
check("Has plot", isinstance(fb.get("plot"), str) and len(fb["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fb.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(fb["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", fb["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# Validate: no corrections = error
check("No corrections rejected", api_expect_error("POST", "/api/analysis/expert-feedback-submit", {"source": "demo", "well": "3P", "corrections": []}))


# ── [105] Wellbore Stability ────────────────────────────
print("\n[105] Wellbore Stability Assessment")
wb = api("POST", "/api/analysis/wellbore-stability", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", wb is not None)
check("Has well", wb.get("well") == "3P")
check("Has depth_range_m", isinstance(wb.get("depth_range_m"), dict))
check("Has stress_regime", wb.get("stress_regime") in ("Normal", "Reverse", "Strike-Slip"))

check("Has stress_parameters", isinstance(wb.get("stress_parameters"), dict))
sp = wb["stress_parameters"]
check("SP has sigma1", isinstance(sp.get("sigma1_MPa"), (int, float)))
check("SP has sigma2", isinstance(sp.get("sigma2_MPa"), (int, float)))
check("SP has sigma3", isinstance(sp.get("sigma3_MPa"), (int, float)))
check("SP has Sv", isinstance(sp.get("Sv_MPa"), (int, float)))
check("SP has SH", isinstance(sp.get("SH_MPa"), (int, float)))
check("SP has Sh", isinstance(sp.get("Sh_MPa"), (int, float)))
check("SP has Pp", isinstance(sp.get("Pp_MPa"), (int, float)))
check("SP has UCS", isinstance(sp.get("UCS_estimate_MPa"), (int, float)))

check("Has mud_weight_window", isinstance(wb.get("mud_weight_window"), dict))
mww = wb["mud_weight_window"]
check("MWW has min_ppg", isinstance(mww.get("min_ppg"), (int, float)))
check("MWW has max_ppg", isinstance(mww.get("max_ppg"), (int, float)))
check("MWW has window_ppg", isinstance(mww.get("window_ppg"), (int, float)))
check("MWW has pore_pressure_ppg", isinstance(mww.get("pore_pressure_ppg"), (int, float)))
check("MWW has status", mww.get("status") in ("SAFE", "NARROW", "CRITICAL"))

check("Has depth_profile", isinstance(wb.get("depth_profile"), list) and len(wb["depth_profile"]) > 0)
dp0 = wb["depth_profile"][0]
check("DP has depth_m", isinstance(dp0.get("depth_m"), (int, float)))
check("DP has Sv_MPa", isinstance(dp0.get("Sv_MPa"), (int, float)))
check("DP has mw_min_ppg", isinstance(dp0.get("mw_min_ppg"), (int, float)))
check("DP has risk_level", dp0.get("risk_level") in ("GREEN", "AMBER", "RED"))

check("Has n_fractures_analyzed", isinstance(wb.get("n_fractures_analyzed"), int) and wb["n_fractures_analyzed"] > 0)
check("Has n_critically_stressed", isinstance(wb.get("n_critically_stressed"), int))
check("Has pct_critically_stressed", isinstance(wb.get("pct_critically_stressed"), (int, float)))

check("Has fracture_risk", isinstance(wb.get("fracture_risk"), list) and len(wb["fracture_risk"]) > 0)
fr0 = wb["fracture_risk"][0]
check("FR has slip_tendency", isinstance(fr0.get("slip_tendency"), (int, float)))
check("FR has dilation_tendency", isinstance(fr0.get("dilation_tendency"), (int, float)))
check("FR has critically_stressed", isinstance(fr0.get("critically_stressed"), bool))

check("Has recommendations", isinstance(wb.get("recommendations"), list) and len(wb["recommendations"]) > 0)
wr0 = wb["recommendations"][0]
check("Rec has priority", wr0.get("priority") in ("CRITICAL", "HIGH", "MEDIUM", "LOW"))
check("Rec has category", isinstance(wr0.get("category"), str))

check("Has plot", isinstance(wb.get("plot"), str) and len(wb["plot"]) > 100)
check("Has stakeholder_brief", isinstance(wb.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(wb["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", wb["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(wb["stakeholder_brief"].get("what_this_means"), str))

# Test with mud weight parameter
wb2 = api("POST", "/api/analysis/wellbore-stability", {"source": "demo", "well": "3P", "mud_weight_ppg": 10.0}, timeout=300)
check("Mud weight assessment exists", wb2.get("current_mud_weight_assessment") is not None)
cma = wb2["current_mud_weight_assessment"]
check("CMA has risk", isinstance(cma.get("risk"), str))
check("CMA has risk_level", cma.get("risk_level") in ("CRITICAL", "HIGH", "MEDIUM", "LOW"))
check("CMA has margin_to_min", isinstance(cma.get("margin_to_min_ppg"), (int, float)))


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.31.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
