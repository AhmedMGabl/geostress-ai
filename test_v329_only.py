"""Focused test for v3.29.0 endpoints: Conformal Prediction, Physics Consistency, Active Learning Strategy."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8101"


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


# ── [97] Conformal Prediction ───────────────────────────
print("\n[97] Conformal Prediction")
cp = api("POST", "/api/analysis/conformal-predict", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", cp is not None)
check("Has well", cp.get("well") == "3P")
check("Has alpha", cp.get("alpha") == 0.1)
check("Has target_coverage", isinstance(cp.get("target_coverage"), (int, float)) and cp["target_coverage"] == 0.9)
check("Has empirical_coverage", isinstance(cp.get("empirical_coverage"), (int, float)) and 0 < cp["empirical_coverage"] <= 1)
check("Has avg_set_size", isinstance(cp.get("avg_set_size"), (int, float)) and cp["avg_set_size"] >= 1)
check("Has singleton_rate", isinstance(cp.get("singleton_rate"), (int, float)) and 0 <= cp["singleton_rate"] <= 1)
check("Has n_samples", isinstance(cp.get("n_samples"), int) and cp["n_samples"] > 0)
check("Has n_test", isinstance(cp.get("n_test"), int) and cp["n_test"] > 0)
check("Has n_calibration", isinstance(cp.get("n_calibration"), int) and cp["n_calibration"] > 0)
check("Has conformal_threshold", isinstance(cp.get("conformal_threshold"), (int, float)))
check("Has model_used", isinstance(cp.get("model_used"), str))
check("Has test_predictions", isinstance(cp.get("test_predictions"), list) and len(cp["test_predictions"]) > 0)

tp0 = cp["test_predictions"][0]
check("TP has index", isinstance(tp0.get("index"), int))
check("TP has true_class", isinstance(tp0.get("true_class"), str))
check("TP has point_prediction", isinstance(tp0.get("point_prediction"), str))
check("TP has point_confidence", isinstance(tp0.get("point_confidence"), (int, float)))
check("TP has conformal_set", isinstance(tp0.get("conformal_set"), list) and len(tp0["conformal_set"]) >= 1)
check("TP has set_size", isinstance(tp0.get("set_size"), int) and tp0["set_size"] >= 1)
check("TP has covered", isinstance(tp0.get("covered"), bool))
check("TP has class_probabilities", isinstance(tp0.get("class_probabilities"), dict))

check("Has full_predictions_summary", isinstance(cp.get("full_predictions_summary"), dict))
check("Summary has n_singletons", isinstance(cp["full_predictions_summary"].get("n_singletons"), int))
check("Has per_class_coverage", isinstance(cp.get("per_class_coverage"), dict))
check("Has plot", isinstance(cp.get("plot"), str) and len(cp["plot"]) > 100)
check("Has stakeholder_brief", isinstance(cp.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(cp["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", cp["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(cp["stakeholder_brief"].get("what_this_means"), str))

# Param validation
check("alpha=0 rejected", api_expect_error("POST", "/api/analysis/conformal-predict", {"source": "demo", "well": "3P", "alpha": 0}))
check("alpha=0.99 rejected", api_expect_error("POST", "/api/analysis/conformal-predict", {"source": "demo", "well": "3P", "alpha": 0.99}))

# Custom alpha
cp2 = api("POST", "/api/analysis/conformal-predict", {"source": "demo", "well": "3P", "alpha": 0.2}, timeout=300)
check("alpha=0.2 works", cp2 is not None and cp2.get("target_coverage") == 0.8)

# ── [98] Physics-Consistency Validation ─────────────────
print("\n[98] Physics-Consistency Validation")
pc = api("POST", "/api/analysis/physics-consistency", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", pc is not None)
check("Has well", pc.get("well") == "3P")
check("Has consistency_level", pc.get("consistency_level") in ("HIGH", "MODERATE", "LOW"))
check("Has overall_score", isinstance(pc.get("overall_score"), int) and 0 <= pc["overall_score"] <= 100)
check("Has n_samples", isinstance(pc.get("n_samples"), int) and pc["n_samples"] > 0)
check("Has n_checks", isinstance(pc.get("n_checks"), int) and pc["n_checks"] >= 5)
check("Has n_violations", isinstance(pc.get("n_violations"), int))
check("Has n_warnings", isinstance(pc.get("n_warnings"), int))
check("Has checks list", isinstance(pc.get("checks"), list) and len(pc["checks"]) >= 5)

c0 = pc["checks"][0]
check("Check has check name", isinstance(c0.get("check"), str))
check("Check has score", isinstance(c0.get("score"), int) and 0 <= c0["score"] <= 100)
check("Check has detail", isinstance(c0.get("detail"), str))

check("Has stress_summary", isinstance(pc.get("stress_summary"), dict))
ss = pc["stress_summary"]
check("SS has sigma1", isinstance(ss.get("sigma1_MPa"), (int, float)))
check("SS has sigma3", isinstance(ss.get("sigma3_MPa"), (int, float)))
check("SS has R_ratio", isinstance(ss.get("R_ratio"), (int, float)))
check("SS has shmax", isinstance(ss.get("shmax_azimuth_deg"), (int, float)))
check("SS has friction", isinstance(ss.get("friction_coefficient"), (int, float)))
check("SS has pct_cs", isinstance(ss.get("pct_critically_stressed"), (int, float)))

check("Has classification_summary", isinstance(pc.get("classification_summary"), dict))
check("CS has model_used", isinstance(pc["classification_summary"].get("model_used"), str))
check("CS has accuracy", isinstance(pc["classification_summary"].get("accuracy"), (int, float)))

check("Has plot", isinstance(pc.get("plot"), str) and len(pc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(pc.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(pc["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", pc["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# Check names
check_names = [c["check"] for c in pc["checks"]]
check("Has CS correlation check", any("Critically" in cn for cn in check_names))
check("Has slip check", any("Slip" in cn for cn in check_names))
check("Has dilation check", any("Dilation" in cn for cn in check_names))

# ── [99] Active Learning Strategy ───────────────────────
print("\n[99] Active Learning Strategy")
al = api("POST", "/api/analysis/active-learning-strategy", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", al is not None)
check("Has well", al.get("well") == "3P")
check("Has strategy", al.get("strategy") in ("uncertainty", "diversity", "hybrid"))
check("Has n_samples", isinstance(al.get("n_samples"), int) and al["n_samples"] > 0)
check("Has n_suggest", isinstance(al.get("n_suggest"), int) and al["n_suggest"] == 20)
check("Has n_models_in_ensemble", isinstance(al.get("n_models_in_ensemble"), int) and al["n_models_in_ensemble"] >= 2)
check("Has current_accuracy", isinstance(al.get("current_accuracy"), (int, float)))
check("Has suggestions", isinstance(al.get("suggestions"), list) and len(al["suggestions"]) > 0)

s0 = al["suggestions"][0]
check("S has index", isinstance(s0.get("index"), int))
check("S has priority_score", isinstance(s0.get("priority_score"), (int, float)))
check("S has current_prediction", isinstance(s0.get("current_prediction"), str))
check("S has confidence", isinstance(s0.get("confidence"), (int, float)))
check("S has entropy", isinstance(s0.get("entropy"), (int, float)))
check("S has model_disagreement", isinstance(s0.get("model_disagreement"), (int, float)))
check("S has diversity_score", isinstance(s0.get("diversity_score"), (int, float)))
check("S has cluster", isinstance(s0.get("cluster"), int))
check("S has why_selected", isinstance(s0.get("why_selected"), list) and len(s0["why_selected"]) > 0)
check("S has candidates", isinstance(s0.get("candidates"), list) and len(s0["candidates"]) > 0)

cand0 = s0["candidates"][0]
check("Cand has type", isinstance(cand0.get("type"), str))
check("Cand has probability", isinstance(cand0.get("probability"), (int, float)))

check("Has learning_curve", isinstance(al.get("learning_curve"), list))
if al["learning_curve"]:
    lp0 = al["learning_curve"][0]
    check("LP has n_samples", isinstance(lp0.get("n_samples"), int))
    check("LP has accuracy", isinstance(lp0.get("accuracy"), (int, float)))

check("Has marginal_gain", isinstance(al.get("marginal_gain"), dict))
if al["marginal_gain"]:
    mg = al["marginal_gain"]
    check("MG has current_accuracy", isinstance(mg.get("current_accuracy"), (int, float)))
    check("MG has projected_accuracy", isinstance(mg.get("projected_accuracy"), (int, float)))
    check("MG has expected_improvement", isinstance(mg.get("expected_improvement"), (int, float)))
    check("MG has worth_collecting", isinstance(mg.get("worth_collecting"), bool))

check("Has class_distribution", isinstance(al.get("class_distribution"), dict))
check("Has underrepresented_classes", isinstance(al.get("underrepresented_classes"), list))
check("Has plot", isinstance(al.get("plot"), str) and len(al["plot"]) > 100)
check("Has stakeholder_brief", isinstance(al.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(al["stakeholder_brief"].get("headline"), str))

# Param validation
check("bad strategy rejected", api_expect_error("POST", "/api/analysis/active-learning-strategy", {"source": "demo", "well": "3P", "strategy": "invalid"}))

# Custom strategy
al2 = api("POST", "/api/analysis/active-learning-strategy", {"source": "demo", "well": "3P", "strategy": "diversity", "n_suggest": 10}, timeout=300)
check("diversity strategy works", al2 is not None and al2.get("strategy") == "diversity")
check("n_suggest=10 works", al2 is not None and al2.get("n_suggest") == 10)

# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.29.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
