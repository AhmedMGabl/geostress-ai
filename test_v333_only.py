"""Focused test for v3.33.0 endpoints: Deployment Readiness + Sensitivity Analysis."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8110"


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


# ── [109] Deployment Readiness Checklist ─────────────────
print("\n[109] Deployment Readiness Checklist")
dr = api("POST", "/api/analysis/deployment-readiness", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", dr is not None)
check("Has well", dr.get("well") == "3P")
check("Has decision", dr.get("decision") in ("GO", "CONDITIONAL GO", "NO-GO"))
check("Has decision_color", dr.get("decision_color") in ("GREEN", "AMBER", "RED"))
check("Has n_checks", isinstance(dr.get("n_checks"), int) and dr["n_checks"] == 10)
check("Has n_passed", isinstance(dr.get("n_passed"), int) and dr["n_passed"] >= 0)
check("Has n_failed", isinstance(dr.get("n_failed"), int) and dr["n_failed"] >= 0)
check("Counts add up", dr["n_passed"] + dr["n_failed"] == dr["n_checks"])

check("Has checks list", isinstance(dr.get("checks"), list) and len(dr["checks"]) == 10)
c0 = dr["checks"][0]
check("Check has check name", isinstance(c0.get("check"), str))
check("Check has requirement", isinstance(c0.get("requirement"), str))
check("Check has actual", isinstance(c0.get("actual"), str))
check("Check has pass", isinstance(c0.get("pass"), bool))
check("Check has category", c0.get("category") in ("Data", "Model", "Process"))
check("Check has detail", isinstance(c0.get("detail"), str) and len(c0["detail"]) > 10)

# Verify all 10 checks are named
check_names = [c["check"] for c in dr["checks"]]
check("Has Sample Size check", "Minimum Sample Size" in check_names)
check("Has Class Representation check", "Class Representation" in check_names)
check("Has Class Imbalance check", "Class Imbalance" in check_names)
check("Has Model Accuracy check", "Model Accuracy" in check_names)
check("Has Model Consistency check", "Model Consistency" in check_names)
check("Has No Failed Class check", "No Failed Class" in check_names)
check("Has Confidence Calibration check", "Confidence Calibration" in check_names)
check("Has Feature-Sample Ratio check", "Feature-Sample Ratio" in check_names)
check("Has Expert Review check", "Expert Review" in check_names)
check("Has Cross-Well Data check", "Cross-Well Data" in check_names)

check("Has failed_checks list", isinstance(dr.get("failed_checks"), list))
check("Has recommendations", isinstance(dr.get("recommendations"), list) and len(dr["recommendations"]) > 0)
check("Has plot", isinstance(dr.get("plot"), str) and len(dr["plot"]) > 100)
check("Has stakeholder_brief", isinstance(dr.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(dr["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", dr["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(dr["stakeholder_brief"].get("what_this_means"), str))
check("Brief has recommendation", isinstance(dr["stakeholder_brief"].get("recommendation"), str))

# Test with different well
dr2 = api("POST", "/api/analysis/deployment-readiness", {"source": "demo", "well": "6P"}, timeout=300)
check("6P works", dr2 is not None and dr2.get("well") == "6P")
check("6P has decision", dr2.get("decision") in ("GO", "CONDITIONAL GO", "NO-GO"))


# ── [110] Sensitivity Analysis ─────────────────────────
print("\n[110] Sensitivity Analysis")
sa = api("POST", "/api/analysis/sensitivity-analysis", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", sa is not None)
check("Has well", sa.get("well") == "3P")
check("Has n_samples", isinstance(sa.get("n_samples"), int) and sa["n_samples"] > 0)
check("Has n_perturbations", isinstance(sa.get("n_perturbations"), int) and sa["n_perturbations"] > 0)
check("Has uncertainties", isinstance(sa.get("uncertainties"), dict))
check("Uncertainty has azimuth", isinstance(sa["uncertainties"].get("azimuth_deg"), (int, float)))
check("Uncertainty has dip", isinstance(sa["uncertainties"].get("dip_deg"), (int, float)))
check("Uncertainty has depth", isinstance(sa["uncertainties"].get("depth_m"), (int, float)))

check("Has mean_flip_rate", isinstance(sa.get("mean_flip_rate"), (int, float)) and 0 <= sa["mean_flip_rate"] <= 1)
check("Has n_stable", isinstance(sa.get("n_stable"), int) and sa["n_stable"] >= 0)
check("Has n_moderate", isinstance(sa.get("n_moderate"), int) and sa["n_moderate"] >= 0)
check("Has n_highly_sensitive", isinstance(sa.get("n_highly_sensitive"), int) and sa["n_highly_sensitive"] >= 0)
check("Counts add up", sa["n_stable"] + sa["n_moderate"] + sa["n_highly_sensitive"] == sa["n_samples"])
check("Has pct_stable", isinstance(sa.get("pct_stable"), (int, float)))
check("Has pct_sensitive", isinstance(sa.get("pct_sensitive"), (int, float)))

check("Has per_class_sensitivity", isinstance(sa.get("per_class_sensitivity"), list) and len(sa["per_class_sensitivity"]) >= 2)
pcs0 = sa["per_class_sensitivity"][0]
check("PCS has class", isinstance(pcs0.get("class"), str))
check("PCS has n_predicted", isinstance(pcs0.get("n_predicted"), int))
check("PCS has mean_flip_rate", isinstance(pcs0.get("mean_flip_rate"), (int, float)))
check("PCS has stability", isinstance(pcs0.get("stability"), (int, float)))
check("PCS has n_highly_sensitive", isinstance(pcs0.get("n_highly_sensitive"), int))
check("PCS has sensitivity_level", pcs0.get("sensitivity_level") in ("HIGH", "MEDIUM", "LOW"))

check("Has sample_sensitivity", isinstance(sa.get("sample_sensitivity"), list) and len(sa["sample_sensitivity"]) > 0)
ss0 = sa["sample_sensitivity"][0]
check("SS has index", isinstance(ss0.get("index"), int))
check("SS has true_class", isinstance(ss0.get("true_class"), str))
check("SS has base_prediction", isinstance(ss0.get("base_prediction"), str))
check("SS has flip_rate", isinstance(ss0.get("flip_rate"), (int, float)) and 0 <= ss0["flip_rate"] <= 1)
check("SS has stability", isinstance(ss0.get("stability"), (int, float)) and 0 <= ss0["stability"] <= 1)
check("SS has max_prob_change", isinstance(ss0.get("max_prob_change"), (int, float)))
check("SS has sensitivity_level", ss0.get("sensitivity_level") in ("HIGH", "MEDIUM", "LOW"))

check("Has feature_sensitivity", isinstance(sa.get("feature_sensitivity"), list) and len(sa["feature_sensitivity"]) == 3)
fs0 = sa["feature_sensitivity"][0]
check("FS has feature", isinstance(fs0.get("feature"), str))
check("FS has uncertainty", isinstance(fs0.get("uncertainty"), (int, float)))
check("FS has unit", isinstance(fs0.get("unit"), str))

check("Has recommendations", isinstance(sa.get("recommendations"), list) and len(sa["recommendations"]) > 0)
r0 = sa["recommendations"][0]
check("Rec has priority", r0.get("priority") in ("CRITICAL", "HIGH", "MEDIUM", "LOW"))
check("Rec has category", isinstance(r0.get("category"), str))
check("Rec has action", isinstance(r0.get("action"), str))
check("Rec has impact", isinstance(r0.get("impact"), str))

check("Has plot", isinstance(sa.get("plot"), str) and len(sa["plot"]) > 100)
check("Has stakeholder_brief", isinstance(sa.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(sa["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", sa["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(sa["stakeholder_brief"].get("what_this_means"), str))

# Param validation
check("n_perturbations=5 rejected", api_expect_error("POST", "/api/analysis/sensitivity-analysis", {"source": "demo", "well": "3P", "n_perturbations": 5}))
check("n_perturbations=300 rejected", api_expect_error("POST", "/api/analysis/sensitivity-analysis", {"source": "demo", "well": "3P", "n_perturbations": 300}))

# Custom n_perturbations
sa2 = api("POST", "/api/analysis/sensitivity-analysis", {"source": "demo", "well": "3P", "n_perturbations": 20}, timeout=300)
check("n_perturbations=20 works", sa2 is not None and sa2.get("n_perturbations") >= 10)


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.33.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
