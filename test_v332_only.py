"""Focused test for v3.32.0 endpoints: Cross-Well Transfer + Reliability Scoring + Executive Dashboard."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8109"


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


# ── [106] Cross-Well Transfer ─────────────────────────
print("\n[106] Cross-Well Transfer Learning")
xfer = api("POST", "/api/analysis/cross-well-transfer", {"source": "demo", "train_well": "3P", "test_well": "6P"}, timeout=300)
check("Status 200", xfer is not None)
check("Has train_well", xfer.get("train_well") == "3P")
check("Has test_well", xfer.get("test_well") == "6P")
check("Has n_train", isinstance(xfer.get("n_train"), int) and xfer["n_train"] > 0)
check("Has n_test", isinstance(xfer.get("n_test"), int) and xfer["n_test"] > 0)
check("Has n_common_features", isinstance(xfer.get("n_common_features"), int) and xfer["n_common_features"] > 3)
check("Has n_classes", isinstance(xfer.get("n_classes"), int) and xfer["n_classes"] >= 2)
check("Has transfer_accuracy", isinstance(xfer.get("transfer_accuracy"), (int, float)))
check("Has transfer_balanced_accuracy", isinstance(xfer.get("transfer_balanced_accuracy"), (int, float)) and xfer["transfer_balanced_accuracy"] > 0)
check("Has baseline_balanced_accuracy", isinstance(xfer.get("baseline_balanced_accuracy"), (int, float)))
check("Has transfer_degradation", isinstance(xfer.get("transfer_degradation"), (int, float)))
check("Has transfer_safe", isinstance(xfer.get("transfer_safe"), bool))

check("Has per_class_transfer", isinstance(xfer.get("per_class_transfer"), list) and len(xfer["per_class_transfer"]) >= 2)
pct0 = xfer["per_class_transfer"][0]
check("PCT has class", isinstance(pct0.get("class"), str))
check("PCT has train_count", isinstance(pct0.get("train_count"), int))
check("PCT has test_count", isinstance(pct0.get("test_count"), int))
check("PCT has f1", isinstance(pct0.get("f1"), (int, float)))
check("PCT has transfer_quality", pct0.get("transfer_quality") in ("GOOD", "FAIR", "POOR"))

check("Has feature_shifts", isinstance(xfer.get("feature_shifts"), list))
if xfer["feature_shifts"]:
    fs0 = xfer["feature_shifts"][0]
    check("FS has feature", isinstance(fs0.get("feature"), str))
    check("FS has normalized_shift", isinstance(fs0.get("normalized_shift"), (int, float)))
    check("FS has severity", fs0.get("severity") in ("HIGH", "MEDIUM", "LOW"))

check("Has recommendations", isinstance(xfer.get("recommendations"), list) and len(xfer["recommendations"]) > 0)
check("Has plot", isinstance(xfer.get("plot"), str) and len(xfer["plot"]) > 100)
check("Has stakeholder_brief", isinstance(xfer.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(xfer["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", xfer["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# Param validation: same well should be rejected
check("Same well rejected", api_expect_error("POST", "/api/analysis/cross-well-transfer", {"source": "demo", "train_well": "3P", "test_well": "3P"}))

# Reverse direction
xfer2 = api("POST", "/api/analysis/cross-well-transfer", {"source": "demo", "train_well": "6P", "test_well": "3P"}, timeout=300)
check("Reverse works", xfer2 is not None and xfer2.get("train_well") == "6P")


# ── [107] Prediction Reliability Scoring ─────────────────
print("\n[107] Prediction Reliability Scoring")
rel = api("POST", "/api/analysis/reliability-scoring", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", rel is not None)
check("Has well", rel.get("well") == "3P")
check("Has n_samples", isinstance(rel.get("n_samples"), int) and rel["n_samples"] > 0)
check("Has mean_reliability", isinstance(rel.get("mean_reliability"), (int, float)) and 0 <= rel["mean_reliability"] <= 100)
check("Has n_green", isinstance(rel.get("n_green"), int))
check("Has n_amber", isinstance(rel.get("n_amber"), int))
check("Has n_red", isinstance(rel.get("n_red"), int))
check("Counts add up", rel["n_green"] + rel["n_amber"] + rel["n_red"] == rel["n_samples"])
check("Has pct_green", isinstance(rel.get("pct_green"), (int, float)))
check("Has pct_amber", isinstance(rel.get("pct_amber"), (int, float)))
check("Has pct_red", isinstance(rel.get("pct_red"), (int, float)))

check("Has signal_weights", isinstance(rel.get("signal_weights"), dict))
sw = rel["signal_weights"]
check("SW has confidence", isinstance(sw.get("confidence"), (int, float)))
check("SW has agreement", isinstance(sw.get("agreement"), (int, float)))

check("Has per_class_reliability", isinstance(rel.get("per_class_reliability"), list) and len(rel["per_class_reliability"]) >= 2)
pcr0 = rel["per_class_reliability"][0]
check("PCR has class", isinstance(pcr0.get("class"), str))
check("PCR has mean_reliability", isinstance(pcr0.get("mean_reliability"), (int, float)))
check("PCR has pct_green", isinstance(pcr0.get("pct_green"), (int, float)))

check("Has sample_scores", isinstance(rel.get("sample_scores"), list) and len(rel["sample_scores"]) > 0)
ss0 = rel["sample_scores"][0]
check("SS has reliability_score", isinstance(ss0.get("reliability_score"), (int, float)) and 0 <= ss0["reliability_score"] <= 100)
check("SS has traffic_light", ss0.get("traffic_light") in ("GREEN", "AMBER", "RED"))
check("SS has signals", isinstance(ss0.get("signals"), dict))
check("SS signals has confidence", isinstance(ss0["signals"].get("confidence"), (int, float)))
check("SS signals has model_agreement", isinstance(ss0["signals"].get("model_agreement"), (int, float)))

check("Has recommendations", isinstance(rel.get("recommendations"), list) and len(rel["recommendations"]) > 0)
check("Has plot", isinstance(rel.get("plot"), str) and len(rel["plot"]) > 100)
check("Has stakeholder_brief", isinstance(rel.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(rel["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", rel["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))


# ── [108] Executive Decision Dashboard ─────────────────
print("\n[108] Executive Decision Dashboard")
ex = api("POST", "/api/analysis/executive-dashboard", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", ex is not None)
check("Has well", ex.get("well") == "3P")
check("Has overall_confidence", ex.get("overall_confidence") in ("HIGH", "MODERATE", "LOW"))
check("Has overall_color", ex.get("overall_color") in ("GREEN", "AMBER", "RED"))
check("Has model_grade", ex.get("model_grade") in ("A", "B", "C", "D"))
check("Has model_summary", isinstance(ex.get("model_summary"), str))
check("Has data_grade", ex.get("data_grade") in ("A", "B", "C", "D"))
check("Has data_summary", isinstance(ex.get("data_summary"), str))

check("Has quick_stats", isinstance(ex.get("quick_stats"), dict))
qs = ex["quick_stats"]
check("QS has n_samples", isinstance(qs.get("n_samples"), int))
check("QS has best_model", isinstance(qs.get("best_model"), str))
check("QS has best_balanced_accuracy", isinstance(qs.get("best_balanced_accuracy"), (int, float)))
check("QS has class_distribution", isinstance(qs.get("class_distribution"), dict))

check("Has risks", isinstance(ex.get("risks"), list) and len(ex["risks"]) > 0)
r0 = ex["risks"][0]
check("Risk has risk", isinstance(r0.get("risk"), str))
check("Risk has severity", r0.get("severity") in ("HIGH", "MEDIUM", "LOW"))
check("Risk has mitigation", isinstance(r0.get("mitigation"), str))

check("Has actions", isinstance(ex.get("actions"), list) and len(ex["actions"]) > 0)
a0 = ex["actions"][0]
check("Action has priority", isinstance(a0.get("priority"), int))
check("Action has action", isinstance(a0.get("action"), str))
check("Action has detail", isinstance(a0.get("detail"), str))
check("Action has timeline", isinstance(a0.get("timeline"), str))

check("Has recommendations", isinstance(ex.get("recommendations"), list) and len(ex["recommendations"]) > 0)
check("Has plot", isinstance(ex.get("plot"), str) and len(ex["plot"]) > 100)
check("Has stakeholder_brief", isinstance(ex.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(ex["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", ex["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(ex["stakeholder_brief"].get("what_this_means"), str))


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.32.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
