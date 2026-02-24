"""Focused test for v3.34.0 endpoints: Model Monitoring + Formation Boundaries + Full Report."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8111"


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


# ── [111] Model Monitoring & Drift Detection ─────────────────
print("\n[111] Model Monitoring & Drift Detection")
mm = api("POST", "/api/analysis/model-monitoring", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", mm is not None)
check("Has well", mm.get("well") == "3P")
check("Has reference_well", isinstance(mm.get("reference_well"), str))
check("Has n_reference", isinstance(mm.get("n_reference"), int) and mm["n_reference"] > 0)
check("Has n_new", isinstance(mm.get("n_new"), int) and mm["n_new"] > 0)
check("Has drift_status", mm.get("drift_status") in ("STABLE", "WARNING", "CRITICAL"))
check("Has drift_color", mm.get("drift_color") in ("GREEN", "AMBER", "RED"))
check("Has avg_psi", isinstance(mm.get("avg_psi"), (int, float)) and mm["avg_psi"] >= 0)
check("Has n_high_drift_features", isinstance(mm.get("n_high_drift_features"), int))
check("Has n_medium_drift_features", isinstance(mm.get("n_medium_drift_features"), int))

check("Has feature_drift", isinstance(mm.get("feature_drift"), list) and len(mm["feature_drift"]) > 0)
fd0 = mm["feature_drift"][0]
check("FD has feature", isinstance(fd0.get("feature"), str))
check("FD has psi", isinstance(fd0.get("psi"), (int, float)) and fd0["psi"] >= 0)
check("FD has severity", fd0.get("severity") in ("HIGH", "MEDIUM", "LOW"))
check("FD has ref_mean", isinstance(fd0.get("ref_mean"), (int, float)))
check("FD has new_mean", isinstance(fd0.get("new_mean"), (int, float)))
check("FD has mean_shift", isinstance(fd0.get("mean_shift"), (int, float)))
check("FD has ref_std", isinstance(fd0.get("ref_std"), (int, float)))
check("FD has new_std", isinstance(fd0.get("new_std"), (int, float)))

check("Has class_drift", isinstance(mm.get("class_drift"), list) and len(mm["class_drift"]) >= 2)
cd0 = mm["class_drift"][0]
check("CD has class", isinstance(cd0.get("class"), str))
check("CD has ref_pct", isinstance(cd0.get("ref_pct"), (int, float)))
check("CD has new_pct", isinstance(cd0.get("new_pct"), (int, float)))
check("CD has shift_pct", isinstance(cd0.get("shift_pct"), (int, float)))
check("CD has severity", cd0.get("severity") in ("HIGH", "MEDIUM", "LOW"))

check("Has model_accuracy", isinstance(mm.get("model_accuracy"), (int, float)))
check("Has recommendations", isinstance(mm.get("recommendations"), list) and len(mm["recommendations"]) > 0)
r0 = mm["recommendations"][0]
check("Rec has priority", r0.get("priority") in ("CRITICAL", "HIGH", "MEDIUM", "LOW"))
check("Rec has category", isinstance(r0.get("category"), str))
check("Rec has action", isinstance(r0.get("action"), str))
check("Rec has impact", isinstance(r0.get("impact"), str))

check("Has plot", isinstance(mm.get("plot"), str) and len(mm["plot"]) > 100)
check("Has stakeholder_brief", isinstance(mm.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(mm["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", mm["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(mm["stakeholder_brief"].get("what_this_means"), str))
check("Brief has recommendation", isinstance(mm["stakeholder_brief"].get("recommendation"), str))

# Cross-well monitoring
mm2 = api("POST", "/api/analysis/model-monitoring", {"source": "demo", "well": "3P", "reference_well": "6P"}, timeout=300)
check("Cross-well monitoring works", mm2 is not None and mm2.get("well") == "3P")
check("Cross-well has reference", "6P" in str(mm2.get("reference_well", "")))


# ── [112] Formation Boundary Detection ─────────────────────
print("\n[112] Formation Boundary Detection")
fb = api("POST", "/api/analysis/formation-boundaries", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", fb is not None)
check("Has well", fb.get("well") == "3P")
check("Has depth_range_m", isinstance(fb.get("depth_range_m"), dict))
check("Depth has min", isinstance(fb["depth_range_m"].get("min"), (int, float)))
check("Depth has max", isinstance(fb["depth_range_m"].get("max"), (int, float)))
check("Has n_fractures", isinstance(fb.get("n_fractures"), int) and fb["n_fractures"] > 0)
check("Has n_boundaries", isinstance(fb.get("n_boundaries"), int) and fb["n_boundaries"] >= 0)
check("Has boundaries list", isinstance(fb.get("boundaries"), list))

if fb["boundaries"]:
    b0 = fb["boundaries"][0]
    check("B has depth_m", isinstance(b0.get("depth_m"), (int, float)))
    check("B has confidence", b0.get("confidence") in ("HIGH", "MEDIUM", "LOW"))
    check("B has z_score", isinstance(b0.get("z_score"), (int, float)))
    check("B has signals_detected", isinstance(b0.get("signals_detected"), list) and len(b0["signals_detected"]) > 0)
    check("B has n_signals", isinstance(b0.get("n_signals"), int) and b0["n_signals"] > 0)
    check("B has above_mean_dip", isinstance(b0.get("above_mean_dip"), (int, float)))
    check("B has below_mean_dip", isinstance(b0.get("below_mean_dip"), (int, float)))
    check("B has dip_change_deg", isinstance(b0.get("dip_change_deg"), (int, float)))
    check("B has above_class_distribution", isinstance(b0.get("above_class_distribution"), dict))
    check("B has below_class_distribution", isinstance(b0.get("below_class_distribution"), dict))

check("Has n_segments", isinstance(fb.get("n_segments"), int) and fb["n_segments"] >= 1)
check("Has segments list", isinstance(fb.get("segments"), list) and len(fb["segments"]) >= 1)
s0 = fb["segments"][0]
check("S has segment", isinstance(s0.get("segment"), int))
check("S has top_m", isinstance(s0.get("top_m"), (int, float)))
check("S has bottom_m", isinstance(s0.get("bottom_m"), (int, float)))
check("S has thickness_m", isinstance(s0.get("thickness_m"), (int, float)))
check("S has n_fractures", isinstance(s0.get("n_fractures"), int))
check("S has mean_dip", isinstance(s0.get("mean_dip"), (int, float)))
check("S has std_dip", isinstance(s0.get("std_dip"), (int, float)))
check("S has dominant_fracture_type", isinstance(s0.get("dominant_fracture_type"), str))
check("S has fracture_density_per_m", isinstance(s0.get("fracture_density_per_m"), (int, float)))

check("Has recommendations", isinstance(fb.get("recommendations"), list) and len(fb["recommendations"]) > 0)
check("Has plot", isinstance(fb.get("plot"), str) and len(fb["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fb.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(fb["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", fb["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(fb["stakeholder_brief"].get("what_this_means"), str))

# Param validation
check("min_seg=1 rejected", api_expect_error("POST", "/api/analysis/formation-boundaries", {"source": "demo", "well": "3P", "min_segment_size": 1}))
check("min_seg=100 rejected", api_expect_error("POST", "/api/analysis/formation-boundaries", {"source": "demo", "well": "3P", "min_segment_size": 100}))

# 6P test
fb2 = api("POST", "/api/analysis/formation-boundaries", {"source": "demo", "well": "6P"}, timeout=300)
check("6P works", fb2 is not None and fb2.get("well") == "6P")


# ── [113] Comprehensive Analysis Report ─────────────────────
print("\n[113] Comprehensive Analysis Report")
rr = api("POST", "/api/reports/full-report", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", rr is not None)
check("Has well", rr.get("well") == "3P")
check("Has report_type", rr.get("report_type") == "comprehensive")
check("Has overall_assessment", rr.get("overall_assessment") in ("STRONG", "ADEQUATE", "INSUFFICIENT"))
check("Has overall_color", rr.get("overall_color") in ("GREEN", "AMBER", "RED"))
check("Has overall_summary", isinstance(rr.get("overall_summary"), str))

check("Has data_summary", isinstance(rr.get("data_summary"), dict))
ds = rr["data_summary"]
check("DS has n_samples", isinstance(ds.get("n_samples"), int) and ds["n_samples"] > 0)
check("DS has n_classes", isinstance(ds.get("n_classes"), int) and ds["n_classes"] >= 2)
check("DS has class_names", isinstance(ds.get("class_names"), list))
check("DS has class_distribution", isinstance(ds.get("class_distribution"), dict))
check("DS has n_features", isinstance(ds.get("n_features"), int))
check("DS has depth_range_m", isinstance(ds.get("depth_range_m"), dict))

check("Has model_summary", isinstance(rr.get("model_summary"), dict))
ms = rr["model_summary"]
check("MS has n_models_evaluated", isinstance(ms.get("n_models_evaluated"), int) and ms["n_models_evaluated"] >= 4)
check("MS has best_model", isinstance(ms.get("best_model"), str))
check("MS has best_balanced_accuracy", isinstance(ms.get("best_balanced_accuracy"), (int, float)))
check("MS has best_accuracy", isinstance(ms.get("best_accuracy"), (int, float)))
check("MS has all_models", isinstance(ms.get("all_models"), list) and len(ms["all_models"]) >= 4)
am0 = ms["all_models"][0]
check("AM has model", isinstance(am0.get("model"), str))
check("AM has balanced_accuracy", isinstance(am0.get("balanced_accuracy"), (int, float)))
check("AM has accuracy", isinstance(am0.get("accuracy"), (int, float)))

check("Has risks", isinstance(rr.get("risks"), list) and len(rr["risks"]) > 0)
rk0 = rr["risks"][0]
check("Risk has risk", isinstance(rk0.get("risk"), str))
check("Risk has severity", rk0.get("severity") in ("HIGH", "MEDIUM", "LOW"))
check("Risk has detail", isinstance(rk0.get("detail"), str))
check("Risk has mitigation", isinstance(rk0.get("mitigation"), str))

check("Has recommendations", isinstance(rr.get("recommendations"), list) and len(rr["recommendations"]) > 0)
rc0 = rr["recommendations"][0]
check("Rec has priority", isinstance(rc0.get("priority"), int))
check("Rec has action", isinstance(rc0.get("action"), str))
check("Rec has detail", isinstance(rc0.get("detail"), str))
check("Rec has timeline", isinstance(rc0.get("timeline"), str))

check("Has plot", isinstance(rr.get("plot"), str) and len(rr["plot"]) > 100)
check("Has stakeholder_brief", isinstance(rr.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(rr["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", rr["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(rr["stakeholder_brief"].get("what_this_means"), str))
check("Brief has recommendation", isinstance(rr["stakeholder_brief"].get("recommendation"), str))

# 6P report
rr2 = api("POST", "/api/reports/full-report", {"source": "demo", "well": "6P"}, timeout=300)
check("6P works", rr2 is not None and rr2.get("well") == "6P")
check("6P has assessment", rr2.get("overall_assessment") in ("STRONG", "ADEQUATE", "INSUFFICIENT"))


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.34.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
