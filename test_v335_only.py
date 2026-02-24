"""Focused test for v3.35.0 endpoints: Fracture Connectivity + Failure Prediction + Batch Analysis."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8112"


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


# ── [114] Fracture Network Connectivity ─────────────────
print("\n[114] Fracture Network Connectivity")
fc = api("POST", "/api/analysis/fracture-connectivity", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", fc is not None)
check("Has well", fc.get("well") == "3P")
check("Has n_fractures", isinstance(fc.get("n_fractures"), int) and fc["n_fractures"] > 0)
check("Has n_sets", isinstance(fc.get("n_sets"), int) and fc["n_sets"] >= 2)

check("Has fracture_sets", isinstance(fc.get("fracture_sets"), list) and len(fc["fracture_sets"]) >= 2)
fs0 = fc["fracture_sets"][0]
check("FS has set_id", isinstance(fs0.get("set_id"), int))
check("FS has n_fractures", isinstance(fs0.get("n_fractures"), int) and fs0["n_fractures"] > 0)
check("FS has pct_of_total", isinstance(fs0.get("pct_of_total"), (int, float)))
check("FS has mean_azimuth", isinstance(fs0.get("mean_azimuth"), (int, float)) and 0 <= fs0["mean_azimuth"] <= 360)
check("FS has mean_dip", isinstance(fs0.get("mean_dip"), (int, float)) and 0 <= fs0["mean_dip"] <= 90)
check("FS has azimuth_std", isinstance(fs0.get("azimuth_std"), (int, float)))
check("FS has dip_std", isinstance(fs0.get("dip_std"), (int, float)))
check("FS has depth_range_m", isinstance(fs0.get("depth_range_m"), dict))
check("FS has mean_spacing_m", isinstance(fs0.get("mean_spacing_m"), (int, float)))
check("FS has fracture_types", isinstance(fs0.get("fracture_types"), dict))
check("FS has dominant_type", isinstance(fs0.get("dominant_type"), str))

# Total fractures across sets = n_fractures
total_set_fracs = sum(fs["n_fractures"] for fs in fc["fracture_sets"])
check("Sets account for all fractures", total_set_fracs == fc["n_fractures"])

check("Has connectivity_matrix", isinstance(fc.get("connectivity_matrix"), list))
if fc["connectivity_matrix"]:
    cm0 = fc["connectivity_matrix"][0]
    check("CM has set_a", isinstance(cm0.get("set_a"), int))
    check("CM has set_b", isinstance(cm0.get("set_b"), int))
    check("CM has intersection_angle", isinstance(cm0.get("intersection_angle"), (int, float)))
    check("CM has connectivity_score", isinstance(cm0.get("connectivity_score"), (int, float)) and 0 <= cm0["connectivity_score"] <= 1)
    check("CM has connectivity_level", cm0.get("connectivity_level") in ("HIGH", "MEDIUM", "LOW"))
    check("CM has spatial_overlap_m", isinstance(cm0.get("spatial_overlap_m"), (int, float)))

check("Has network_quality", fc.get("network_quality") in ("WELL-CONNECTED", "MODERATELY-CONNECTED", "POORLY-CONNECTED"))
check("Has network_color", fc.get("network_color") in ("GREEN", "AMBER", "RED"))
check("Has dominant_flow_azimuth", isinstance(fc.get("dominant_flow_azimuth"), (int, float)) and 0 <= fc["dominant_flow_azimuth"] <= 360)
check("Has anisotropy_ratio", isinstance(fc.get("anisotropy_ratio"), (int, float)) and 0 <= fc["anisotropy_ratio"] <= 1)

check("Has recommendations", isinstance(fc.get("recommendations"), list) and len(fc["recommendations"]) > 0)
check("Has plot", isinstance(fc.get("plot"), str) and len(fc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fc.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(fc["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", fc["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(fc["stakeholder_brief"].get("what_this_means"), str))

# 6P test
fc2 = api("POST", "/api/analysis/fracture-connectivity", {"source": "demo", "well": "6P"}, timeout=300)
check("6P works", fc2 is not None and fc2.get("well") == "6P")


# ── [115] Wellbore Failure Prediction ─────────────────────
print("\n[115] Wellbore Failure Prediction")
fp = api("POST", "/api/analysis/failure-prediction", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", fp is not None)
check("Has well", fp.get("well") == "3P")
check("Has mud_weight_ppg", isinstance(fp.get("mud_weight_ppg"), (int, float)))
check("Has depth_range_m", isinstance(fp.get("depth_range_m"), dict))
check("Has n_depth_intervals", isinstance(fp.get("n_depth_intervals"), int) and fp["n_depth_intervals"] > 0)
check("Has n_high_risk", isinstance(fp.get("n_high_risk"), int) and fp["n_high_risk"] >= 0)
check("Has n_medium_risk", isinstance(fp.get("n_medium_risk"), int) and fp["n_medium_risk"] >= 0)
check("Has n_low_risk", isinstance(fp.get("n_low_risk"), int) and fp["n_low_risk"] >= 0)
check("Risk counts sum", fp["n_high_risk"] + fp["n_medium_risk"] + fp["n_low_risk"] == fp["n_depth_intervals"])
check("Has n_risk_zones", isinstance(fp.get("n_risk_zones"), int))

check("Has mud_weight_window", isinstance(fp.get("mud_weight_window"), dict))
mww = fp["mud_weight_window"]
check("MWW has min_safe_MPa", isinstance(mww.get("min_safe_MPa"), (int, float)))
check("MWW has max_safe_MPa", isinstance(mww.get("max_safe_MPa"), (int, float)))
check("MWW has window_MPa", isinstance(mww.get("window_MPa"), (int, float)))
check("MWW has status", mww.get("status") in ("SAFE", "NARROW", "CRITICAL"))

check("Has failure_profile", isinstance(fp.get("failure_profile"), list) and len(fp["failure_profile"]) > 0)
fp0 = fp["failure_profile"][0]
check("FP has depth_m", isinstance(fp0.get("depth_m"), (int, float)))
check("FP has Sv_MPa", isinstance(fp0.get("Sv_MPa"), (int, float)))
check("FP has SH_MPa", isinstance(fp0.get("SH_MPa"), (int, float)))
check("FP has Sh_MPa", isinstance(fp0.get("Sh_MPa"), (int, float)))
check("FP has Pp_MPa", isinstance(fp0.get("Pp_MPa"), (int, float)))
check("FP has breakout_risk", isinstance(fp0.get("breakout_risk"), (int, float)) and 0 <= fp0["breakout_risk"] <= 1)
check("FP has tensile_risk", isinstance(fp0.get("tensile_risk"), (int, float)) and 0 <= fp0["tensile_risk"] <= 1)
check("FP has overall_risk", isinstance(fp0.get("overall_risk"), (int, float)) and 0 <= fp0["overall_risk"] <= 1)
check("FP has risk_level", fp0.get("risk_level") in ("HIGH", "MEDIUM", "LOW"))
check("FP has dominant_failure_type", fp0.get("dominant_failure_type") in ("BREAKOUT", "TENSILE"))
check("FP has nearby_fracture_count", isinstance(fp0.get("nearby_fracture_count"), int))

check("Has recommendations", isinstance(fp.get("recommendations"), list) and len(fp["recommendations"]) > 0)
check("Has plot", isinstance(fp.get("plot"), str) and len(fp["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fp.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(fp["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", fp["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# Param validation
check("mw=3 rejected", api_expect_error("POST", "/api/analysis/failure-prediction", {"source": "demo", "well": "3P", "mud_weight_ppg": 3.0}))
check("mw=25 rejected", api_expect_error("POST", "/api/analysis/failure-prediction", {"source": "demo", "well": "3P", "mud_weight_ppg": 25.0}))

# Custom mud weight
fp2 = api("POST", "/api/analysis/failure-prediction", {"source": "demo", "well": "3P", "mud_weight_ppg": 12.0}, timeout=300)
check("mw=12 works", fp2 is not None and fp2.get("mud_weight_ppg") == 12.0)


# ── [116] Batch Multi-Well Analysis ─────────────────────
print("\n[116] Batch Multi-Well Analysis")
ba = api("POST", "/api/analysis/batch-analysis", {"source": "demo"}, timeout=300)
check("Status 200", ba is not None)
check("Has n_wells", isinstance(ba.get("n_wells"), int) and ba["n_wells"] >= 2)
check("Has n_wells_analyzed", isinstance(ba.get("n_wells_analyzed"), int) and ba["n_wells_analyzed"] >= 2)

check("Has well_results", isinstance(ba.get("well_results"), list) and len(ba["well_results"]) >= 2)
wr0 = ba["well_results"][0]
check("WR has well", isinstance(wr0.get("well"), str))
check("WR has n_samples", isinstance(wr0.get("n_samples"), int) and wr0["n_samples"] > 0)
check("WR has best_model", isinstance(wr0.get("best_model"), str))
check("WR has best_balanced_accuracy", isinstance(wr0.get("best_balanced_accuracy"), (int, float)))
check("WR has best_accuracy", isinstance(wr0.get("best_accuracy"), (int, float)))
check("WR has n_classes", isinstance(wr0.get("n_classes"), int))
check("WR has class_distribution", isinstance(wr0.get("class_distribution"), dict))
check("WR has imbalance_ratio", isinstance(wr0.get("imbalance_ratio"), (int, float)))
check("WR has data_quality", wr0.get("data_quality") in ("GOOD", "FAIR", "POOR"))
check("WR has model_quality", wr0.get("model_quality") in ("GOOD", "FAIR", "POOR"))

check("Has field_summary", isinstance(ba.get("field_summary"), dict))
fs = ba["field_summary"]
check("FS has n_wells", isinstance(fs.get("n_wells"), int))
check("FS has total_samples", isinstance(fs.get("total_samples"), int))
check("FS has avg_balanced_accuracy", isinstance(fs.get("avg_balanced_accuracy"), (int, float)))
check("FS has best_well", isinstance(fs.get("best_well"), str))
check("FS has best_accuracy", isinstance(fs.get("best_accuracy"), (int, float)))
check("FS has worst_well", isinstance(fs.get("worst_well"), str))
check("FS has worst_accuracy", isinstance(fs.get("worst_accuracy"), (int, float)))
check("FS has accuracy_spread", isinstance(fs.get("accuracy_spread"), (int, float)))

check("Has recommendations", isinstance(ba.get("recommendations"), list) and len(ba["recommendations"]) > 0)
check("Has plot", isinstance(ba.get("plot"), str) and len(ba["plot"]) > 100)
check("Has stakeholder_brief", isinstance(ba.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(ba["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", ba["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(ba["stakeholder_brief"].get("what_this_means"), str))


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.35.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
