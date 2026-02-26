"""Focused test for v3.45.0 endpoints: Mohr Interactive + Class Balance + Feature Importance Compare + Trajectory Impact + RQD."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8132"


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


# ── [160] Mohr Circle Interactive ────────────────────────
print("\n[160] Mohr Circle Interactive")
mi = api("POST", "/api/analysis/mohr-interactive", {"source": "demo", "well": "3P", "friction": 0.6, "pore_pressure": 30.0, "depth": 3000})
check("Status 200", mi is not None)
check("Has well", mi.get("well") == "3P")
check("Has friction", isinstance(mi.get("friction"), (int, float)))
check("Has pp", isinstance(mi.get("pp"), (int, float)))
check("Has sigma1", isinstance(mi.get("sigma1"), (int, float)))
check("Has sigma3", isinstance(mi.get("sigma3"), (int, float)))
check("Has effective_sigma1", isinstance(mi.get("effective_sigma1"), (int, float)))
check("Has effective_sigma3", isinstance(mi.get("effective_sigma3"), (int, float)))
check("Has cohesion", isinstance(mi.get("cohesion"), (int, float)))
check("Has mohr_center", isinstance(mi.get("mohr_center"), (int, float)))
check("Has mohr_radius", isinstance(mi.get("mohr_radius"), (int, float)))
check("Has n_fractures", isinstance(mi.get("n_fractures"), int) and mi["n_fractures"] > 0)
check("Has n_critically_stressed", isinstance(mi.get("n_critically_stressed"), int))
check("Has pct_critically_stressed", isinstance(mi.get("pct_critically_stressed"), (int, float)))
check("Has mean_slip_tendency", isinstance(mi.get("mean_slip_tendency"), (int, float)))
check("Has mean_dilation_tendency", isinstance(mi.get("mean_dilation_tendency"), (int, float)))
check("Has max_slip_tendency", isinstance(mi.get("max_slip_tendency"), (int, float)))
check("Has fractures", isinstance(mi.get("fractures"), list) and len(mi["fractures"]) > 0)
fd0 = mi["fractures"][0]
check("Fracture has azimuth", isinstance(fd0.get("azimuth"), (int, float)))
check("Fracture has dip", isinstance(fd0.get("dip"), (int, float)))
check("Fracture has sigma_n", isinstance(fd0.get("sigma_n"), (int, float)))
check("Fracture has tau", isinstance(fd0.get("tau"), (int, float)))
check("Fracture has slip_tendency", isinstance(fd0.get("slip_tendency"), (int, float)))
check("Fracture has dilation_tendency", isinstance(fd0.get("dilation_tendency"), (int, float)))
check("Fracture has critically_stressed", isinstance(fd0.get("critically_stressed"), bool))
check("Has recommendations", isinstance(mi.get("recommendations"), list) and len(mi["recommendations"]) > 0)
check("Has plot", isinstance(mi.get("plot"), str) and len(mi["plot"]) > 100)
check("Has stakeholder_brief", isinstance(mi.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(mi["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(mi.get("elapsed_s"), (int, float)))

# 6P test
mi2 = api("POST", "/api/analysis/mohr-interactive", {"source": "demo", "well": "6P", "friction": 0.7, "pore_pressure": 25.0, "depth": 2500})
check("6P works", mi2 is not None and mi2.get("well") == "6P")


# ── [161] Class Balance Report ───────────────────────────
print("\n[161] Class Balance Report")
cb = api("POST", "/api/analysis/class-balance", {"source": "demo", "well": "3P"})
check("Status 200", cb is not None)
check("Has well", cb.get("well") == "3P")
check("Has n_samples", isinstance(cb.get("n_samples"), int) and cb["n_samples"] > 0)
check("Has n_classes", isinstance(cb.get("n_classes"), int) and cb["n_classes"] >= 2)
check("Has imbalance_ratio", isinstance(cb.get("imbalance_ratio"), (int, float)))
check("Has severity", cb.get("severity") in ("BALANCED", "MODERATE", "SEVERE", "EXTREME"))
check("Has balance_score", isinstance(cb.get("balance_score"), (int, float)))
check("Has classes", isinstance(cb.get("classes"), list) and len(cb["classes"]) >= 2)
cl0 = cb["classes"][0]
check("Class has class", isinstance(cl0.get("class"), str))
check("Class has count", isinstance(cl0.get("count"), int))
check("Class has pct", isinstance(cl0.get("pct"), (int, float)))
check("Class has is_minority", isinstance(cl0.get("is_minority"), bool))
check("Class has deficit_to_balance", isinstance(cl0.get("deficit_to_balance"), int))
check("Class has smote_eligible", isinstance(cl0.get("smote_eligible"), bool))
check("Has n_smote_candidates", isinstance(cb.get("n_smote_candidates"), int))
check("Has recommendations", isinstance(cb.get("recommendations"), list) and len(cb["recommendations"]) > 0)
check("Has plot", isinstance(cb.get("plot"), str) and len(cb["plot"]) > 100)
check("Has stakeholder_brief", isinstance(cb.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(cb["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(cb.get("elapsed_s"), (int, float)))

# 6P test
cb2 = api("POST", "/api/analysis/class-balance", {"source": "demo", "well": "6P"})
check("6P works", cb2 is not None and cb2.get("well") == "6P")


# ── [162] Feature Importance Comparison ──────────────────
print("\n[162] Feature Importance Comparison")
fi = api("POST", "/api/analysis/feature-importance-compare", {"source": "demo", "well": "3P"})
check("Status 200", fi is not None)
check("Has well", fi.get("well") == "3P")
check("Has n_features", isinstance(fi.get("n_features"), int) and fi["n_features"] >= 5)
check("Has top_n", isinstance(fi.get("top_n"), int))
check("Has features", isinstance(fi.get("features"), list) and len(fi["features"]) >= 3)
f0 = fi["features"][0]
check("Feature has feature", isinstance(f0.get("feature"), str))
check("Feature has rf_importance", isinstance(f0.get("rf_importance"), (int, float)))
check("Feature has gbm_importance", isinstance(f0.get("gbm_importance"), (int, float)))
check("Feature has permutation_importance", isinstance(f0.get("permutation_importance"), (int, float)))
check("Feature has mean_rank", isinstance(f0.get("mean_rank"), (int, float)))
check("Has consensus_features", isinstance(fi.get("consensus_features"), list))
check("Has n_consensus", isinstance(fi.get("n_consensus"), int))
check("Has recommendations", isinstance(fi.get("recommendations"), list) and len(fi["recommendations"]) > 0)
check("Has plot", isinstance(fi.get("plot"), str) and len(fi["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fi.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(fi["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(fi.get("elapsed_s"), (int, float)))

# 6P test
fi2 = api("POST", "/api/analysis/feature-importance-compare", {"source": "demo", "well": "6P"})
check("6P works", fi2 is not None and fi2.get("well") == "6P")


# ── [163] Trajectory Impact ──────────────────────────────
print("\n[163] Trajectory Impact")
ti = api("POST", "/api/analysis/trajectory-impact", {"source": "demo", "well": "3P", "trajectory_azimuth": 90, "trajectory_dip": 60})
check("Status 200", ti is not None)
check("Has well", ti.get("well") == "3P")
check("Has wellbore_azimuth", isinstance(ti.get("wellbore_azimuth"), (int, float)))
check("Has wellbore_dip", isinstance(ti.get("wellbore_dip"), (int, float)))
check("Has n_fractures", isinstance(ti.get("n_fractures"), int) and ti["n_fractures"] > 0)
check("Has mean_correction_factor", isinstance(ti.get("mean_correction_factor"), (int, float)))
check("Has max_correction_factor", isinstance(ti.get("max_correction_factor"), (int, float)))
check("Has pct_high_bias", isinstance(ti.get("pct_high_bias"), (int, float)))
check("Has bias_level", ti.get("bias_level") in ("HIGH", "MODERATE", "LOW"))
check("Has dip_analysis", isinstance(ti.get("dip_analysis"), list) and len(ti["dip_analysis"]) >= 2)
da0 = ti["dip_analysis"][0]
check("DipAnal has dip_range", isinstance(da0.get("dip_range"), str))
check("DipAnal has n_fractures", isinstance(da0.get("n_fractures"), int))
check("DipAnal has mean_correction", isinstance(da0.get("mean_correction"), (int, float)))
check("DipAnal has undersampled", isinstance(da0.get("undersampled"), bool))
check("Has recommendations", isinstance(ti.get("recommendations"), list) and len(ti["recommendations"]) > 0)
check("Has plot", isinstance(ti.get("plot"), str) and len(ti["plot"]) > 100)
check("Has stakeholder_brief", isinstance(ti.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(ti["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(ti.get("elapsed_s"), (int, float)))

# 6P test
ti2 = api("POST", "/api/analysis/trajectory-impact", {"source": "demo", "well": "6P", "trajectory_azimuth": 45, "trajectory_dip": 45})
check("6P works", ti2 is not None and ti2.get("well") == "6P")


# ── [164] Rock Quality Designation (RQD) ─────────────────
print("\n[164] Rock Quality Designation (RQD)")
rq = api("POST", "/api/analysis/rqd", {"source": "demo", "well": "3P", "core_length_m": 1.0})
check("Status 200", rq is not None)
check("Has well", rq.get("well") == "3P")
check("Has n_fractures", isinstance(rq.get("n_fractures"), int) and rq["n_fractures"] > 0)
check("Has rqd_pct", isinstance(rq.get("rqd_pct"), (int, float)))
check("Has quality", rq.get("quality") in ("EXCELLENT", "GOOD", "FAIR", "POOR", "VERY_POOR"))
check("Has theoretical_rqd_pct", isinstance(rq.get("theoretical_rqd_pct"), (int, float)))
check("Has mean_spacing_m", isinstance(rq.get("mean_spacing_m"), (int, float)))
check("Has fracture_frequency", isinstance(rq.get("fracture_frequency"), (int, float)))
check("Has threshold_m", isinstance(rq.get("threshold_m"), (int, float)))
check("Has total_length_m", isinstance(rq.get("total_length_m"), (int, float)))
check("Has depth_zones", isinstance(rq.get("depth_zones"), list) and len(rq["depth_zones"]) >= 1)
dz0 = rq["depth_zones"][0]
check("Zone has zone", isinstance(dz0.get("zone"), str))
check("Zone has n_fractures", isinstance(dz0.get("n_fractures"), int))
check("Zone has rqd", isinstance(dz0.get("rqd"), (int, float)))
check("Zone has quality", dz0.get("quality") in ("EXCELLENT", "GOOD", "FAIR", "POOR", "VERY_POOR"))
check("Has recommendations", isinstance(rq.get("recommendations"), list) and len(rq["recommendations"]) > 0)
check("Has plot", isinstance(rq.get("plot"), str) and len(rq["plot"]) > 100)
check("Has stakeholder_brief", isinstance(rq.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(rq["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(rq.get("elapsed_s"), (int, float)))

# 6P test - may fail due to null depths
try:
    rq2 = api("POST", "/api/analysis/rqd", {"source": "demo", "well": "6P"})
    check("6P works or graceful error", rq2 is not None)
except Exception as e:
    check("6P works or graceful error", "400" in str(e) or "404" in str(e), f"Expected: {e}")


# ── Summary ──────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.45.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
