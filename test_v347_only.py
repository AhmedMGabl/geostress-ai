"""Focused test for v3.47.0 endpoints: Slip Tendency Map + Formation Pressure + Fracture Sets + Breakout Prediction + Data Completeness."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8134"


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


# ── [170] Slip Tendency Map ──────────────────────────────
print("\n[170] Slip Tendency Map")
st = api("POST", "/api/analysis/slip-tendency-map", {"source": "demo", "well": "3P", "friction": 0.6, "depth": 3000})
check("Status 200", st is not None)
check("Has well", st.get("well") == "3P")
check("Has depth_m", isinstance(st.get("depth_m"), (int, float)))
check("Has friction", isinstance(st.get("friction"), (int, float)))
check("Has sigma1_MPa", isinstance(st.get("sigma1_MPa"), (int, float)))
check("Has sigma3_MPa", isinstance(st.get("sigma3_MPa"), (int, float)))
check("Has Pp_MPa", isinstance(st.get("Pp_MPa"), (int, float)))
check("Has grid_size", isinstance(st.get("grid_size"), int))
check("Has max_slip_tendency", isinstance(st.get("max_slip_tendency"), (int, float)))
check("Has mean_slip_tendency", isinstance(st.get("mean_slip_tendency"), (int, float)))
check("Has max_dilation_tendency", isinstance(st.get("max_dilation_tendency"), (int, float)))
check("Has mean_dilation_tendency", isinstance(st.get("mean_dilation_tendency"), (int, float)))
check("Has critical_orientation", isinstance(st.get("critical_orientation"), dict))
co = st["critical_orientation"]
check("Critical has azimuth_deg", isinstance(co.get("azimuth_deg"), (int, float)))
check("Critical has dip_deg", isinstance(co.get("dip_deg"), (int, float)))
check("Critical has slip_tendency", isinstance(co.get("slip_tendency"), (int, float)))
check("Has n_fractures", isinstance(st.get("n_fractures"), int) and st["n_fractures"] > 0)
check("Has n_critically_stressed", isinstance(st.get("n_critically_stressed"), int))
check("Has pct_critically_stressed", isinstance(st.get("pct_critically_stressed"), (int, float)))
check("Has recommendations", isinstance(st.get("recommendations"), list) and len(st["recommendations"]) > 0)
check("Has plot", isinstance(st.get("plot"), str) and len(st["plot"]) > 100)
check("Has stakeholder_brief", isinstance(st.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(st["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(st.get("elapsed_s"), (int, float)))

# 6P test
st2 = api("POST", "/api/analysis/slip-tendency-map", {"source": "demo", "well": "6P"})
check("6P works", st2 is not None and st2.get("well") == "6P")


# ── [171] Formation Pressure ─────────────────────────────
print("\n[171] Formation Pressure")
fp = api("POST", "/api/analysis/formation-pressure", {"source": "demo", "well": "3P", "depth_start": 500, "depth_end": 5000, "n_points": 20})
check("Status 200", fp is not None)
check("Has well", fp.get("well") == "3P")
check("Has depth_range_m", isinstance(fp.get("depth_range_m"), list) and len(fp["depth_range_m"]) == 2)
check("Has n_points", isinstance(fp.get("n_points"), int))
check("Has pp_gradient_MPa_per_km", isinstance(fp.get("pp_gradient_MPa_per_km"), (int, float)))
check("Has sv_gradient_MPa_per_km", isinstance(fp.get("sv_gradient_MPa_per_km"), (int, float)))
check("Has fracture_gradient_MPa_per_km", isinstance(fp.get("fracture_gradient_MPa_per_km"), (int, float)))
check("Has pressure_regime", fp.get("pressure_regime") in ("HYDROSTATIC", "OVERPRESSURED", "UNDERPRESSURED"))
check("Has pp_sv_ratio", isinstance(fp.get("pp_sv_ratio"), (int, float)))
check("Has data_coverage_pct", isinstance(fp.get("data_coverage_pct"), (int, float)))
check("Has profile", isinstance(fp.get("profile"), list) and len(fp["profile"]) >= 2)
p0 = fp["profile"][0]
check("Profile has depth_m", isinstance(p0.get("depth_m"), (int, float)))
check("Profile has Sv_MPa", isinstance(p0.get("Sv_MPa"), (int, float)))
check("Profile has Pp_MPa", isinstance(p0.get("Pp_MPa"), (int, float)))
check("Profile has Fg_MPa", isinstance(p0.get("Fg_MPa"), (int, float)))
check("Profile has effective_stress_MPa", isinstance(p0.get("effective_stress_MPa"), (int, float)))
check("Profile has Sv_SG", isinstance(p0.get("Sv_SG"), (int, float)))
check("Has recommendations", isinstance(fp.get("recommendations"), list) and len(fp["recommendations"]) > 0)
check("Has plot", isinstance(fp.get("plot"), str) and len(fp["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fp.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(fp["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(fp.get("elapsed_s"), (int, float)))

# 6P test
fp2 = api("POST", "/api/analysis/formation-pressure", {"source": "demo", "well": "6P"})
check("6P works", fp2 is not None and fp2.get("well") == "6P")


# ── [172] Fracture Sets ──────────────────────────────────
print("\n[172] Fracture Sets")
fs = api("POST", "/api/analysis/fracture-sets", {"source": "demo", "well": "3P", "max_sets": 5})
check("Status 200", fs is not None)
check("Has well", fs.get("well") == "3P")
check("Has n_fractures", isinstance(fs.get("n_fractures"), int) and fs["n_fractures"] > 0)
check("Has n_sets", isinstance(fs.get("n_sets"), int) and fs["n_sets"] >= 2)
check("Has best_silhouette", isinstance(fs.get("best_silhouette"), (int, float)))
check("Has k_analysis", isinstance(fs.get("k_analysis"), list) and len(fs["k_analysis"]) >= 1)
ka0 = fs["k_analysis"][0]
check("k_analysis has k", isinstance(ka0.get("k"), int))
check("k_analysis has silhouette", isinstance(ka0.get("silhouette"), (int, float)))
check("Has sets", isinstance(fs.get("sets"), list) and len(fs["sets"]) >= 2)
s0 = fs["sets"][0]
check("Set has set_id", isinstance(s0.get("set_id"), int))
check("Set has n_fractures", isinstance(s0.get("n_fractures"), int))
check("Set has pct", isinstance(s0.get("pct"), (int, float)))
check("Set has mean_azimuth_deg", isinstance(s0.get("mean_azimuth_deg"), (int, float)))
check("Set has mean_dip_deg", isinstance(s0.get("mean_dip_deg"), (int, float)))
check("Set has dip_class", isinstance(s0.get("dip_class"), str))
check("Set has strike_deg", isinstance(s0.get("strike_deg"), (int, float)))
check("Has conjugate_pairs", isinstance(fs.get("conjugate_pairs"), list))
check("Has recommendations", isinstance(fs.get("recommendations"), list) and len(fs["recommendations"]) > 0)
check("Has plot", isinstance(fs.get("plot"), str) and len(fs["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fs.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(fs["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(fs.get("elapsed_s"), (int, float)))

# 6P test
fs2 = api("POST", "/api/analysis/fracture-sets", {"source": "demo", "well": "6P"})
check("6P works", fs2 is not None and fs2.get("well") == "6P")


# ── [173] Breakout Prediction ────────────────────────────
print("\n[173] Breakout Prediction")
bp = api("POST", "/api/analysis/breakout-prediction", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6, "ucs_mpa": 80, "mud_weight_sg": 1.2})
check("Status 200", bp is not None)
check("Has well", bp.get("well") == "3P")
check("Has depth_m", isinstance(bp.get("depth_m"), (int, float)))
check("Has friction", isinstance(bp.get("friction"), (int, float)))
check("Has UCS_MPa", isinstance(bp.get("UCS_MPa"), (int, float)))
check("Has mud_weight_SG", isinstance(bp.get("mud_weight_SG"), (int, float)))
check("Has SHmax_MPa", isinstance(bp.get("SHmax_MPa"), (int, float)))
check("Has Shmin_MPa", isinstance(bp.get("Shmin_MPa"), (int, float)))
check("Has Sv_MPa", isinstance(bp.get("Sv_MPa"), (int, float)))
check("Has Pp_MPa", isinstance(bp.get("Pp_MPa"), (int, float)))
check("Has Pw_MPa", isinstance(bp.get("Pw_MPa"), (int, float)))
check("Has max_hoop_stress_MPa", isinstance(bp.get("max_hoop_stress_MPa"), (int, float)))
check("Has safety_factor", isinstance(bp.get("safety_factor"), (int, float)))
check("Has breakout_exists", isinstance(bp.get("breakout_exists"), bool))
check("Has breakout_width_deg", isinstance(bp.get("breakout_width_deg"), (int, float)))
check("Has dif_exists", isinstance(bp.get("dif_exists"), bool))
check("Has stability", bp.get("stability") in ("STABLE", "BREAKOUT_ONLY", "TENSILE_ONLY", "BOTH"))
check("Has recommendations", isinstance(bp.get("recommendations"), list) and len(bp["recommendations"]) > 0)
check("Has plot", isinstance(bp.get("plot"), str) and len(bp["plot"]) > 100)
check("Has stakeholder_brief", isinstance(bp.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(bp["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(bp.get("elapsed_s"), (int, float)))

# 6P test
bp2 = api("POST", "/api/analysis/breakout-prediction", {"source": "demo", "well": "6P", "depth": 2500})
check("6P works", bp2 is not None and bp2.get("well") == "6P")


# ── [174] Data Completeness ──────────────────────────────
print("\n[174] Data Completeness")
dc = api("POST", "/api/analysis/data-completeness", {"source": "demo", "well": "3P"})
check("Status 200", dc is not None)
check("Has well", dc.get("well") == "3P")
check("Has n_rows", isinstance(dc.get("n_rows"), int) and dc["n_rows"] > 0)
check("Has columns", isinstance(dc.get("columns"), list) and len(dc["columns"]) >= 3)
c0 = dc["columns"][0]
check("Col has column", isinstance(c0.get("column"), str))
check("Col has n_present", isinstance(c0.get("n_present"), int))
check("Col has n_missing", isinstance(c0.get("n_missing"), int))
check("Col has pct_complete", isinstance(c0.get("pct_complete"), (int, float)))
check("Col has completeness", c0.get("completeness") in ("COMPLETE", "GOOD", "FAIR", "POOR", "PARTIAL"))
check("Has n_gaps", isinstance(dc.get("n_gaps"), int))
check("Has gaps", isinstance(dc.get("gaps"), list))
check("Has quality_score", isinstance(dc.get("quality_score"), int))
check("Has quality_grade", dc.get("quality_grade") in ("A", "B", "C", "D"))
check("Has recommendations", isinstance(dc.get("recommendations"), list) and len(dc["recommendations"]) > 0)
check("Has plot", isinstance(dc.get("plot"), str) and len(dc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(dc.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(dc["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(dc.get("elapsed_s"), (int, float)))

# 6P test
dc2 = api("POST", "/api/analysis/data-completeness", {"source": "demo", "well": "6P"})
check("6P works", dc2 is not None and dc2.get("well") == "6P")


# ── Summary ──────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.47.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
