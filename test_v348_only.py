"""Focused test for v3.48.0 endpoints: Stress Rotation + Hydraulic Conductivity + Trajectory Optimization + Criticality Ranking + Geomech Log."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8135"


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


# ── [175] Stress Rotation ────────────────────────────────
print("\n[175] Stress Rotation")
sr = api("POST", "/api/analysis/stress-rotation", {"source": "demo", "well": "3P", "window": 30})
check("Status 200", sr is not None)
check("Has well", sr.get("well") == "3P")
check("Has n_fractures", isinstance(sr.get("n_fractures"), int) and sr["n_fractures"] > 0)
check("Has window_size", isinstance(sr.get("window_size"), int))
check("Has rotation_rate_deg_per_km", isinstance(sr.get("rotation_rate_deg_per_km"), (int, float)))
check("Has total_rotation_deg", isinstance(sr.get("total_rotation_deg"), (int, float)))
check("Has rotation_class", sr.get("rotation_class") in ("SIGNIFICANT", "MODERATE", "MINIMAL"))
check("Has n_breakpoints", isinstance(sr.get("n_breakpoints"), int))
check("Has breakpoints", isinstance(sr.get("breakpoints"), list))
check("Has rolling_data", isinstance(sr.get("rolling_data"), list) and len(sr["rolling_data"]) >= 1)
rd0 = sr["rolling_data"][0]
check("Rolling has depth_m", isinstance(rd0.get("depth_m"), (int, float)))
check("Rolling has mean_azimuth_deg", isinstance(rd0.get("mean_azimuth_deg"), (int, float)))
check("Rolling has std_azimuth_deg", isinstance(rd0.get("std_azimuth_deg"), (int, float)))
check("Has recommendations", isinstance(sr.get("recommendations"), list) and len(sr["recommendations"]) > 0)
check("Has plot", isinstance(sr.get("plot"), str) and len(sr["plot"]) > 100)
check("Has stakeholder_brief", isinstance(sr.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(sr["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(sr.get("elapsed_s"), (int, float)))

# 6P test — may fail due to null depths
try:
    sr2 = api("POST", "/api/analysis/stress-rotation", {"source": "demo", "well": "6P"})
    check("6P works or graceful", sr2 is not None)
except Exception as e:
    check("6P works or graceful", "400" in str(e) or "404" in str(e), f"Expected: {e}")


# ── [176] Hydraulic Conductivity ─────────────────────────
print("\n[176] Hydraulic Conductivity")
hc = api("POST", "/api/analysis/hydraulic-conductivity", {"source": "demo", "well": "3P", "aperture_mm": 0.5})
check("Status 200", hc is not None)
check("Has well", hc.get("well") == "3P")
check("Has n_fractures", isinstance(hc.get("n_fractures"), int) and hc["n_fractures"] > 0)
check("Has aperture_mm", isinstance(hc.get("aperture_mm"), (int, float)))
check("Has P10_per_m", isinstance(hc.get("P10_per_m"), (int, float)))
check("Has mean_spacing_m", isinstance(hc.get("mean_spacing_m"), (int, float)))
check("Has T_single_m2_per_s", isinstance(hc.get("T_single_m2_per_s"), (int, float)))
check("Has K_bulk_m_per_s", isinstance(hc.get("K_bulk_m_per_s"), (int, float)))
check("Has k_permeability_darcy", isinstance(hc.get("k_permeability_darcy"), (int, float)))
check("Has anisotropy_ratio", isinstance(hc.get("anisotropy_ratio"), (int, float)))
check("Has directional_conductivity", isinstance(hc.get("directional_conductivity"), list) and len(hc["directional_conductivity"]) >= 4)
dc0 = hc["directional_conductivity"][0]
check("Dir has azimuth_range", isinstance(dc0.get("azimuth_range"), str))
check("Dir has n_fractures", isinstance(dc0.get("n_fractures"), int))
check("Dir has K_m_per_s", isinstance(dc0.get("K_m_per_s"), (int, float)))
check("Has recommendations", isinstance(hc.get("recommendations"), list) and len(hc["recommendations"]) > 0)
check("Has plot", isinstance(hc.get("plot"), str) and len(hc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(hc.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(hc["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(hc.get("elapsed_s"), (int, float)))

# 6P test
try:
    hc2 = api("POST", "/api/analysis/hydraulic-conductivity", {"source": "demo", "well": "6P"})
    check("6P works or graceful", hc2 is not None)
except Exception as e:
    check("6P works or graceful", "400" in str(e) or "404" in str(e), f"Expected: {e}")


# ── [177] Trajectory Optimization ────────────────────────
print("\n[177] Trajectory Optimization")
to = api("POST", "/api/analysis/trajectory-optimization", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6, "ucs_mpa": 80, "mud_weight_sg": 1.2})
check("Status 200", to is not None)
check("Has well", to.get("well") == "3P")
check("Has depth_m", isinstance(to.get("depth_m"), (int, float)))
check("Has UCS_MPa", isinstance(to.get("UCS_MPa"), (int, float)))
check("Has mud_weight_SG", isinstance(to.get("mud_weight_SG"), (int, float)))
check("Has SHmax_MPa", isinstance(to.get("SHmax_MPa"), (int, float)))
check("Has Shmin_MPa", isinstance(to.get("Shmin_MPa"), (int, float)))
check("Has Sv_MPa", isinstance(to.get("Sv_MPa"), (int, float)))
check("Has optimal", isinstance(to.get("optimal"), dict))
opt = to["optimal"]
check("Optimal has azimuth_deg", isinstance(opt.get("azimuth_deg"), int))
check("Optimal has dip_deg", isinstance(opt.get("dip_deg"), int))
check("Optimal has safety_factor", isinstance(opt.get("safety_factor"), (int, float)))
check("Has worst", isinstance(to.get("worst"), dict))
check("Has n_trajectories_tested", isinstance(to.get("n_trajectories_tested"), int) and to["n_trajectories_tested"] > 10)
check("Has pct_stable", isinstance(to.get("pct_stable"), (int, float)))
check("Has recommendations", isinstance(to.get("recommendations"), list) and len(to["recommendations"]) > 0)
check("Has plot", isinstance(to.get("plot"), str) and len(to["plot"]) > 100)
check("Has stakeholder_brief", isinstance(to.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(to["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(to.get("elapsed_s"), (int, float)))

# 6P test
to2 = api("POST", "/api/analysis/trajectory-optimization", {"source": "demo", "well": "6P"})
check("6P works", to2 is not None and to2.get("well") == "6P")


# ── [178] Criticality Ranking ────────────────────────────
print("\n[178] Criticality Ranking")
cr = api("POST", "/api/analysis/criticality-ranking", {"source": "demo", "well": "3P", "friction": 0.6, "depth": 3000, "top_n": 20})
check("Status 200", cr is not None)
check("Has well", cr.get("well") == "3P")
check("Has n_fractures", isinstance(cr.get("n_fractures"), int) and cr["n_fractures"] > 0)
check("Has n_critically_stressed", isinstance(cr.get("n_critically_stressed"), int))
check("Has pct_critically_stressed", isinstance(cr.get("pct_critically_stressed"), (int, float)))
check("Has mean_score", isinstance(cr.get("mean_score"), (int, float)))
check("Has std_score", isinstance(cr.get("std_score"), (int, float)))
check("Has top_fractures", isinstance(cr.get("top_fractures"), list) and len(cr["top_fractures"]) >= 5)
tf0 = cr["top_fractures"][0]
check("Frac has index", isinstance(tf0.get("index"), int))
check("Frac has azimuth_deg", isinstance(tf0.get("azimuth_deg"), (int, float)))
check("Frac has dip_deg", isinstance(tf0.get("dip_deg"), (int, float)))
check("Frac has slip_tendency", isinstance(tf0.get("slip_tendency"), (int, float)))
check("Frac has dilation_tendency", isinstance(tf0.get("dilation_tendency"), (int, float)))
check("Frac has composite_score", isinstance(tf0.get("composite_score"), (int, float)))
check("Frac has critically_stressed", isinstance(tf0.get("critically_stressed"), bool))
check("Has recommendations", isinstance(cr.get("recommendations"), list) and len(cr["recommendations"]) > 0)
check("Has plot", isinstance(cr.get("plot"), str) and len(cr["plot"]) > 100)
check("Has stakeholder_brief", isinstance(cr.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(cr["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(cr.get("elapsed_s"), (int, float)))

# 6P test
cr2 = api("POST", "/api/analysis/criticality-ranking", {"source": "demo", "well": "6P"})
check("6P works", cr2 is not None and cr2.get("well") == "6P")


# ── [179] Geomech Log ────────────────────────────────────
print("\n[179] Geomech Log")
gl = api("POST", "/api/analysis/geomech-log", {"source": "demo", "well": "3P", "bin_size_m": 10})
check("Status 200", gl is not None)
check("Has well", gl.get("well") == "3P")
check("Has n_fractures", isinstance(gl.get("n_fractures"), int) and gl["n_fractures"] > 0)
check("Has bin_size_m", isinstance(gl.get("bin_size_m"), (int, float)))
check("Has depth_range_m", isinstance(gl.get("depth_range_m"), list) and len(gl["depth_range_m"]) == 2)
check("Has n_intervals", isinstance(gl.get("n_intervals"), int) and gl["n_intervals"] >= 1)
check("Has n_with_data", isinstance(gl.get("n_with_data"), int))
check("Has coverage_pct", isinstance(gl.get("coverage_pct"), (int, float)))
check("Has log_entries", isinstance(gl.get("log_entries"), list) and len(gl["log_entries"]) >= 1)
le0 = gl["log_entries"][0]
check("Entry has depth_from", isinstance(le0.get("depth_from"), (int, float)))
check("Entry has depth_to", isinstance(le0.get("depth_to"), (int, float)))
check("Entry has depth_mid", isinstance(le0.get("depth_mid"), (int, float)))
check("Entry has n_fractures", isinstance(le0.get("n_fractures"), int))
check("Entry has P10", isinstance(le0.get("P10"), (int, float)))
check("Entry has fracture_quality", le0.get("fracture_quality") in ("GOOD", "FAIR", "POOR", "NO_DATA"))
check("Has recommendations", isinstance(gl.get("recommendations"), list) and len(gl["recommendations"]) > 0)
check("Has plot", isinstance(gl.get("plot"), str) and len(gl["plot"]) > 100)
check("Has stakeholder_brief", isinstance(gl.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(gl["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(gl.get("elapsed_s"), (int, float)))

# 6P test — may fail due to null depths
try:
    gl2 = api("POST", "/api/analysis/geomech-log", {"source": "demo", "well": "6P"})
    check("6P works or graceful", gl2 is not None)
except Exception as e:
    check("6P works or graceful", "400" in str(e) or "404" in str(e), f"Expected: {e}")


# ── Summary ──────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.48.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
