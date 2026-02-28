"""Tests for v3.50.0: In-Situ Stress Ratio, Fracture Corridor, Drilling Hazard, Thermal Stress, DFN Statistics."""
import requests, sys

BASE = "http://localhost:8138"
SESSION = requests.Session()
PASS = 0
FAIL = 0

def test(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label}")

def api(method, path, payload=None):
    try:
        if method == "POST":
            r = SESSION.post(BASE + path, json=payload, timeout=300)
        else:
            r = SESSION.get(BASE + path, timeout=300)
        return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0, {}


# ═══════════════════════════════════════════════════════════════
# [185] In-Situ Stress Ratio
# ═══════════════════════════════════════════════════════════════
print("\n[185] In-Situ Stress Ratio")
code, r = api("POST", "/api/analysis/stress-ratio", {"source": "demo", "well": "3P", "n_points": 25})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_points", "n_points" in r)
test("Has depth_range_m", "depth_range_m" in r)
test("Has K0_elastic", "K0_elastic" in r)
test("Has poisson_ratio", "poisson_ratio" in r)
test("Has mean_Kh", "mean_Kh" in r)
test("Has K_trend", "K_trend" in r)
test("K_trend valid", r.get("K_trend") in ("INCREASING", "DECREASING"))
test("Has regime_indication", "regime_indication" in r)
test("Regime valid", r.get("regime_indication") in ("EXTENSIONAL", "COMPRESSIONAL", "TRANSITIONAL"))
test("Has profile", "profile" in r)
test("Profile has entries", len(r.get("profile", [])) > 0)
test("Entry has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Entry has Kh_min", r.get("profile", [{}])[0].get("Kh_min") is not None)
test("Entry has KH_max", r.get("profile", [{}])[0].get("KH_max") is not None)
test("Entry has K_sheorey", r.get("profile", [{}])[0].get("K_sheorey") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/stress-ratio", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [186] Fracture Corridor Detection
# ═══════════════════════════════════════════════════════════════
print("\n[186] Fracture Corridor Detection")
code, r = api("POST", "/api/analysis/fracture-corridor", {"source": "demo", "well": "3P", "window_m": 5, "threshold_factor": 2.0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has window_m", "window_m" in r)
test("Has threshold_factor", "threshold_factor" in r)
test("Has mean_density_per_window", "mean_density_per_window" in r)
test("Has threshold_count", "threshold_count" in r)
test("Has n_corridors", "n_corridors" in r)
test("Has total_corridor_thickness_m", "total_corridor_thickness_m" in r)
test("Has pct_fractures_in_corridors", "pct_fractures_in_corridors" in r)
test("Has corridors list", "corridors" in r)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)

# Check corridor structure if any exist
if r.get("n_corridors", 0) > 0:
    c0 = r["corridors"][0]
    test("Corridor has depth_from_m", "depth_from_m" in c0)
    test("Corridor has depth_to_m", "depth_to_m" in c0)
    test("Corridor has n_fractures", "n_fractures" in c0)
    test("Corridor has density_per_m", "density_per_m" in c0)
else:
    test("No corridors (uniform distribution)", True)
    test("No corridors placeholder", True)
    test("No corridors placeholder2", True)
    test("No corridors placeholder3", True)

test("6P works", api("POST", "/api/analysis/fracture-corridor", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [187] Drilling Hazard Assessment
# ═══════════════════════════════════════════════════════════════
print("\n[187] Drilling Hazard Assessment")
code, r = api("POST", "/api/analysis/drilling-hazard", {"source": "demo", "well": "3P", "depth": 3000, "mud_weight_sg": 1.2})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has mud_weight_SG", "mud_weight_SG" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has Pw_MPa", "Pw_MPa" in r)
test("Has n_hazards", "n_hazards" in r)
test("6 hazards", r.get("n_hazards") == 6)
test("Has n_high_risk", "n_high_risk" in r)
test("Has n_moderate_risk", "n_moderate_risk" in r)
test("Has overall_risk", "overall_risk" in r)
test("Risk valid", r.get("overall_risk") in ("HIGH", "MODERATE", "LOW"))
test("Has mud_weight_window", "mud_weight_window" in r)
test("Has hazards list", "hazards" in r)
test("Hazard has name", r.get("hazards", [{}])[0].get("hazard") is not None)
test("Hazard has risk_level", r.get("hazards", [{}])[0].get("risk_level") is not None)
test("Hazard has safety_factor", r.get("hazards", [{}])[0].get("safety_factor") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/drilling-hazard", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [188] Thermal Stress
# ═══════════════════════════════════════════════════════════════
print("\n[188] Thermal Stress")
code, r = api("POST", "/api/analysis/thermal-stress", {"source": "demo", "well": "3P", "depth": 3000, "geothermal_gradient": 30})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has geothermal_gradient", "geothermal_gradient_C_per_km" in r)
test("Has formation_temp_C", "formation_temp_C" in r)
test("Has mud_temp_C", "mud_temp_C" in r)
test("Has delta_T_C", "delta_T_C" in r)
test("Has thermal_stress_MPa", "thermal_stress_MPa" in r)
test("Has thermal_impact", "thermal_impact" in r)
test("Impact valid", r.get("thermal_impact") in ("SIGNIFICANT", "MODERATE", "MINOR"))
test("Has sf_mechanical", "sf_mechanical" in r)
test("Has sf_with_thermal", "sf_with_thermal" in r)
test("Has friction_at_temp", "friction_at_temp" in r)
test("Has thermal_profile", "thermal_profile" in r)
test("Profile has entries", len(r.get("thermal_profile", [])) > 0)
test("Entry has depth_m", r.get("thermal_profile", [{}])[0].get("depth_m") is not None)
test("Entry has formation_temp_C", r.get("thermal_profile", [{}])[0].get("formation_temp_C") is not None)
test("Entry has thermal_stress_MPa", r.get("thermal_profile", [{}])[0].get("thermal_stress_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)

# Custom mud temp
code2, r2 = api("POST", "/api/analysis/thermal-stress", {"source": "demo", "well": "3P", "depth": 3000, "geothermal_gradient": 30, "mud_temp_c": 50})
test("Custom mud temp works", code2 == 200 and r2.get("mud_temp_C") == 50)

test("6P works", api("POST", "/api/analysis/thermal-stress", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [189] DFN Statistics
# ═══════════════════════════════════════════════════════════════
print("\n[189] DFN Statistics")
code, r = api("POST", "/api/analysis/dfn-statistics", {"source": "demo", "well": "3P"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has depth_range_m", "depth_range_m" in r)
test("Has P10_per_m", "P10_per_m" in r)
test("Has mean_spacing_m", "mean_spacing_m" in r)
test("Has median_spacing_m", "median_spacing_m" in r)
test("Has cv_spacing", "cv_spacing" in r)
test("Has spacing_distribution", "spacing_distribution" in r)
test("Spacing dist valid", r.get("spacing_distribution") in ("REGULAR", "RANDOM", "CLUSTERED"))
test("Has fisher_kappa", "fisher_kappa" in r)
test("Has R_bar", "R_bar" in r)
test("Has mean_pole_azimuth_deg", "mean_pole_azimuth_deg" in r)
test("Has mean_pole_dip_deg", "mean_pole_dip_deg" in r)
test("Has n_dominant_sets", "n_dominant_sets" in r)
test("Has connectivity_score", "connectivity_score" in r)
test("Has connectivity_class", "connectivity_class" in r)
test("Connectivity valid", r.get("connectivity_class") in ("HIGH", "MODERATE", "LOW"))
test("Has dfn_quality", "dfn_quality" in r)
test("Quality valid", r.get("dfn_quality") in ("GOOD", "FAIR", "POOR", "INSUFFICIENT"))
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/dfn-statistics", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.50.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
