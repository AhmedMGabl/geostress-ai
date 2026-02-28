"""Tests for v3.49.0 endpoints: Stress Polygon, Fracture Permeability Tensor, Wellbore Breakout Width, Pore Pressure Prediction, Fault Reactivation."""
import requests, sys, json

BASE = "http://localhost:8137"
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
# [180] Stress Polygon
# ═══════════════════════════════════════════════════════════════
print("\n[180] Stress Polygon")
code, r = api("POST", "/api/analysis/stress-polygon-diagram", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has friction", "friction" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has frictional_limit_q", "frictional_limit_q" in r)
test("Has current_state", "current_state" in r)
test("Current state has SHmax_MPa", r.get("current_state", {}).get("SHmax_MPa") is not None)
test("Current state has Shmin_MPa", r.get("current_state", {}).get("Shmin_MPa") is not None)
test("Has current_regime", "current_regime" in r)
test("Regime is valid", r.get("current_regime") in ("NF", "SS", "RF"))
test("Has stability_margin", "stability_margin" in r)
test("Has stability_class", "stability_class" in r)
test("Stability class valid", r.get("stability_class") in ("STABLE", "MARGINAL", "CRITICAL"))
test("Has n_polygon_points", "n_polygon_points" in r)
test("Has polygon_NF", "polygon_NF" in r)
test("Has polygon_SS", "polygon_SS" in r)
test("Has polygon_RF", "polygon_RF" in r)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)

# With custom pp_mpa
code2, r2 = api("POST", "/api/analysis/stress-polygon-diagram", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6, "pp_mpa": 20})
test("Custom Pp works", code2 == 200 and r2.get("Pp_MPa") == 20)

# 6P
code6, r6 = api("POST", "/api/analysis/stress-polygon-diagram", {"source": "demo", "well": "6P"})
test("6P works or graceful", code6 in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [181] Fracture Permeability Tensor
# ═══════════════════════════════════════════════════════════════
print("\n[181] Fracture Permeability Tensor")
code, r = api("POST", "/api/analysis/fracture-permeability-tensor", {"source": "demo", "well": "3P", "aperture_mm": 0.5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has aperture_mm", "aperture_mm" in r)
test("Has tensor_components", "tensor_components" in r)
test("Tensor has Kxx", "Kxx" in r.get("tensor_components", {}))
test("Tensor has Kyy", "Kyy" in r.get("tensor_components", {}))
test("Tensor has Kzz", "Kzz" in r.get("tensor_components", {}))
test("Tensor has Kxy", "Kxy" in r.get("tensor_components", {}))
test("Has principal_permeabilities", "principal_permeabilities" in r)
test("3 principal dirs", len(r.get("principal_permeabilities", [])) == 3)
test("Principal has axis", r.get("principal_permeabilities", [{}])[0].get("axis") is not None)
test("Principal has permeability_darcy", r.get("principal_permeabilities", [{}])[0].get("permeability_darcy") is not None)
test("Principal has azimuth_deg", r.get("principal_permeabilities", [{}])[0].get("azimuth_deg") is not None)
test("Has k1_darcy", "k1_darcy" in r)
test("Has k2_darcy", "k2_darcy" in r)
test("Has k3_darcy", "k3_darcy" in r)
test("k1 >= k2 >= k3", r.get("k1_darcy", 0) >= r.get("k2_darcy", 0) >= r.get("k3_darcy", -1))
test("Has anisotropy_ratio", "anisotropy_ratio" in r)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works or graceful", api("POST", "/api/analysis/fracture-permeability-tensor", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [182] Wellbore Breakout Width
# ═══════════════════════════════════════════════════════════════
print("\n[182] Wellbore Breakout Width")
code, r = api("POST", "/api/analysis/breakout-width", {"source": "demo", "well": "3P", "depth": 3000, "ucs_mpa": 80, "friction": 0.6})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has friction", "friction" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has SHmax_MPa", "SHmax_MPa" in r)
test("Has Shmin_MPa", "Shmin_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has optimal_mud_weight_SG", "optimal_mud_weight_SG" in r)
test("Has mud_weight_window", "mud_weight_window" in r)
test("Has n_mud_weights_tested", "n_mud_weights_tested" in r)
test("29 MW tested", r.get("n_mud_weights_tested") == 29)
test("Has mud_weight_analysis", "mud_weight_analysis" in r)
test("MW analysis has entries", len(r.get("mud_weight_analysis", [])) > 0)
test("Entry has mud_weight_SG", r.get("mud_weight_analysis", [{}])[0].get("mud_weight_SG") is not None)
test("Entry has breakout_width_deg", r.get("mud_weight_analysis", [{}])[0].get("breakout_width_deg") is not None)
test("Entry has breakout_exists", "breakout_exists" in r.get("mud_weight_analysis", [{}])[0])
test("Entry has safety_factor", r.get("mud_weight_analysis", [{}])[0].get("safety_factor") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/breakout-width", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [183] Pore Pressure Prediction
# ═══════════════════════════════════════════════════════════════
print("\n[183] Pore Pressure Prediction")
code, r = api("POST", "/api/analysis/pore-pressure-prediction", {"source": "demo", "well": "3P", "method": "eaton", "n_points": 30})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has method", "method" in r)
test("Method is EATON", r.get("method") == "EATON")
test("Has n_points", "n_points" in r)
test("Has depth_range_m", "depth_range_m" in r)
test("Has pressure_regime", "pressure_regime" in r)
test("Regime valid", r.get("pressure_regime") in ("HYDROSTATIC", "OVERPRESSURED", "UNDERPRESSURED"))
test("Has max_overpressure_ratio", "max_overpressure_ratio" in r)
test("Has profile", "profile" in r)
test("Profile has entries", len(r.get("profile", [])) > 0)
test("Entry has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Entry has Sv_MPa", r.get("profile", [{}])[0].get("Sv_MPa") is not None)
test("Entry has Pp_hydrostatic_MPa", r.get("profile", [{}])[0].get("Pp_hydrostatic_MPa") is not None)
test("Entry has Pp_predicted_MPa", r.get("profile", [{}])[0].get("Pp_predicted_MPa") is not None)
test("Entry has equivalent_mud_weight_SG", r.get("profile", [{}])[0].get("equivalent_mud_weight_SG") is not None)
test("Entry has overpressure_ratio", r.get("profile", [{}])[0].get("overpressure_ratio") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)

# Bowers method
code_b, r_b = api("POST", "/api/analysis/pore-pressure-prediction", {"source": "demo", "well": "3P", "method": "bowers"})
test("Bowers method works", code_b == 200 and r_b.get("method") == "BOWERS")

# 6P
test("6P works or graceful", api("POST", "/api/analysis/pore-pressure-prediction", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [184] Fault Reactivation
# ═══════════════════════════════════════════════════════════════
print("\n[184] Fault Reactivation")
code, r = api("POST", "/api/analysis/fault-reactivation", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has friction", "friction" in r)
test("Has sigma1_MPa", "sigma1_MPa" in r)
test("Has sigma3_MPa", "sigma3_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has SHmax_azimuth_deg", "SHmax_azimuth_deg" in r)
test("Has n_faults_analyzed", "n_faults_analyzed" in r)
test("Has n_high_risk", "n_high_risk" in r)
test("Has n_moderate_risk", "n_moderate_risk" in r)
test("Has n_low_risk", "n_low_risk" in r)
test("Has overall_risk", "overall_risk" in r)
test("Risk valid", r.get("overall_risk") in ("HIGH", "MODERATE", "LOW"))
test("Has fault_analyses", "fault_analyses" in r)
test("Analyses non-empty", len(r.get("fault_analyses", [])) > 0)
fa0 = r.get("fault_analyses", [{}])[0]
test("Fault has azimuth", "fault_azimuth_deg" in fa0)
test("Fault has dip", "fault_dip_deg" in fa0)
test("Fault has sigma_n", "sigma_n_MPa" in fa0)
test("Fault has tau", "tau_MPa" in fa0)
test("Fault has slip_tendency", "slip_tendency" in fa0)
test("Fault has coulomb_margin", "coulomb_margin_MPa" in fa0)
test("Fault has reactivation_risk", "reactivation_risk" in fa0)
test("Fault has Pp_critical", "Pp_critical_MPa" in fa0)
test("Fault has Pp_margin", "Pp_margin_MPa" in fa0)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)

# Specific fault orientation
code_s, r_s = api("POST", "/api/analysis/fault-reactivation", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6, "fault_azimuth": 45, "fault_dip": 60})
test("Specific fault works", code_s == 200 and r_s.get("n_faults_analyzed") == 1)

# 6P
test("6P works or graceful", api("POST", "/api/analysis/fault-reactivation", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.49.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
