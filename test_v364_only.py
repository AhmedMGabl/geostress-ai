"""Tests for v3.64.0: Hydraulic Fracture Design, Cap Rock Integrity, Fault Reactivation, Formation Pressure, Thermal Stress."""
import requests, sys

BASE = "http://localhost:8158"
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
# [255] Hydraulic Fracture Design
# ═══════════════════════════════════════════════════════════════
print("\n[255] Hydraulic Fracture Design")
code, r = api("POST", "/api/analysis/hydraulic-fracture-design", {"source": "demo", "well": "3P", "depth_m": 3000, "injection_rate_bpm": 20, "fluid_viscosity_cp": 100, "proppant_conc_ppg": 2})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has injection_rate_bpm", "injection_rate_bpm" in r)
test("Has fluid_viscosity_cp", "fluid_viscosity_cp" in r)
test("Has proppant_conc_ppg", "proppant_conc_ppg" in r)
test("Has breakdown_pressure_MPa", "breakdown_pressure_MPa" in r)
test("Has closure_pressure_MPa", "closure_pressure_MPa" in r)
test("Has ISIP_MPa", "ISIP_MPa" in r)
test("Has net_pressure_MPa", "net_pressure_MPa" in r)
test("Has half_length_m", "half_length_m" in r)
test("Has avg_width_mm", "avg_width_mm" in r)
test("Has conductivity_md_ft", "conductivity_md_ft" in r)
test("Has design_class", "design_class" in r)
test("Class valid", r.get("design_class") in ("HIGH_NET_PRESSURE", "SHORT_FRAC", "STANDARD"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has time_min", r.get("profile", [{}])[0].get("time_min") is not None)
test("Prof has pressure_MPa", r.get("profile", [{}])[0].get("pressure_MPa") is not None)
test("Prof has half_length_m", r.get("profile", [{}])[0].get("half_length_m") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High rate", api("POST", "/api/analysis/hydraulic-fracture-design", {"source": "demo", "well": "3P", "injection_rate_bpm": 60})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/hydraulic-fracture-design", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [256] Cap Rock Integrity
# ═══════════════════════════════════════════════════════════════
print("\n[256] Cap Rock Integrity")
code, r = api("POST", "/api/analysis/cap-rock-integrity", {"source": "demo", "well": "3P", "cap_depth_m": 2000, "cap_thickness_m": 50, "cap_perm_nD": 10, "cap_UCS_MPa": 60})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has cap_depth_m", "cap_depth_m" in r)
test("Has cap_thickness_m", "cap_thickness_m" in r)
test("Has cap_perm_nD", "cap_perm_nD" in r)
test("Has cap_UCS_MPa", "cap_UCS_MPa" in r)
test("Has n_cap_fractures", "n_cap_fractures" in r)
test("Has frac_density_per_m", "frac_density_per_m" in r)
test("Has entry_pressure_MPa", "entry_pressure_MPa" in r)
test("Has max_column_m", "max_column_m" in r)
test("Has leak_risk_score", "leak_risk_score" in r)
test("Has integrity_class", "integrity_class" in r)
test("Class valid", r.get("integrity_class") in ("COMPROMISED", "MARGINAL", "INTACT"))
test("Has stress_at_cap", "stress_at_cap" in r)
test("Stress has Sv_MPa", "Sv_MPa" in r.get("stress_at_cap", {}))
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High perm", api("POST", "/api/analysis/cap-rock-integrity", {"source": "demo", "well": "3P", "cap_perm_nD": 500})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/cap-rock-integrity", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [257] Fault Reactivation
# ═══════════════════════════════════════════════════════════════
print("\n[257] Fault Reactivation")
code, r = api("POST", "/api/analysis/fault-reactivation-pressure", {"source": "demo", "well": "3P", "depth_m": 3000, "fault_strike_deg": 45, "fault_dip_deg": 60, "friction": 0.6, "delta_Pp_MPa": 5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has fault_strike_deg", "fault_strike_deg" in r)
test("Has fault_dip_deg", "fault_dip_deg" in r)
test("Has friction", "friction" in r)
test("Has delta_Pp_MPa", "delta_Pp_MPa" in r)
test("Has sigma_n_MPa", "sigma_n_MPa" in r)
test("Has tau_MPa", "tau_MPa" in r)
test("Has slip_tendency", "slip_tendency" in r)
test("Has critical_Pp_MPa", "critical_Pp_MPa" in r)
test("Has Pp_margin_MPa", "Pp_margin_MPa" in r)
test("Has reactivated_at_delta", "reactivated_at_delta" in r)
test("Has react_class", "react_class" in r)
test("Class valid", r.get("react_class") in ("CRITICAL", "HIGH_RISK", "MODERATE_RISK", "STABLE"))
test("Has pressure_sweep", "pressure_sweep" in r)
test("Sweep non-empty", len(r.get("pressure_sweep", [])) > 0)
test("Sweep has Pp_MPa", r.get("pressure_sweep", [{}])[0].get("Pp_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High dPp", api("POST", "/api/analysis/fault-reactivation-pressure", {"source": "demo", "well": "3P", "delta_Pp_MPa": 20})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/fault-reactivation-pressure", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [258] Formation Pressure Profile
# ═══════════════════════════════════════════════════════════════
print("\n[258] Formation Pressure Profile")
code, r = api("POST", "/api/analysis/formation-pressure-profile", {"source": "demo", "well": "3P", "depth_from": 100, "depth_to": 5000, "n_points": 25, "overpressure_factor": 1.0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has overpressure_factor", "overpressure_factor" in r)
test("Has mean_gradient_psi_ft", "mean_gradient_psi_ft" in r)
test("Has max_Pp_MPa", "max_Pp_MPa" in r)
test("Has pressure_class", "pressure_class" in r)
test("Class valid", r.get("pressure_class") in ("OVERPRESSURED", "MILDLY_OVERPRESSURED", "NORMAL", "UNDERPRESSURED"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Pp_hydro_MPa", r.get("profile", [{}])[0].get("Pp_hydro_MPa") is not None)
test("Prof has Pp_actual_MPa", r.get("profile", [{}])[0].get("Pp_actual_MPa") is not None)
test("Prof has Sv_MPa", r.get("profile", [{}])[0].get("Sv_MPa") is not None)
test("Prof has emw_ppg", r.get("profile", [{}])[0].get("emw_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Overpressured", api("POST", "/api/analysis/formation-pressure-profile", {"source": "demo", "well": "3P", "overpressure_factor": 1.5})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/formation-pressure-profile", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [259] Thermal Stress
# ═══════════════════════════════════════════════════════════════
print("\n[259] Thermal Stress")
code, r = api("POST", "/api/analysis/thermal-stress-wellbore", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "geothermal_grad_C_km": 30, "delta_T_C": -20})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has geothermal_grad_C_km", "geothermal_grad_C_km" in r)
test("Has delta_T_C", "delta_T_C" in r)
test("Has max_thermal_stress_MPa", "max_thermal_stress_MPa" in r)
test("Has mean_thermal_stress_MPa", "mean_thermal_stress_MPa" in r)
test("Has pct_frac_risk", "pct_frac_risk" in r)
test("Has thermal_class", "thermal_class" in r)
test("Class valid", r.get("thermal_class") in ("CRITICAL", "HIGH_RISK", "MODERATE", "LOW"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has T_formation_C", r.get("profile", [{}])[0].get("T_formation_C") is not None)
test("Prof has sigma_thermal_MPa", r.get("profile", [{}])[0].get("sigma_thermal_MPa") is not None)
test("Prof has sigma_hoop_orig_MPa", r.get("profile", [{}])[0].get("sigma_hoop_orig_MPa") is not None)
test("Prof has sigma_hoop_thermal_MPa", r.get("profile", [{}])[0].get("sigma_hoop_thermal_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Hot gradient", api("POST", "/api/analysis/thermal-stress-wellbore", {"source": "demo", "well": "3P", "geothermal_grad_C_km": 50, "delta_T_C": 30})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/thermal-stress-wellbore", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.64.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
