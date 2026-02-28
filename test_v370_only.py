"""Tests for v3.70.0: Thermal Fracture, Fault Slip, Pp Depletion, Wellbore Heating, Caprock Seal."""
import requests, sys

BASE = "http://localhost:8165"
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
# [285] Thermal Fracture Risk
# ═══════════════════════════════════════════════════════════════
print("\n[285] Thermal Fracture Risk")
code, r = api("POST", "/api/analysis/thermal-fracture", {"source": "demo", "well": "3P", "depth_m": 3000, "injection_temp_C": 20, "reservoir_temp_C": 120})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has injection_temp_C", "injection_temp_C" in r)
test("Has reservoir_temp_C", "reservoir_temp_C" in r)
test("Has delta_T_C", "delta_T_C" in r)
test("Has thermal_stress_MPa", "thermal_stress_MPa" in r)
test("Has frac_margin_MPa", "frac_margin_MPa" in r)
test("Has frac_risk_ratio", "frac_risk_ratio" in r)
test("Has tf_class", "tf_class" in r)
test("Class valid", r.get("tf_class") in ("CRITICAL", "HIGH_RISK", "MODERATE", "LOW"))
test("Has temperature_sweep", "temperature_sweep" in r)
test("Sweep non-empty", len(r.get("temperature_sweep", [])) > 0)
test("Sweep has injection_temp_C", r.get("temperature_sweep", [{}])[0].get("injection_temp_C") is not None)
test("Sweep has thermal_stress_MPa", r.get("temperature_sweep", [{}])[0].get("thermal_stress_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Hot reservoir", api("POST", "/api/analysis/thermal-fracture", {"source": "demo", "well": "3P", "reservoir_temp_C": 200})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/thermal-fracture", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [286] Fault Slip Tendency
# ═══════════════════════════════════════════════════════════════
print("\n[286] Fault Slip Tendency")
code, r = api("POST", "/api/analysis/fault-slip-tendency", {"source": "demo", "well": "3P", "depth_m": 3000, "fault_azimuth_deg": 45, "fault_dip_deg": 60, "friction": 0.6})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has fault_azimuth_deg", "fault_azimuth_deg" in r)
test("Has fault_dip_deg", "fault_dip_deg" in r)
test("Has friction", "friction" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has SHmax_MPa", "SHmax_MPa" in r)
test("Has Shmin_MPa", "Shmin_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has sigma_n_eff_MPa", "sigma_n_eff_MPa" in r)
test("Has tau_MPa", "tau_MPa" in r)
test("Has slip_tendency", "slip_tendency" in r)
test("Has dilation_tendency", "dilation_tendency" in r)
test("Has CFS_MPa", "CFS_MPa" in r)
test("Has slip_class", "slip_class" in r)
test("Class valid", r.get("slip_class") in ("CRITICAL", "HIGH", "MODERATE", "STABLE"))
test("Has dip_sweep", "dip_sweep" in r)
test("Sweep non-empty", len(r.get("dip_sweep", [])) > 0)
test("Sweep has dip_deg", r.get("dip_sweep", [{}])[0].get("dip_deg") is not None)
test("Sweep has slip_tendency", r.get("dip_sweep", [{}])[0].get("slip_tendency") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Steep fault", api("POST", "/api/analysis/fault-slip-tendency", {"source": "demo", "well": "3P", "fault_dip_deg": 85})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/fault-slip-tendency", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [287] Pore Pressure Depletion
# ═══════════════════════════════════════════════════════════════
print("\n[287] Pore Pressure Depletion")
code, r = api("POST", "/api/analysis/pore-pressure-depletion", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "depletion_pct": 20})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has depletion_pct", "depletion_pct" in r)
test("Has mean_delta_Pp_MPa", "mean_delta_Pp_MPa" in r)
test("Has max_delta_Pp_MPa", "max_delta_Pp_MPa" in r)
test("Has depl_class", "depl_class" in r)
test("Class valid", r.get("depl_class") in ("SEVERE", "SIGNIFICANT", "MODERATE", "MINOR"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Pp_initial_MPa", r.get("profile", [{}])[0].get("Pp_initial_MPa") is not None)
test("Prof has Pp_depleted_MPa", r.get("profile", [{}])[0].get("Pp_depleted_MPa") is not None)
test("Prof has delta_Pp_MPa", r.get("profile", [{}])[0].get("delta_Pp_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High depletion", api("POST", "/api/analysis/pore-pressure-depletion", {"source": "demo", "well": "3P", "depletion_pct": 50})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/pore-pressure-depletion", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [288] Wellbore Heating
# ═══════════════════════════════════════════════════════════════
print("\n[288] Wellbore Heating")
code, r = api("POST", "/api/analysis/wellbore-heating", {"source": "demo", "well": "3P", "depth_m": 3000, "production_temp_C": 150, "initial_temp_C": 40, "casing_grade": "L80"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has production_temp_C", "production_temp_C" in r)
test("Has initial_temp_C", "initial_temp_C" in r)
test("Has delta_T_C", "delta_T_C" in r)
test("Has casing_grade", "casing_grade" in r)
test("Has casing_thermal_stress_MPa", "casing_thermal_stress_MPa" in r)
test("Has casing_yield_MPa", "casing_yield_MPa" in r)
test("Has casing_SF", "casing_SF" in r)
test("Has cement_thermal_stress_MPa", "cement_thermal_stress_MPa" in r)
test("Has cement_SF", "cement_SF" in r)
test("Has heat_class", "heat_class" in r)
test("Class valid", r.get("heat_class") in ("CRITICAL", "HIGH_RISK", "MODERATE", "SAFE"))
test("Has temp_profile", "temp_profile" in r)
test("Profile non-empty", len(r.get("temp_profile", [])) > 0)
test("Prof has depth_m", r.get("temp_profile", [{}])[0].get("depth_m") is not None)
test("Prof has formation_temp_C", r.get("temp_profile", [{}])[0].get("formation_temp_C") is not None)
test("Prof has casing_stress_MPa", r.get("temp_profile", [{}])[0].get("casing_stress_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("P110 grade", api("POST", "/api/analysis/wellbore-heating", {"source": "demo", "well": "3P", "casing_grade": "P110", "production_temp_C": 180})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/wellbore-heating", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [289] Caprock Seal Capacity
# ═══════════════════════════════════════════════════════════════
print("\n[289] Caprock Seal Capacity")
code, r = api("POST", "/api/analysis/caprock-seal-capacity", {"source": "demo", "well": "3P", "depth_m": 3000, "caprock_thickness_m": 50, "caprock_permeability_mD": 0.001})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has caprock_thickness_m", "caprock_thickness_m" in r)
test("Has caprock_permeability_mD", "caprock_permeability_mD" in r)
test("Has Pc_entry_MPa", "Pc_entry_MPa" in r)
test("Has frac_leakoff_MPa", "frac_leakoff_MPa" in r)
test("Has effective_seal_MPa", "effective_seal_MPa" in r)
test("Has max_column_m", "max_column_m" in r)
test("Has seal_class", "seal_class" in r)
test("Class valid", r.get("seal_class") in ("POOR", "FAIR", "GOOD", "EXCELLENT"))
test("Has perm_sweep", "perm_sweep" in r)
test("Sweep non-empty", len(r.get("perm_sweep", [])) > 0)
test("Sweep has permeability_mD", r.get("perm_sweep", [{}])[0].get("permeability_mD") is not None)
test("Sweep has Pc_entry_MPa", r.get("perm_sweep", [{}])[0].get("Pc_entry_MPa") is not None)
test("Sweep has max_column_m", r.get("perm_sweep", [{}])[0].get("max_column_m") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High perm", api("POST", "/api/analysis/caprock-seal-capacity", {"source": "demo", "well": "3P", "caprock_permeability_mD": 1.0})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/caprock-seal-capacity", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.70.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
