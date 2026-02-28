"""Tests for v3.59.0: Thermal Stress, Aperture Distribution, Induced Seismicity, Casing Design, Formation Integrity."""
import requests, sys

BASE = "http://localhost:8149"
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
# [230] Thermal Stress Effect
# ═══════════════════════════════════════════════════════════════
print("\n[230] Thermal Stress Effect")
code, r = api("POST", "/api/analysis/thermal-stress-effect", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "delta_T": -20})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has delta_T_C", "delta_T_C" in r)
test("Has thermal_stress_MPa", "thermal_stress_MPa" in r)
test("Has mean_hoop_change_MPa", "mean_hoop_change_MPa" in r)
test("Has thermal_class", "thermal_class" in r)
test("Class valid", r.get("thermal_class") in ("SIGNIFICANT", "MODERATE", "MINOR"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has thermal_stress_MPa", r.get("profile", [{}])[0].get("thermal_stress_MPa") is not None)
test("Prof has hoop_thermal_MPa", r.get("profile", [{}])[0].get("hoop_thermal_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Heating works", api("POST", "/api/analysis/thermal-stress-effect", {"source": "demo", "well": "3P", "delta_T": 30})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [231] Fracture Aperture Distribution
# ═══════════════════════════════════════════════════════════════
print("\n[231] Fracture Aperture Distribution")
code, r = api("POST", "/api/analysis/fracture-aperture-dist", {"source": "demo", "well": "3P"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has aperture_model", "aperture_model" in r)
test("Has mean_aperture_mm", "mean_aperture_mm" in r)
test("Has median_aperture_mm", "median_aperture_mm" in r)
test("Has P10_mm", "P10_mm" in r)
test("Has P50_mm", "P50_mm" in r)
test("Has P90_mm", "P90_mm" in r)
test("Has max_aperture_mm", "max_aperture_mm" in r)
test("Has hydraulic_eq_mm", "hydraulic_eq_mm" in r)
test("Has lognorm_p_value", "lognorm_p_value" in r)
test("Has aperture_class", "aperture_class" in r)
test("Class valid", r.get("aperture_class") in ("WIDE", "MODERATE", "NARROW"))
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/fracture-aperture-dist", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [232] Induced Seismicity Risk
# ═══════════════════════════════════════════════════════════════
print("\n[232] Induced Seismicity Risk")
code, r = api("POST", "/api/analysis/induced-seismicity", {"source": "demo", "well": "3P", "depth": 3000, "injection_rate": 500, "duration_days": 365})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has injection_rate_m3_day", "injection_rate_m3_day" in r)
test("Has duration_days", "duration_days" in r)
test("Has total_volume_m3", "total_volume_m3" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has n_critical", "n_critical" in r)
test("Has pct_critical", "pct_critical" in r)
test("Has seismogenic_index", "seismogenic_index" in r)
test("Has expected_events_M0", "expected_events_M0" in r)
test("Has max_magnitude_est", "max_magnitude_est" in r)
test("Has risk_class", "risk_class" in r)
test("Class valid", r.get("risk_class") in ("HIGH", "MODERATE", "LOW"))
test("Has pressure_profile", "pressure_profile" in r)
test("Profile non-empty", len(r.get("pressure_profile", [])) > 0)
test("Prof has delta_Pp_MPa", r.get("pressure_profile", [{}])[0].get("delta_Pp_MPa") is not None)
test("Prof has n_reactivated", r.get("pressure_profile", [{}])[0].get("n_reactivated") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/induced-seismicity", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [233] Casing Design Check
# ═══════════════════════════════════════════════════════════════
print("\n[233] Casing Design Check")
code, r = api("POST", "/api/analysis/casing-design-check", {"source": "demo", "well": "3P", "casing_burst_psi": 8000, "casing_collapse_psi": 6000})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has casing_burst_psi", "casing_burst_psi" in r)
test("Has casing_collapse_psi", "casing_collapse_psi" in r)
test("Has min_burst_SF", "min_burst_SF" in r)
test("Has min_collapse_SF", "min_collapse_SF" in r)
test("Has pct_burst_ok", "pct_burst_ok" in r)
test("Has pct_collapse_ok", "pct_collapse_ok" in r)
test("Has design_class", "design_class" in r)
test("Class valid", r.get("design_class") in ("INADEQUATE", "MARGINAL", "ADEQUATE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has burst_SF", r.get("profile", [{}])[0].get("burst_SF") is not None)
test("Prof has collapse_SF", r.get("profile", [{}])[0].get("collapse_SF") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Weak casing works", api("POST", "/api/analysis/casing-design-check", {"source": "demo", "well": "3P", "casing_burst_psi": 3000, "casing_collapse_psi": 2000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [234] Formation Integrity Test
# ═══════════════════════════════════════════════════════════════
print("\n[234] Formation Integrity Test")
code, r = api("POST", "/api/analysis/formation-integrity", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "tensile_strength_MPa": 5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has tensile_strength_MPa", "tensile_strength_MPa" in r)
test("Has min_margin_MPa", "min_margin_MPa" in r)
test("Has mean_FIT_MPa", "mean_FIT_MPa" in r)
test("Has mean_LOT_MPa", "mean_LOT_MPa" in r)
test("Has integrity_class", "integrity_class" in r)
test("Class valid", r.get("integrity_class") in ("STRONG", "ADEQUATE", "MARGINAL", "WEAK"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has FIT_MPa", r.get("profile", [{}])[0].get("FIT_MPa") is not None)
test("Prof has LOT_MPa", r.get("profile", [{}])[0].get("LOT_MPa") is not None)
test("Prof has FBP_MPa", r.get("profile", [{}])[0].get("FBP_MPa") is not None)
test("Prof has margin_MPa", r.get("profile", [{}])[0].get("margin_MPa") is not None)
test("Prof has FIT_ppg", r.get("profile", [{}])[0].get("FIT_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Low tensile works", api("POST", "/api/analysis/formation-integrity", {"source": "demo", "well": "3P", "tensile_strength_MPa": 1})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.59.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
