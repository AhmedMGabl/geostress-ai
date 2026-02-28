"""Tests for v3.56.0: Aperture Profile, Critical Injection, Stress Path Evolution, Fracture Set ID, Pp Prediction."""
import requests, sys

BASE = "http://localhost:8146"
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
# [215] Fracture Aperture Profile
# ═══════════════════════════════════════════════════════════════
print("\n[215] Fracture Aperture Profile")
code, r = api("POST", "/api/analysis/aperture-profile", {"source": "demo", "well": "3P", "aperture_mm": 0.5, "bin_size_m": 50})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has aperture_mm", "aperture_mm" in r)
test("Has bin_size_m", "bin_size_m" in r)
test("Has depth_range_m", "depth_range_m" in r)
test("Has mean_aperture_mm", "mean_aperture_mm" in r)
test("Has max_aperture_mm", "max_aperture_mm" in r)
test("Has max_conductivity_m_s", "max_conductivity_m_s" in r)
test("Has aperture_class", "aperture_class" in r)
test("Class valid", r.get("aperture_class") in ("WIDE", "MODERATE", "NARROW"))
test("Has n_bins", "n_bins" in r)
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_from_m", r.get("profile", [{}])[0].get("depth_from_m") is not None)
test("Prof has n_fractures", r.get("profile", [{}])[0].get("n_fractures") is not None)
test("Prof has mean_aperture_mm", r.get("profile", [{}])[0].get("mean_aperture_mm") is not None)
test("Prof has hydraulic_conductivity_m_s", r.get("profile", [{}])[0].get("hydraulic_conductivity_m_s") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/aperture-profile", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [216] Critical Injection Pressure
# ═══════════════════════════════════════════════════════════════
print("\n[216] Critical Injection Pressure")
code, r = api("POST", "/api/analysis/critical-injection", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has friction", "friction" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has Pp_hydrostatic_MPa", "Pp_hydrostatic_MPa" in r)
test("Has Shmin_MPa", "Shmin_MPa" in r)
test("Has SHmax_MPa", "SHmax_MPa" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has first_reactivation_MPa", "first_reactivation_MPa" in r)
test("Has margin_to_first_MPa", "margin_to_first_MPa" in r)
test("Has frac_initiation_MPa", "frac_initiation_MPa" in r)
test("Has thresh_10pct_above_hydro_MPa", "thresh_10pct_above_hydro_MPa" in r)
test("Has risk_class", "risk_class" in r)
test("Class valid", r.get("risk_class") in ("CRITICAL", "HIGH", "MODERATE", "LOW"))
test("Has pressure_profile", "pressure_profile" in r)
test("Profile non-empty", len(r.get("pressure_profile", [])) > 0)
test("Prof has injection_above_hydrostatic_MPa", r.get("pressure_profile", [{}])[0].get("injection_above_hydrostatic_MPa") is not None)
test("Prof has n_reactivated", r.get("pressure_profile", [{}])[0].get("n_reactivated") is not None)
test("Prof has pct_reactivated", r.get("pressure_profile", [{}])[0].get("pct_reactivated") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/critical-injection", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [217] Stress Path Evolution
# ═══════════════════════════════════════════════════════════════
print("\n[217] Stress Path Evolution")
code, r = api("POST", "/api/analysis/stress-path-evolution", {"source": "demo", "well": "3P", "depth": 3000, "scenario": "depletion", "delta_pp": 20})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has scenario", "scenario" in r)
test("Has delta_pp_mpa", "delta_pp_mpa" in r)
test("Has n_steps", "n_steps" in r)
test("Has stress_path_coefficient", "stress_path_coefficient" in r)
test("Has biot_coefficient", "biot_coefficient" in r)
test("Has poisson_ratio", "poisson_ratio" in r)
test("Has initial_regime", "initial_regime" in r)
test("Has final_regime", "final_regime" in r)
test("Has regime_changed", "regime_changed" in r)
test("Has path", "path" in r)
test("Path non-empty", len(r.get("path", [])) > 0)
test("Path has step", r.get("path", [{}])[0].get("step") is not None)
test("Path has delta_Pp_MPa", r.get("path", [{}])[0].get("delta_Pp_MPa") is not None)
test("Path has Sv_MPa", r.get("path", [{}])[0].get("Sv_MPa") is not None)
test("Path has SHmax_MPa", r.get("path", [{}])[0].get("SHmax_MPa") is not None)
test("Path has p_prime_MPa", r.get("path", [{}])[0].get("p_prime_MPa") is not None)
test("Path has q_MPa", r.get("path", [{}])[0].get("q_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Injection works", api("POST", "/api/analysis/stress-path-evolution", {"source": "demo", "well": "3P", "scenario": "injection"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [218] Fracture Set Identification
# ═══════════════════════════════════════════════════════════════
print("\n[218] Fracture Set Identification")
code, r = api("POST", "/api/analysis/fracture-set-id", {"source": "demo", "well": "3P", "n_sets": 3})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has n_sets", "n_sets" in r)
test("Has dominant_set", "dominant_set" in r)
test("Dominant has set_id", r.get("dominant_set", {}).get("set_id") is not None)
test("Dominant has n_fractures", r.get("dominant_set", {}).get("n_fractures") is not None)
test("Dominant has mean_azimuth_deg", r.get("dominant_set", {}).get("mean_azimuth_deg") is not None)
test("Dominant has mean_dip_deg", r.get("dominant_set", {}).get("mean_dip_deg") is not None)
test("Dominant has concentration_R", r.get("dominant_set", {}).get("concentration_R") is not None)
test("Has sets", "sets" in r)
test("Sets has 3", len(r.get("sets", [])) == 3)
test("Set has pct_total", r.get("sets", [{}])[0].get("pct_total") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/fracture-set-id", {"source": "demo", "well": "6P", "n_sets": 2})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [219] Pore Pressure Prediction
# ═══════════════════════════════════════════════════════════════
print("\n[219] Pore Pressure Prediction")
code, r = api("POST", "/api/analysis/pp-prediction", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "overpressure_factor": 1.0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has overpressure_factor", "overpressure_factor" in r)
test("Has pressure_class", "pressure_class" in r)
test("Class valid", r.get("pressure_class") in ("NORMAL", "OVERPRESSURED", "UNDERPRESSURED"))
test("Has mean_gradient_MPa_per_km", "mean_gradient_MPa_per_km" in r)
test("Has max_Pp_MPa", "max_Pp_MPa" in r)
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Pp_hydrostatic_MPa", r.get("profile", [{}])[0].get("Pp_hydrostatic_MPa") is not None)
test("Prof has Pp_predicted_MPa", r.get("profile", [{}])[0].get("Pp_predicted_MPa") is not None)
test("Prof has Pp_lithostatic_MPa", r.get("profile", [{}])[0].get("Pp_lithostatic_MPa") is not None)
test("Prof has EMW_ppg", r.get("profile", [{}])[0].get("EMW_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Overpressured works", api("POST", "/api/analysis/pp-prediction", {"source": "demo", "well": "3P", "overpressure_factor": 1.3})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.56.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
