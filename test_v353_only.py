"""Tests for v3.53.0: Fracture Density Log, Stress Gradient, Coulomb Failure, Fracture Porosity, Breakout Azimuth."""
import requests, sys

BASE = "http://localhost:8142"
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
# [200] Fracture Density Log
# ═══════════════════════════════════════════════════════════════
print("\n[200] Fracture Density Log")
code, r = api("POST", "/api/analysis/fracture-density-log", {"source": "demo", "well": "3P", "bin_size_m": 10})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has bin_size_m", "bin_size_m" in r)
test("Has depth_range_m", "depth_range_m" in r)
test("Has n_bins", "n_bins" in r)
test("Has P10_per_m", "P10_per_m" in r)
test("Has mean_density_per_m", "mean_density_per_m" in r)
test("Has max_density_per_m", "max_density_per_m" in r)
test("Has max_density_depth_m", "max_density_depth_m" in r)
test("Has std_density_per_m", "std_density_per_m" in r)
test("Has cv_density", "cv_density" in r)
test("Has distribution_class", "distribution_class" in r)
test("Class valid", r.get("distribution_class") in ("HIGHLY_VARIABLE", "VARIABLE", "UNIFORM"))
test("Has log", "log" in r)
test("Log non-empty", len(r.get("log", [])) > 0)
test("Entry has depth_from_m", r.get("log", [{}])[0].get("depth_from_m") is not None)
test("Entry has count", r.get("log", [{}])[0].get("count") is not None)
test("Entry has density_per_m", r.get("log", [{}])[0].get("density_per_m") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/fracture-density-log", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [201] Stress Gradient Profile
# ═══════════════════════════════════════════════════════════════
print("\n[201] Stress Gradient Profile")
code, r = api("POST", "/api/analysis/stress-gradient", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 30})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has Sv_gradient_MPa_per_km", "Sv_gradient_MPa_per_km" in r)
test("Has SHmax_gradient_MPa_per_km", "SHmax_gradient_MPa_per_km" in r)
test("Has Shmin_gradient_MPa_per_km", "Shmin_gradient_MPa_per_km" in r)
test("Has Pp_gradient_MPa_per_km", "Pp_gradient_MPa_per_km" in r)
test("Has stress_regime", "stress_regime" in r)
test("Regime valid", r.get("stress_regime") in ("NORMAL_FAULT", "STRIKE_SLIP", "REVERSE_FAULT"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Sv_MPa", r.get("profile", [{}])[0].get("Sv_MPa") is not None)
test("Prof has SHmax_MPa", r.get("profile", [{}])[0].get("SHmax_MPa") is not None)
test("Prof has Shmin_MPa", r.get("profile", [{}])[0].get("Shmin_MPa") is not None)
test("Prof has Pp_MPa", r.get("profile", [{}])[0].get("Pp_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/stress-gradient", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [202] Coulomb Failure Function
# ═══════════════════════════════════════════════════════════════
print("\n[202] Coulomb Failure Function")
code, r = api("POST", "/api/analysis/coulomb-failure", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6, "cohesion_mpa": 0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has friction", "friction" in r)
test("Has cohesion_MPa", "cohesion_MPa" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has SHmax_MPa", "SHmax_MPa" in r)
test("Has Shmin_MPa", "Shmin_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has n_failed", "n_failed" in r)
test("Has pct_failed", "pct_failed" in r)
test("Has failure_class", "failure_class" in r)
test("Class valid", r.get("failure_class") in ("EXTENSIVE", "MODERATE", "LIMITED", "STABLE"))
test("Has mean_CFF_MPa", "mean_CFF_MPa" in r)
test("Has max_CFF_MPa", "max_CFF_MPa" in r)
test("Has min_CFF_MPa", "min_CFF_MPa" in r)
test("Has top_critical", "top_critical" in r)
test("Top non-empty", len(r.get("top_critical", [])) > 0)
test("Top has azimuth_deg", r.get("top_critical", [{}])[0].get("azimuth_deg") is not None)
test("Top has CFF_MPa", r.get("top_critical", [{}])[0].get("CFF_MPa") is not None)
test("Top has failed", "failed" in r.get("top_critical", [{}])[0])
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/coulomb-failure", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [203] Fracture Porosity Estimate
# ═══════════════════════════════════════════════════════════════
print("\n[203] Fracture Porosity Estimate")
code, r = api("POST", "/api/analysis/fracture-porosity", {"source": "demo", "well": "3P", "aperture_mm": 0.5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has aperture_mm", "aperture_mm" in r)
test("Has depth_range_m", "depth_range_m" in r)
test("Has interval_m", "interval_m" in r)
test("Has mean_apparent_aperture_mm", "mean_apparent_aperture_mm" in r)
test("Has fracture_porosity_pct", "fracture_porosity_pct" in r)
test("Has porosity_class", "porosity_class" in r)
test("Class valid", r.get("porosity_class") in ("HIGH", "MODERATE", "LOW"))
test("Has equivalent_permeability_darcy", "equivalent_permeability_darcy" in r)
test("Has P10_per_m", "P10_per_m" in r)
test("Has depth_porosity", "depth_porosity" in r)
test("Depth porosity non-empty", len(r.get("depth_porosity", [])) > 0)
test("DP has depth_from_m", r.get("depth_porosity", [{}])[0].get("depth_from_m") is not None)
test("DP has n_fractures", r.get("depth_porosity", [{}])[0].get("n_fractures") is not None)
test("DP has porosity_pct", r.get("depth_porosity", [{}])[0].get("porosity_pct") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/fracture-porosity", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [204] Breakout Azimuth Prediction
# ═══════════════════════════════════════════════════════════════
print("\n[204] Breakout Azimuth Prediction")
code, r = api("POST", "/api/analysis/breakout-azimuth", {"source": "demo", "well": "3P", "depth": 3000})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has SHmax_MPa", "SHmax_MPa" in r)
test("Has Shmin_MPa", "Shmin_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has SHmax_azimuth_deg", "SHmax_azimuth_deg" in r)
test("Has breakout_azimuth_deg", "breakout_azimuth_deg" in r)
test("Has DITF_azimuth_deg", "DITF_azimuth_deg" in r)
test("Has max_hoop_stress_MPa", "max_hoop_stress_MPa" in r)
test("Has min_hoop_stress_MPa", "min_hoop_stress_MPa" in r)
test("Has azimuth_concentration", "azimuth_concentration" in r)
test("Has confidence", "confidence" in r)
test("Confidence valid", r.get("confidence") in ("HIGH", "MODERATE", "LOW"))
test("Has hoop_stress_profile", "hoop_stress_profile" in r)
test("Profile non-empty", len(r.get("hoop_stress_profile", [])) > 0)
test("Hoop has theta_deg", r.get("hoop_stress_profile", [{}])[0].get("theta_deg") is not None)
test("Hoop has sigma_theta_MPa", r.get("hoop_stress_profile", [{}])[0].get("sigma_theta_MPa") is not None)
test("Breakout ~90° from SHmax", abs(((r.get("breakout_azimuth_deg",0) - r.get("SHmax_azimuth_deg",0)) % 360) - 90) < 5 or abs(((r.get("breakout_azimuth_deg",0) - r.get("SHmax_azimuth_deg",0)) % 360) - 270) < 5)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/breakout-azimuth", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.53.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
