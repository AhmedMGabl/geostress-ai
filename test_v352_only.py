"""Tests for v3.52.0: Aperture Distribution, Stability Window, Stress Anisotropy, FIT Prediction, Susceptibility Map."""
import requests, sys

BASE = "http://localhost:8141"
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
# [195] Fracture Aperture Distribution
# ═══════════════════════════════════════════════════════════════
print("\n[195] Fracture Aperture Distribution")
code, r = api("POST", "/api/analysis/aperture-distribution-stats", {"source": "demo", "well": "3P", "base_aperture_mm": 0.5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has base_aperture_mm", "base_aperture_mm" in r)
test("Has mean_aperture_mm", "mean_aperture_mm" in r)
test("Has median_aperture_mm", "median_aperture_mm" in r)
test("Has std_aperture_mm", "std_aperture_mm" in r)
test("Has min_aperture_mm", "min_aperture_mm" in r)
test("Has max_aperture_mm", "max_aperture_mm" in r)
test("Has p10_mm", "p10_mm" in r)
test("Has p50_mm", "p50_mm" in r)
test("Has p90_mm", "p90_mm" in r)
test("Has lognormal_mu", "lognormal_mu" in r)
test("Has lognormal_sigma", "lognormal_sigma" in r)
test("Has aperture_class", "aperture_class" in r)
test("Class valid", r.get("aperture_class") in ("WIDE", "MODERATE", "NARROW"))
test("Has mean_permeability_darcy", "mean_permeability_darcy" in r)
test("Has median_permeability_darcy", "median_permeability_darcy" in r)
test("Has histogram", "histogram" in r)
test("Histogram non-empty", len(r.get("histogram", [])) > 0)
test("Bin has bin_from_mm", r.get("histogram", [{}])[0].get("bin_from_mm") is not None)
test("Bin has count", r.get("histogram", [{}])[0].get("count") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/aperture-distribution-stats", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [196] Wellbore Stability Window
# ═══════════════════════════════════════════════════════════════
print("\n[196] Wellbore Stability Window")
code, r = api("POST", "/api/analysis/stability-window-map", {"source": "demo", "well": "3P", "depth": 3000, "ucs_mpa": 80, "tensile_mpa": 5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has tensile_MPa", "tensile_MPa" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has SHmax_MPa", "SHmax_MPa" in r)
test("Has Shmin_MPa", "Shmin_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has n_orientations", "n_orientations" in r)
test("Has pct_stable", "pct_stable" in r)
test("Has stability_class", "stability_class" in r)
test("Class valid", r.get("stability_class") in ("WIDE", "MODERATE", "NARROW", "CRITICAL"))
test("Has min_window_SG", "min_window_SG" in r)
test("Has max_window_SG", "max_window_SG" in r)
test("Has mean_window_SG", "mean_window_SG" in r)
test("Has best_orientation", "best_orientation" in r)
test("Has worst_orientation", "worst_orientation" in r)
test("Best has azimuth", "azimuth_deg" in r.get("best_orientation", {}))
test("Best has inclination", "inclination_deg" in r.get("best_orientation", {}))
test("Best has window_SG", "window_SG" in r.get("best_orientation", {}))
test("Has grid", "grid" in r)
test("Grid non-empty", len(r.get("grid", [])) > 0)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/stability-window-map", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [197] Stress Anisotropy Index
# ═══════════════════════════════════════════════════════════════
print("\n[197] Stress Anisotropy Index")
code, r = api("POST", "/api/analysis/stress-anisotropy", {"source": "demo", "well": "3P", "depth": 3000})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has SHmax_MPa", "SHmax_MPa" in r)
test("Has Shmin_MPa", "Shmin_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has horizontal_anisotropy", "horizontal_anisotropy" in r)
test("Has total_anisotropy", "total_anisotropy" in r)
test("Has R_ratio", "R_ratio" in r)
test("Has A_phi", "A_phi" in r)
test("Has von_mises_MPa", "von_mises_MPa" in r)
test("Has mean_stress_MPa", "mean_stress_MPa" in r)
test("Has anisotropy_class", "anisotropy_class" in r)
test("Class valid", r.get("anisotropy_class") in ("HIGH", "MODERATE", "LOW"))
test("Has azimuth_concentration", "azimuth_concentration" in r)
test("Has depth_profile", "depth_profile" in r)
test("Profile non-empty", len(r.get("depth_profile", [])) > 0)
test("Profile has depth_m", r.get("depth_profile", [{}])[0].get("depth_m") is not None)
test("Profile has horizontal_anisotropy", r.get("depth_profile", [{}])[0].get("horizontal_anisotropy") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/stress-anisotropy", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [198] FIT/LOT Prediction
# ═══════════════════════════════════════════════════════════════
print("\n[198] FIT/LOT Prediction")
code, r = api("POST", "/api/analysis/fit-prediction", {"source": "demo", "well": "3P", "depth": 3000, "tensile_mpa": 5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has tensile_MPa", "tensile_MPa" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has SHmax_MPa", "SHmax_MPa" in r)
test("Has Shmin_MPa", "Shmin_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has Pfrac_MPa", "Pfrac_MPa" in r)
test("Has Pleak_MPa", "Pleak_MPa" in r)
test("Has Pfit_MPa", "Pfit_MPa" in r)
test("Has EMW_frac_SG", "EMW_frac_SG" in r)
test("Has EMW_leak_SG", "EMW_leak_SG" in r)
test("Has EMW_fit_SG", "EMW_fit_SG" in r)
test("Has margin_MPa", "margin_MPa" in r)
test("Has safety_class", "safety_class" in r)
test("Class valid", r.get("safety_class") in ("SAFE", "ADEQUATE", "MARGINAL", "CRITICAL"))
test("Has n_profile_points", "n_profile_points" in r)
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Pfrac_MPa", r.get("profile", [{}])[0].get("Pfrac_MPa") is not None)
test("Prof has Pleak_MPa", r.get("profile", [{}])[0].get("Pleak_MPa") is not None)
test("Prof has EMW_frac_SG", r.get("profile", [{}])[0].get("EMW_frac_SG") is not None)
test("Pfrac > Pleak", r.get("Pfrac_MPa", 0) > r.get("Pleak_MPa", 999))
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/fit-prediction", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [199] Fracture Susceptibility Map
# ═══════════════════════════════════════════════════════════════
print("\n[199] Fracture Susceptibility Map")
code, r = api("POST", "/api/analysis/susceptibility-map", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has friction", "friction" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has SHmax_MPa", "SHmax_MPa" in r)
test("Has Shmin_MPa", "Shmin_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has n_grid_cells", "n_grid_cells" in r)
test("Has n_critically_stressed", "n_critically_stressed" in r)
test("Has pct_critically_stressed", "pct_critically_stressed" in r)
test("Has max_slip_tendency", "max_slip_tendency" in r)
test("Has n_actual_fractures", "n_actual_fractures" in r)
test("Has n_actual_cs", "n_actual_cs" in r)
test("Has pct_actual_cs", "pct_actual_cs" in r)
test("Has risk_class", "risk_class" in r)
test("Class valid", r.get("risk_class") in ("HIGH", "MODERATE", "LOW"))
test("Has top_susceptible", "top_susceptible" in r)
test("Top non-empty", len(r.get("top_susceptible", [])) > 0)
test("Top has azimuth_deg", r.get("top_susceptible", [{}])[0].get("azimuth_deg") is not None)
test("Top has dip_deg", r.get("top_susceptible", [{}])[0].get("dip_deg") is not None)
test("Top has slip_tendency", r.get("top_susceptible", [{}])[0].get("slip_tendency") is not None)
test("Top has critically_stressed", "critically_stressed" in r.get("top_susceptible", [{}])[0])
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/susceptibility-map", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.52.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
