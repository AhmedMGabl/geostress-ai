"""Tests for v3.60.0: Horizon Stress, Fracture Swarm, Effective Permeability, Wellbore Trajectory, Stress Anisotropy."""
import requests, sys

BASE = "http://localhost:8150"
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
# [235] Horizon Stress
# ═══════════════════════════════════════════════════════════════
print("\n[235] Horizon Stress")
code, r = api("POST", "/api/analysis/horizon-stress", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "regime": "normal"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has regime", "regime" in r)
test("Has mean_Sv_MPa", "mean_Sv_MPa" in r)
test("Has min_eff_Shmin_MPa", "min_eff_Shmin_MPa" in r)
test("Has max_stress_ratio", "max_stress_ratio" in r)
test("Has horizon_class", "horizon_class" in r)
test("Class valid", r.get("horizon_class") in ("HIGH_ANISOTROPY", "MODERATE_ANISOTROPY", "LOW_ANISOTROPY"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Sv_MPa", r.get("profile", [{}])[0].get("Sv_MPa") is not None)
test("Prof has SHmax_MPa", r.get("profile", [{}])[0].get("SHmax_MPa") is not None)
test("Prof has Shmin_MPa", r.get("profile", [{}])[0].get("Shmin_MPa") is not None)
test("Prof has stress_ratio_H_h", r.get("profile", [{}])[0].get("stress_ratio_H_h") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Strike-slip works", api("POST", "/api/analysis/horizon-stress", {"source": "demo", "well": "3P", "regime": "strike_slip"})[0] in (200, 422, 500))
test("Reverse works", api("POST", "/api/analysis/horizon-stress", {"source": "demo", "well": "6P", "regime": "reverse"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [236] Fracture Swarm Analysis
# ═══════════════════════════════════════════════════════════════
print("\n[236] Fracture Swarm Analysis")
code, r = api("POST", "/api/analysis/fracture-swarm-analysis", {"source": "demo", "well": "3P", "window_m": 10, "min_count": 3})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has window_m", "window_m" in r)
test("Has min_count", "min_count" in r)
test("Has n_swarms", "n_swarms" in r)
test("Has total_in_swarms", "total_in_swarms" in r)
test("Has pct_in_swarms", "pct_in_swarms" in r)
test("Has max_intensity_per_m", "max_intensity_per_m" in r)
test("Has mean_intensity_per_m", "mean_intensity_per_m" in r)
test("Has swarm_class", "swarm_class" in r)
test("Class valid", r.get("swarm_class") in ("INTENSE", "MODERATE", "SPARSE"))
test("Has swarms", "swarms" in r)
test("Swarms non-empty", len(r.get("swarms", [])) > 0)
test("Swarm has depth_from_m", r.get("swarms", [{}])[0].get("depth_from_m") is not None)
test("Swarm has count", r.get("swarms", [{}])[0].get("count") is not None)
test("Swarm has intensity_per_m", r.get("swarms", [{}])[0].get("intensity_per_m") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/fracture-swarm-analysis", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))
test("Large window", api("POST", "/api/analysis/fracture-swarm-analysis", {"source": "demo", "well": "3P", "window_m": 50})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [237] Effective Permeability
# ═══════════════════════════════════════════════════════════════
print("\n[237] Effective Permeability")
code, r = api("POST", "/api/analysis/effective-permeability", {"source": "demo", "well": "3P", "aperture_mm": 0.1, "matrix_perm_mD": 0.01})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has aperture_mm", "aperture_mm" in r)
test("Has matrix_perm_mD", "matrix_perm_mD" in r)
test("Has thickness_m", "thickness_m" in r)
test("Has P10_per_m", "P10_per_m" in r)
test("Has fracture_porosity", "fracture_porosity" in r)
test("Has fracture_perm_mD", "fracture_perm_mD" in r)
test("Has k_eff_mD", "k_eff_mD" in r)
test("Has k_connected_mD", "k_connected_mD" in r)
test("Has connectivity_fraction", "connectivity_fraction" in r)
test("Has enhancement_factor", "enhancement_factor" in r)
test("Has perm_class", "perm_class" in r)
test("Class valid", r.get("perm_class") in ("HIGH", "MODERATE", "LOW"))
test("Has sensitivity", "sensitivity" in r)
test("Sensitivity non-empty", len(r.get("sensitivity", [])) > 0)
test("Sens has aperture_mm", r.get("sensitivity", [{}])[0].get("aperture_mm") is not None)
test("Sens has k_eff_mD", r.get("sensitivity", [{}])[0].get("k_eff_mD") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Large aperture", api("POST", "/api/analysis/effective-permeability", {"source": "demo", "well": "3P", "aperture_mm": 1.0})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/effective-permeability", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [238] Wellbore Trajectory
# ═══════════════════════════════════════════════════════════════
print("\n[238] Wellbore Trajectory")
code, r = api("POST", "/api/analysis/wellbore-trajectory", {"source": "demo", "well": "3P", "deviation_deg": 15, "azimuth_well_deg": 90, "depth_from": 500, "depth_to": 5000})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has deviation_deg", "deviation_deg" in r)
test("Has azimuth_well_deg", "azimuth_well_deg" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has SHmax_est_deg", "SHmax_est_deg" in r)
test("Has optimal_azimuth_deg", "optimal_azimuth_deg" in r)
test("Has worst_azimuth_deg", "worst_azimuth_deg" in r)
test("Has min_collapse_margin_MPa", "min_collapse_margin_MPa" in r)
test("Has min_frac_margin_MPa", "min_frac_margin_MPa" in r)
test("Has traj_class", "traj_class" in r)
test("Class valid", r.get("traj_class") in ("CRITICAL", "MARGINAL", "STABLE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has TVD_m", r.get("profile", [{}])[0].get("TVD_m") is not None)
test("Prof has sigma_axial_MPa", r.get("profile", [{}])[0].get("sigma_axial_MPa") is not None)
test("Prof has collapse_margin_MPa", r.get("profile", [{}])[0].get("collapse_margin_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Vertical well", api("POST", "/api/analysis/wellbore-trajectory", {"source": "demo", "well": "3P", "deviation_deg": 0})[0] in (200, 422, 500))
test("High deviation", api("POST", "/api/analysis/wellbore-trajectory", {"source": "demo", "well": "3P", "deviation_deg": 60})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/wellbore-trajectory", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [239] Stress Anisotropy Ratio
# ═══════════════════════════════════════════════════════════════
print("\n[239] Stress Anisotropy Ratio")
code, r = api("POST", "/api/analysis/stress-anisotropy-ratio", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has mean_aniso_ratio", "mean_aniso_ratio" in r)
test("Has max_aniso_ratio", "max_aniso_ratio" in r)
test("Has mean_diff_stress_MPa", "mean_diff_stress_MPa" in r)
test("Has max_deviatoric_ratio", "max_deviatoric_ratio" in r)
test("Has aniso_class", "aniso_class" in r)
test("Class valid", r.get("aniso_class") in ("HIGH", "MODERATE", "LOW"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has aniso_ratio", r.get("profile", [{}])[0].get("aniso_ratio") is not None)
test("Prof has eff_aniso_ratio", r.get("profile", [{}])[0].get("eff_aniso_ratio") is not None)
test("Prof has diff_stress_MPa", r.get("profile", [{}])[0].get("diff_stress_MPa") is not None)
test("Prof has deviatoric_ratio", r.get("profile", [{}])[0].get("deviatoric_ratio") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/stress-anisotropy-ratio", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))
test("Narrow range", api("POST", "/api/analysis/stress-anisotropy-ratio", {"source": "demo", "well": "3P", "depth_from": 2000, "depth_to": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.60.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
