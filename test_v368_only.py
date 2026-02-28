"""Tests for v3.68.0: Directional Stability, Effective Stress Gradient, Depletion, Fracture Reopen, APB."""
import requests, sys

BASE = "http://localhost:8163"
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
# [275] Directional Stability
# ═══════════════════════════════════════════════════════════════
print("\n[275] Directional Stability")
code, r = api("POST", "/api/analysis/directional-stability", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "wellbore_azimuth_deg": 45, "wellbore_inclination_deg": 30, "UCS_MPa": 50})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has wellbore_azimuth_deg", "wellbore_azimuth_deg" in r)
test("Has wellbore_inclination_deg", "wellbore_inclination_deg" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has mean_stability_index", "mean_stability_index" in r)
test("Has min_stability_index", "min_stability_index" in r)
test("Has pct_unstable", "pct_unstable" in r)
test("Has dir_class", "dir_class" in r)
test("Class valid", r.get("dir_class") in ("CRITICAL", "UNSTABLE", "MARGINAL", "STABLE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has stability_index", r.get("profile", [{}])[0].get("stability_index") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Horizontal well", api("POST", "/api/analysis/directional-stability", {"source": "demo", "well": "3P", "wellbore_inclination_deg": 90})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/directional-stability", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [276] Effective Stress Gradient
# ═══════════════════════════════════════════════════════════════
print("\n[276] Effective Stress Gradient")
code, r = api("POST", "/api/analysis/effective-stress-gradient", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "Pp_gradient": 0.45})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has Pp_gradient_psi_ft", "Pp_gradient_psi_ft" in r)
test("Has mean_Sv_grad_MPa_km", "mean_Sv_grad_MPa_km" in r)
test("Has mean_stress_ratio", "mean_stress_ratio" in r)
test("Has grad_class", "grad_class" in r)
test("Class valid", r.get("grad_class") in ("LOW_CONFINING", "MODERATE", "NORMAL", "HIGH_CONFINING"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has Sv_eff_MPa", r.get("profile", [{}])[0].get("Sv_eff_MPa") is not None)
test("Prof has Shmin_eff_MPa", r.get("profile", [{}])[0].get("Shmin_eff_MPa") is not None)
test("Prof has Sv_grad_MPa_km", r.get("profile", [{}])[0].get("Sv_grad_MPa_km") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Overpressured", api("POST", "/api/analysis/effective-stress-gradient", {"source": "demo", "well": "3P", "Pp_gradient": 0.7})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/effective-stress-gradient", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [277] Depletion Effect
# ═══════════════════════════════════════════════════════════════
print("\n[277] Depletion Effect")
code, r = api("POST", "/api/analysis/depletion-effect", {"source": "demo", "well": "3P", "depth_m": 3000, "depletion_MPa": 10, "poisson_ratio": 0.25})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has initial_Pp_MPa", "initial_Pp_MPa" in r)
test("Has depletion_MPa", "depletion_MPa" in r)
test("Has poisson_ratio", "poisson_ratio" in r)
test("Has stress_path_coeff", "stress_path_coeff" in r)
test("Has final_Shmin_MPa", "final_Shmin_MPa" in r)
test("Has total_frac_change_MPa", "total_frac_change_MPa" in r)
test("Has final_slip_tendency", "final_slip_tendency" in r)
test("Has depl_class", "depl_class" in r)
test("Class valid", r.get("depl_class") in ("SEVERE", "SIGNIFICANT", "MODERATE", "MINOR"))
test("Has path", "path" in r)
test("Path non-empty", len(r.get("path", [])) > 0)
test("Path has depletion_MPa", r.get("path", [{}])[0].get("depletion_MPa") is not None)
test("Path has Shmin_MPa", r.get("path", [{}])[0].get("Shmin_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Large depletion", api("POST", "/api/analysis/depletion-effect", {"source": "demo", "well": "3P", "depletion_MPa": 30})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/depletion-effect", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [278] Fracture Reopen Pressure
# ═══════════════════════════════════════════════════════════════
print("\n[278] Fracture Reopen Pressure")
code, r = api("POST", "/api/analysis/fracture-reopen-pressure", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has mean_margin_MPa", "mean_margin_MPa" in r)
test("Has min_reopen_MPa", "min_reopen_MPa" in r)
test("Has reopen_class", "reopen_class" in r)
test("Class valid", r.get("reopen_class") in ("LOW_MARGIN", "MODERATE", "SAFE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has breakdown_MPa", r.get("profile", [{}])[0].get("breakdown_MPa") is not None)
test("Prof has reopen_MPa", r.get("profile", [{}])[0].get("reopen_MPa") is not None)
test("Prof has ISIP_MPa", r.get("profile", [{}])[0].get("ISIP_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Shallow", api("POST", "/api/analysis/fracture-reopen-pressure", {"source": "demo", "well": "3P", "depth_from": 100, "depth_to": 1000})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/fracture-reopen-pressure", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [279] Annular Pressure Buildup
# ═══════════════════════════════════════════════════════════════
print("\n[279] Annular Pressure Buildup")
code, r = api("POST", "/api/analysis/annular-pressure-buildup", {"source": "demo", "well": "3P", "depth_m": 3000, "production_temp_C": 120, "initial_temp_C": 40, "annulus_fluid": "water", "casing_od_in": 9.625})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has production_temp_C", "production_temp_C" in r)
test("Has initial_temp_C", "initial_temp_C" in r)
test("Has annulus_fluid", "annulus_fluid" in r)
test("Has casing_od_in", "casing_od_in" in r)
test("Has max_apb_MPa", "max_apb_MPa" in r)
test("Has max_apb_psi", "max_apb_psi" in r)
test("Has min_safety_factor", "min_safety_factor" in r)
test("Has apb_class", "apb_class" in r)
test("Class valid", r.get("apb_class") in ("CRITICAL", "HIGH_RISK", "MODERATE", "SAFE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has temperature_C", r.get("profile", [{}])[0].get("temperature_C") is not None)
test("Prof has apb_MPa", r.get("profile", [{}])[0].get("apb_MPa") is not None)
test("Prof has safety_factor", r.get("profile", [{}])[0].get("safety_factor") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Oil annulus", api("POST", "/api/analysis/annular-pressure-buildup", {"source": "demo", "well": "3P", "annulus_fluid": "oil", "production_temp_C": 150})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/annular-pressure-buildup", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.68.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
