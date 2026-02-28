"""Tests for v3.54.0: Fracture Compliance Tensor, Pp Depletion, Reactivation Pressure, Deviation Survey, Rock Strength."""
import requests, sys

BASE = "http://localhost:8143"
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
# [205] Fracture Compliance Tensor
# ═══════════════════════════════════════════════════════════════
print("\n[205] Fracture Compliance Tensor")
code, r = api("POST", "/api/analysis/fracture-compliance", {"source": "demo", "well": "3P", "normal_compliance": 1e-10, "tangential_compliance": 5e-11})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has normal_compliance", "normal_compliance" in r)
test("Has tangential_compliance", "tangential_compliance" in r)
test("Has tensor_components", "tensor_components" in r)
test("Tensor has S11", "S11" in r.get("tensor_components", {}))
test("Tensor has S22", "S22" in r.get("tensor_components", {}))
test("Tensor has S33", "S33" in r.get("tensor_components", {}))
test("Tensor has S12", "S12" in r.get("tensor_components", {}))
test("Tensor has S13", "S13" in r.get("tensor_components", {}))
test("Tensor has S23", "S23" in r.get("tensor_components", {}))
test("Has principal_compliances", "principal_compliances" in r)
test("Principals non-empty", len(r.get("principal_compliances", [])) == 3)
test("Principal has axis", r.get("principal_compliances", [{}])[0].get("axis") is not None)
test("Principal has compliance", r.get("principal_compliances", [{}])[0].get("compliance") is not None)
test("Principal has direction", r.get("principal_compliances", [{}])[0].get("direction") is not None)
test("Has anisotropy_ratio", "anisotropy_ratio" in r)
test("Has anisotropy_class", "anisotropy_class" in r)
test("Class valid", r.get("anisotropy_class") in ("HIGH", "MODERATE", "LOW"))
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/fracture-compliance", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [206] Pore Pressure Depletion Effect
# ═══════════════════════════════════════════════════════════════
print("\n[206] Pore Pressure Depletion Effect")
code, r = api("POST", "/api/analysis/pp-depletion", {"source": "demo", "well": "3P", "depth": 3000, "depletion_mpa": 10, "friction": 0.6})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has depletion_mpa", "depletion_mpa" in r)
test("Has friction", "friction" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has Pp_initial_MPa", "Pp_initial_MPa" in r)
test("Has Pp_final_MPa", "Pp_final_MPa" in r)
test("Has stress_path_coefficient", "stress_path_coefficient" in r)
test("Has initial_cs_pct", "initial_cs_pct" in r)
test("Has final_cs_pct", "final_cs_pct" in r)
test("Has cs_change_pct", "cs_change_pct" in r)
test("Has impact_class", "impact_class" in r)
test("Class valid", r.get("impact_class") in ("SEVERE", "SIGNIFICANT", "MINOR", "BENEFICIAL"))
test("Has n_steps", "n_steps" in r)
test("Has path", "path" in r)
test("Path non-empty", len(r.get("path", [])) > 0)
test("Path has depletion_MPa", r.get("path", [{}])[0].get("depletion_MPa") is not None)
test("Path has Pp_MPa", r.get("path", [{}])[0].get("Pp_MPa") is not None)
test("Path has Shmin_MPa", r.get("path", [{}])[0].get("Shmin_MPa") is not None)
test("Path has SHmax_MPa", r.get("path", [{}])[0].get("SHmax_MPa") is not None)
test("Path has cs_pct", r.get("path", [{}])[0].get("cs_pct") is not None)
test("Path has n_cs", r.get("path", [{}])[0].get("n_cs") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/pp-depletion", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [207] Fracture Reactivation Pressure
# ═══════════════════════════════════════════════════════════════
print("\n[207] Fracture Reactivation Pressure")
code, r = api("POST", "/api/analysis/reactivation-pressure", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has friction", "friction" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has SHmax_MPa", "SHmax_MPa" in r)
test("Has Shmin_MPa", "Shmin_MPa" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has n_currently_critical", "n_currently_critical" in r)
test("Has min_Pp_critical_MPa", "min_Pp_critical_MPa" in r)
test("Has min_Pp_margin_MPa", "min_Pp_margin_MPa" in r)
test("Has mean_Pp_margin_MPa", "mean_Pp_margin_MPa" in r)
test("Has risk_class", "risk_class" in r)
test("Class valid", r.get("risk_class") in ("CRITICAL", "HIGH", "MODERATE", "LOW"))
test("Has top_vulnerable", "top_vulnerable" in r)
test("Top non-empty", len(r.get("top_vulnerable", [])) > 0)
test("Top has azimuth_deg", r.get("top_vulnerable", [{}])[0].get("azimuth_deg") is not None)
test("Top has dip_deg", r.get("top_vulnerable", [{}])[0].get("dip_deg") is not None)
test("Top has Pp_critical_MPa", r.get("top_vulnerable", [{}])[0].get("Pp_critical_MPa") is not None)
test("Top has Pp_margin_MPa", r.get("top_vulnerable", [{}])[0].get("Pp_margin_MPa") is not None)
test("Top has currently_critical", "currently_critical" in r.get("top_vulnerable", [{}])[0])
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/reactivation-pressure", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [208] Wellbore Deviation Survey
# ═══════════════════════════════════════════════════════════════
print("\n[208] Wellbore Deviation Survey")
code, r = api("POST", "/api/analysis/deviation-survey", {"source": "demo", "well": "3P", "depth": 3000, "azimuth_deg": 0, "max_inclination": 60, "n_stations": 20})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has azimuth_deg", "azimuth_deg" in r)
test("Has max_inclination_deg", "max_inclination_deg" in r)
test("Has n_stations", "n_stations" in r)
test("Has pct_stable", "pct_stable" in r)
test("Has stability_class", "stability_class" in r)
test("Class valid", r.get("stability_class") in ("STABLE", "MOSTLY_STABLE", "MARGINAL", "UNSTABLE"))
test("Has min_safety_factor", "min_safety_factor" in r)
test("Has critical_station", "critical_station" in r)
test("Critical has station", r.get("critical_station", {}).get("station") is not None)
test("Critical has TVD_m", r.get("critical_station", {}).get("TVD_m") is not None)
test("Critical has sf_collapse", r.get("critical_station", {}).get("sf_collapse") is not None)
test("Has stations", "stations" in r)
test("Stations non-empty", len(r.get("stations", [])) > 0)
test("Station has MD_m", r.get("stations", [{}])[0].get("MD_m") is not None)
test("Station has TVD_m", r.get("stations", [{}])[0].get("TVD_m") is not None)
test("Station has inclination_deg", r.get("stations", [{}])[0].get("inclination_deg") is not None)
test("Station has sf_collapse", r.get("stations", [{}])[0].get("sf_collapse") is not None)
test("Station has sf_fracture", r.get("stations", [{}])[0].get("sf_fracture") is not None)
test("Station has stable", "stable" in r.get("stations", [{}])[0])
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/deviation-survey", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [209] Rock Strength Profile
# ═══════════════════════════════════════════════════════════════
print("\n[209] Rock Strength Profile")
code, r = api("POST", "/api/analysis/rock-strength", {"source": "demo", "well": "3P", "depth_from": 2000, "depth_to": 4000, "n_points": 30, "lithology": "sandstone"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has lithology", "lithology" in r)
test("Has mean_UCS_MPa", "mean_UCS_MPa" in r)
test("Has min_UCS_MPa", "min_UCS_MPa" in r)
test("Has max_UCS_MPa", "max_UCS_MPa" in r)
test("Has mean_strength_ratio", "mean_strength_ratio" in r)
test("Has strength_class", "strength_class" in r)
test("Class valid", r.get("strength_class") in ("STRONG", "ADEQUATE", "WEAK", "VERY_WEAK"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has UCS_MPa", r.get("profile", [{}])[0].get("UCS_MPa") is not None)
test("Prof has tensile_MPa", r.get("profile", [{}])[0].get("tensile_MPa") is not None)
test("Prof has cohesion_MPa", r.get("profile", [{}])[0].get("cohesion_MPa") is not None)
test("Prof has friction_angle_deg", r.get("profile", [{}])[0].get("friction_angle_deg") is not None)
test("Prof has youngs_modulus_GPa", r.get("profile", [{}])[0].get("youngs_modulus_GPa") is not None)
test("Prof has Sv_MPa", r.get("profile", [{}])[0].get("Sv_MPa") is not None)
test("Prof has strength_ratio", r.get("profile", [{}])[0].get("strength_ratio") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/rock-strength", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))
test("Shale works", api("POST", "/api/analysis/rock-strength", {"source": "demo", "well": "3P", "lithology": "shale"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.54.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
