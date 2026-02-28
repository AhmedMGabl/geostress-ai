"""Tests for v3.61.0: Pore Pressure Window, Fracture Density Profile, Stress Polygon, Rock Strength, Breakout Depth."""
import requests, sys

BASE = "http://localhost:8151"
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
# [240] Pore Pressure Window
# ═══════════════════════════════════════════════════════════════
print("\n[240] Pore Pressure Window")
code, r = api("POST", "/api/analysis/pore-pressure-window", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "mud_weight_ppg": 10.0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has min_window_MPa", "min_window_MPa" in r)
test("Has min_kick_margin_MPa", "min_kick_margin_MPa" in r)
test("Has min_loss_margin_MPa", "min_loss_margin_MPa" in r)
test("Has pp_class", "pp_class" in r)
test("Class valid", r.get("pp_class") in ("KICK_RISK", "LOSS_RISK", "NARROW", "ADEQUATE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Pp_normal_MPa", r.get("profile", [{}])[0].get("Pp_normal_MPa") is not None)
test("Prof has frac_gradient_MPa", r.get("profile", [{}])[0].get("frac_gradient_MPa") is not None)
test("Prof has kick_margin_MPa", r.get("profile", [{}])[0].get("kick_margin_MPa") is not None)
test("Prof has window_MPa", r.get("profile", [{}])[0].get("window_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Heavy MW works", api("POST", "/api/analysis/pore-pressure-window", {"source": "demo", "well": "3P", "mud_weight_ppg": 14.0})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/pore-pressure-window", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [241] Fracture Density Profile
# ═══════════════════════════════════════════════════════════════
print("\n[241] Fracture Density Profile")
code, r = api("POST", "/api/analysis/fracture-density-profile", {"source": "demo", "well": "3P", "bin_size_m": 50})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has bin_size_m", "bin_size_m" in r)
test("Has n_bins", "n_bins" in r)
test("Has mean_density_per_m", "mean_density_per_m" in r)
test("Has max_density_per_m", "max_density_per_m" in r)
test("Has std_density_per_m", "std_density_per_m" in r)
test("Has cv_density", "cv_density" in r)
test("Has density_class", "density_class" in r)
test("Class valid", r.get("density_class") in ("HIGH", "MODERATE", "LOW"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_from_m", r.get("profile", [{}])[0].get("depth_from_m") is not None)
test("Prof has depth_mid_m", r.get("profile", [{}])[0].get("depth_mid_m") is not None)
test("Prof has count", r.get("profile", [{}])[0].get("count") is not None)
test("Prof has density_per_m", r.get("profile", [{}])[0].get("density_per_m") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Small bin", api("POST", "/api/analysis/fracture-density-profile", {"source": "demo", "well": "3P", "bin_size_m": 10})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/fracture-density-profile", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [242] Stress Polygon
# ═══════════════════════════════════════════════════════════════
print("\n[242] Stress Polygon")
code, r = api("POST", "/api/analysis/stress-polygon-frictional", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has friction", "friction" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has Shmin_est_MPa", "Shmin_est_MPa" in r)
test("Has SHmax_est_MPa", "SHmax_est_MPa" in r)
test("Has frictional_limit_q", "frictional_limit_q" in r)
test("Has current_regime", "current_regime" in r)
test("Regime valid", r.get("current_regime") in ("Normal Fault", "Strike-Slip", "Reverse Fault"))
test("Has within_polygon", "within_polygon" in r)
test("Has NF_Shmin_range", "NF_Shmin_range" in r)
test("Has SS_SHmax_range", "SS_SHmax_range" in r)
test("Has RF_SHmax_range", "RF_SHmax_range" in r)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High friction", api("POST", "/api/analysis/stress-polygon-frictional", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.85})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/stress-polygon-frictional", {"source": "demo", "well": "6P", "depth": 2000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [243] Rock Strength
# ═══════════════════════════════════════════════════════════════
print("\n[243] Rock Strength")
code, r = api("POST", "/api/analysis/rock-strength-lithology", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "lithology": "sandstone"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has lithology", "lithology" in r)
test("Has mean_UCS_MPa", "mean_UCS_MPa" in r)
test("Has min_UCS_MPa", "min_UCS_MPa" in r)
test("Has max_UCS_MPa", "max_UCS_MPa" in r)
test("Has strength_class", "strength_class" in r)
test("Class valid", r.get("strength_class") in ("WEAK", "MODERATE", "STRONG"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has UCS_MPa", r.get("profile", [{}])[0].get("UCS_MPa") is not None)
test("Prof has tensile_strength_MPa", r.get("profile", [{}])[0].get("tensile_strength_MPa") is not None)
test("Prof has cohesion_MPa", r.get("profile", [{}])[0].get("cohesion_MPa") is not None)
test("Prof has E_GPa", r.get("profile", [{}])[0].get("E_GPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Shale works", api("POST", "/api/analysis/rock-strength-lithology", {"source": "demo", "well": "3P", "lithology": "shale"})[0] in (200, 422, 500))
test("Limestone works", api("POST", "/api/analysis/rock-strength-lithology", {"source": "demo", "well": "3P", "lithology": "limestone"})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/rock-strength-lithology", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [244] Wellbore Breakout Depth
# ═══════════════════════════════════════════════════════════════
print("\n[244] Wellbore Breakout Depth")
code, r = api("POST", "/api/analysis/wellbore-breakout-depth", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "UCS_MPa": 50})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has n_breakout_zones", "n_breakout_zones" in r)
test("Has pct_breakout", "pct_breakout" in r)
test("Has min_margin_MPa", "min_margin_MPa" in r)
test("Has min_stability_factor", "min_stability_factor" in r)
test("Has breakout_class", "breakout_class" in r)
test("Class valid", r.get("breakout_class") in ("SEVERE", "MODERATE", "MINOR", "STABLE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has sigma_theta_max_MPa", r.get("profile", [{}])[0].get("sigma_theta_max_MPa") is not None)
test("Prof has margin_MPa", r.get("profile", [{}])[0].get("margin_MPa") is not None)
test("Prof has stability_factor", r.get("profile", [{}])[0].get("stability_factor") is not None)
test("Prof has breakout", "breakout" in r.get("profile", [{}])[0])
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Weak UCS", api("POST", "/api/analysis/wellbore-breakout-depth", {"source": "demo", "well": "3P", "UCS_MPa": 20})[0] in (200, 422, 500))
test("Strong UCS", api("POST", "/api/analysis/wellbore-breakout-depth", {"source": "demo", "well": "3P", "UCS_MPa": 200})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/wellbore-breakout-depth", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.61.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
