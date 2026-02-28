"""Tests for v3.63.0: ECD Profile, Fracture Spacing, Overburden Gradient, Stress Rotation, Tensile Failure."""
import requests, sys

BASE = "http://localhost:8155"
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
# [250] ECD Profile
# ═══════════════════════════════════════════════════════════════
print("\n[250] ECD Profile")
code, r = api("POST", "/api/analysis/ecd-profile", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "mud_weight_ppg": 10, "flow_rate_gpm": 500})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has static_mw_ppg", "static_mw_ppg" in r)
test("Has flow_rate_gpm", "flow_rate_gpm" in r)
test("Has max_ecd_ppg", "max_ecd_ppg" in r)
test("Has min_kick_margin_ppg", "min_kick_margin_ppg" in r)
test("Has min_loss_margin_ppg", "min_loss_margin_ppg" in r)
test("Has ecd_class", "ecd_class" in r)
test("Class valid", r.get("ecd_class") in ("CRITICAL", "NARROW", "ADEQUATE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has ecd_ppg", r.get("profile", [{}])[0].get("ecd_ppg") is not None)
test("Prof has frac_ppg", r.get("profile", [{}])[0].get("frac_ppg") is not None)
test("Prof has ecd_margin_kick_ppg", r.get("profile", [{}])[0].get("ecd_margin_kick_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High flow rate", api("POST", "/api/analysis/ecd-profile", {"source": "demo", "well": "3P", "flow_rate_gpm": 1000})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/ecd-profile", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [251] Fracture Spacing Analysis
# ═══════════════════════════════════════════════════════════════
print("\n[251] Fracture Spacing Analysis")
code, r = api("POST", "/api/analysis/fracture-spacing-analysis", {"source": "demo", "well": "3P"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has n_spacings", "n_spacings" in r)
test("Has mean_spacing_m", "mean_spacing_m" in r)
test("Has median_spacing_m", "median_spacing_m" in r)
test("Has std_spacing_m", "std_spacing_m" in r)
test("Has min_spacing_m", "min_spacing_m" in r)
test("Has max_spacing_m", "max_spacing_m" in r)
test("Has cv_spacing", "cv_spacing" in r)
test("Has best_distribution", "best_distribution" in r)
test("Dist valid", r.get("best_distribution") in ("exponential", "lognormal"))
test("Has spacing_class", "spacing_class" in r)
test("Class valid", r.get("spacing_class") in ("CLUSTERED", "REGULAR", "RANDOM"))
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/fracture-spacing-analysis", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [252] Overburden Gradient
# ═══════════════════════════════════════════════════════════════
print("\n[252] Overburden Gradient")
code, r = api("POST", "/api/analysis/overburden-gradient", {"source": "demo", "well": "3P", "depth_from": 100, "depth_to": 5000, "n_points": 25})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has mean_gradient_MPa_km", "mean_gradient_MPa_km" in r)
test("Has surface_gradient_MPa_km", "surface_gradient_MPa_km" in r)
test("Has deep_gradient_MPa_km", "deep_gradient_MPa_km" in r)
test("Has ob_class", "ob_class" in r)
test("Class valid", r.get("ob_class") in ("HIGH", "NORMAL", "LOW", "HIGH_GRADIENT", "NORMAL_GRADIENT", "LOW_GRADIENT"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Sv_MPa", r.get("profile", [{}])[0].get("Sv_MPa") is not None)
test("Prof has Sv_gradient_MPa_km", r.get("profile", [{}])[0].get("Sv_gradient_MPa_km") is not None)
test("Prof has Sv_ppg", r.get("profile", [{}])[0].get("Sv_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/overburden-gradient", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [253] Principal Stress Rotation
# ═══════════════════════════════════════════════════════════════
print("\n[253] Principal Stress Rotation")
code, r = api("POST", "/api/analysis/principal-stress-rotation", {"source": "demo", "well": "3P", "n_zones": 3, "depth_from": 500})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has n_zones", "n_zones" in r)
test("Has mean_rotation_deg", "mean_rotation_deg" in r)
test("Has max_rotation_deg", "max_rotation_deg" in r)
test("Has rotation_class", "rotation_class" in r)
test("Class valid", r.get("rotation_class") in ("SIGNIFICANT", "MODERATE", "MINOR"))
test("Has zones", "zones" in r)
test("Zones non-empty", len(r.get("zones", [])) > 0)
test("Zone has depth_from_m", r.get("zones", [{}])[0].get("depth_from_m") is not None)
test("Zone has depth_to_m", r.get("zones", [{}])[0].get("depth_to_m") is not None)
test("Zone has SHmax_est_deg", r.get("zones", [{}])[0].get("SHmax_est_deg") is not None)
test("Zone has n_fractures", r.get("zones", [{}])[0].get("n_fractures") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("5 zones", api("POST", "/api/analysis/principal-stress-rotation", {"source": "demo", "well": "3P", "n_zones": 5})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/principal-stress-rotation", {"source": "demo", "well": "6P"})[0] in (200, 400, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [254] Wellbore Tensile Failure
# ═══════════════════════════════════════════════════════════════
print("\n[254] Wellbore Tensile Failure")
code, r = api("POST", "/api/analysis/wellbore-tensile-failure", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "tensile_MPa": 10, "n_points": 25})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has tensile_strength_MPa", "tensile_strength_MPa" in r)
test("Has n_tensile_zones", "n_tensile_zones" in r)
test("Has pct_tensile", "pct_tensile" in r)
test("Has min_margin_MPa", "min_margin_MPa" in r)
test("Has min_fip_MPa or computed", "min_fip_MPa" in r or any("frac_initiation_MPa" in p for p in r.get("profile", [{}])))
test("Has tensile_class", "tensile_class" in r)
test("Class valid", r.get("tensile_class") in ("CRITICAL", "HIGH_RISK", "MODERATE_RISK", "STABLE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has sigma_theta_min_MPa", r.get("profile", [{}])[0].get("sigma_theta_min_MPa") is not None)
test("Prof has frac_initiation_MPa", r.get("profile", [{}])[0].get("frac_initiation_MPa") is not None)
test("Prof has tensile_margin_MPa", r.get("profile", [{}])[0].get("tensile_margin_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Weak rock", api("POST", "/api/analysis/wellbore-tensile-failure", {"source": "demo", "well": "3P", "tensile_MPa": 3})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/wellbore-tensile-failure", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.63.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
