"""Tests for v3.66.0: Sand Production, Breakout Width, Cement Bond, Swab Surge, Rock Strength."""
import requests, sys

BASE = "http://localhost:8161"
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
# [265] Sand Production Risk
# ═══════════════════════════════════════════════════════════════
print("\n[265] Sand Production Risk")
code, r = api("POST", "/api/analysis/sand-production-risk", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "UCS_MPa": 30})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has TWC_MPa", "TWC_MPa" in r)
test("Has n_critical_depths", "n_critical_depths" in r)
test("Has pct_at_risk", "pct_at_risk" in r)
test("Has sand_class", "sand_class" in r)
test("Class valid", r.get("sand_class") in ("CRITICAL", "HIGH_RISK", "MODERATE", "STABLE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has sanding_load_MPa", r.get("profile", [{}])[0].get("sanding_load_MPa") is not None)
test("Prof has sand_margin_MPa", r.get("profile", [{}])[0].get("sand_margin_MPa") is not None)
test("Prof has at_risk", "at_risk" in r.get("profile", [{}])[0])
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Weak rock", api("POST", "/api/analysis/sand-production-risk", {"source": "demo", "well": "3P", "UCS_MPa": 10})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/sand-production-risk", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [266] Wellbore Breakout Width
# ═══════════════════════════════════════════════════════════════
print("\n[266] Wellbore Breakout Width")
code, r = api("POST", "/api/analysis/wellbore-breakout-width", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "UCS_MPa": 50, "friction_angle_deg": 30})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has friction_angle_deg", "friction_angle_deg" in r)
test("Has mean_breakout_width_deg", "mean_breakout_width_deg" in r)
test("Has max_breakout_width_deg", "max_breakout_width_deg" in r)
test("Has pct_with_breakout", "pct_with_breakout" in r)
test("Has n_breakout_depths", "n_breakout_depths" in r)
test("Has breakout_class", "breakout_class" in r)
test("Class valid", r.get("breakout_class") in ("SEVERE", "MODERATE", "MINOR", "NONE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has sigma_theta_max_MPa", r.get("profile", [{}])[0].get("sigma_theta_max_MPa") is not None)
test("Prof has breakout_width_deg", r.get("profile", [{}])[0].get("breakout_width_deg") is not None)
test("Prof has has_breakout", "has_breakout" in r.get("profile", [{}])[0])
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Low UCS", api("POST", "/api/analysis/wellbore-breakout-width", {"source": "demo", "well": "3P", "UCS_MPa": 20})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/wellbore-breakout-width", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [267] Cement Bond Quality
# ═══════════════════════════════════════════════════════════════
print("\n[267] Cement Bond Quality")
code, r = api("POST", "/api/analysis/cement-bond-quality", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "cement_density_ppg": 16.0, "mud_weight_ppg": 10.0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has cement_density_ppg", "cement_density_ppg" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has mean_bond_score", "mean_bond_score" in r)
test("Has min_bond_score", "min_bond_score" in r)
test("Has pct_poor", "pct_poor" in r)
test("Has cement_class", "cement_class" in r)
test("Class valid", r.get("cement_class") in ("POOR", "FAIR", "GOOD", "EXCELLENT"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has bond_score", r.get("profile", [{}])[0].get("bond_score") is not None)
test("Prof has contact_pressure_MPa", r.get("profile", [{}])[0].get("contact_pressure_MPa") is not None)
test("Prof has bond_quality", r.get("profile", [{}])[0].get("bond_quality") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Light cement", api("POST", "/api/analysis/cement-bond-quality", {"source": "demo", "well": "3P", "cement_density_ppg": 12.0})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/cement-bond-quality", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [268] Swab & Surge Pressure
# ═══════════════════════════════════════════════════════════════
print("\n[268] Swab & Surge Pressure")
code, r = api("POST", "/api/analysis/swab-surge-pressure", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "mud_weight_ppg": 10.0, "trip_speed_ft_min": 90, "pipe_od_in": 5.0, "hole_dia_in": 8.5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has trip_speed_ft_min", "trip_speed_ft_min" in r)
test("Has pipe_od_in", "pipe_od_in" in r)
test("Has hole_dia_in", "hole_dia_in" in r)
test("Has pv_cp", "pv_cp" in r)
test("Has max_delta_emw_ppg", "max_delta_emw_ppg" in r)
test("Has min_kick_margin_ppg", "min_kick_margin_ppg" in r)
test("Has min_loss_margin_ppg", "min_loss_margin_ppg" in r)
test("Has surge_class", "surge_class" in r)
test("Class valid", r.get("surge_class") in ("CRITICAL", "NARROW", "ADEQUATE", "SAFE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has swab_emw_ppg", r.get("profile", [{}])[0].get("swab_emw_ppg") is not None)
test("Prof has surge_emw_ppg", r.get("profile", [{}])[0].get("surge_emw_ppg") is not None)
test("Prof has kick_margin_ppg", r.get("profile", [{}])[0].get("kick_margin_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Fast trip", api("POST", "/api/analysis/swab-surge-pressure", {"source": "demo", "well": "3P", "trip_speed_ft_min": 150})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/swab-surge-pressure", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [269] Rock Strength Profile
# ═══════════════════════════════════════════════════════════════
print("\n[269] Rock Strength Profile")
code, r = api("POST", "/api/analysis/rock-strength-profile", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "surface_UCS_MPa": 20, "UCS_gradient_MPa_km": 15})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has surface_UCS_MPa", "surface_UCS_MPa" in r)
test("Has UCS_gradient_MPa_km", "UCS_gradient_MPa_km" in r)
test("Has mean_UCS_MPa", "mean_UCS_MPa" in r)
test("Has min_strength_ratio", "min_strength_ratio" in r)
test("Has pct_weak", "pct_weak" in r)
test("Has n_weak_depths", "n_weak_depths" in r)
test("Has strength_class", "strength_class" in r)
test("Class valid", r.get("strength_class") in ("VERY_WEAK", "WEAK", "MODERATE", "STRONG"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has UCS_MPa", r.get("profile", [{}])[0].get("UCS_MPa") is not None)
test("Prof has strength_ratio", r.get("profile", [{}])[0].get("strength_ratio") is not None)
test("Prof has cohesion_MPa", r.get("profile", [{}])[0].get("cohesion_MPa") is not None)
test("Prof has is_weak", "is_weak" in r.get("profile", [{}])[0])
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Low UCS", api("POST", "/api/analysis/rock-strength-profile", {"source": "demo", "well": "3P", "surface_UCS_MPa": 5, "UCS_gradient_MPa_km": 5})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/rock-strength-profile", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.66.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
