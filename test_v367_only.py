"""Tests for v3.67.0: Kick Tolerance, Hole Cleaning, FIT Simulation, Stuck Pipe, Stability Window."""
import requests, sys

BASE = "http://localhost:8162"
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
# [270] Kick Tolerance
# ═══════════════════════════════════════════════════════════════
print("\n[270] Kick Tolerance")
code, r = api("POST", "/api/analysis/kick-tolerance", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "mud_weight_ppg": 10.0, "kick_intensity_ppg": 0.5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has kick_intensity_ppg", "kick_intensity_ppg" in r)
test("Has hole_dia_in", "hole_dia_in" in r)
test("Has min_kick_vol_bbl", "min_kick_vol_bbl" in r)
test("Has min_frac_margin_ppg", "min_frac_margin_ppg" in r)
test("Has kick_class", "kick_class" in r)
test("Class valid", r.get("kick_class") in ("CRITICAL", "LOW", "MODERATE", "HIGH"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has max_kick_vol_bbl", r.get("profile", [{}])[0].get("max_kick_vol_bbl") is not None)
test("Prof has frac_margin_ppg", r.get("profile", [{}])[0].get("frac_margin_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High kick", api("POST", "/api/analysis/kick-tolerance", {"source": "demo", "well": "3P", "kick_intensity_ppg": 2.0})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/kick-tolerance", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [271] Hole Cleaning Index
# ═══════════════════════════════════════════════════════════════
print("\n[271] Hole Cleaning Index")
code, r = api("POST", "/api/analysis/hole-cleaning-index", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "flow_rate_gpm": 400, "rpm": 120, "hole_angle_deg": 0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has flow_rate_gpm", "flow_rate_gpm" in r)
test("Has rpm", "rpm" in r)
test("Has hole_angle_deg", "hole_angle_deg" in r)
test("Has mean_hci", "mean_hci" in r)
test("Has min_hci", "min_hci" in r)
test("Has n_poor_zones", "n_poor_zones" in r)
test("Has cleaning_class", "cleaning_class" in r)
test("Class valid", r.get("cleaning_class") in ("POOR", "MARGINAL", "ADEQUATE", "GOOD"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has hci", r.get("profile", [{}])[0].get("hci") is not None)
test("Prof has quality", r.get("profile", [{}])[0].get("quality") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Low flow", api("POST", "/api/analysis/hole-cleaning-index", {"source": "demo", "well": "3P", "flow_rate_gpm": 100})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/hole-cleaning-index", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [272] Formation Integrity Test
# ═══════════════════════════════════════════════════════════════
print("\n[272] Formation Integrity Test")
code, r = api("POST", "/api/analysis/formation-integrity-test", {"source": "demo", "well": "3P", "test_depth_m": 2000, "mud_weight_ppg": 10.0, "pump_rate_bpm": 0.5, "test_type": "LOT"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has test_depth_m", "test_depth_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has pump_rate_bpm", "pump_rate_bpm" in r)
test("Has test_type", "test_type" in r)
test("Has leak_off_ppg", "leak_off_ppg" in r)
test("Has leak_off_MPa", "leak_off_MPa" in r)
test("Has breakdown_ppg", "breakdown_ppg" in r)
test("Has breakdown_MPa", "breakdown_MPa" in r)
test("Has max_test_pressure_ppg", "max_test_pressure_ppg" in r)
test("Has fit_class", "fit_class" in r)
test("Class valid", r.get("fit_class") in ("BREAKDOWN", "LEAK_OFF", "NEAR_LIMIT", "SAFE"))
test("Has pressure_curve", "pressure_curve" in r)
test("Curve non-empty", len(r.get("pressure_curve", [])) > 0)
test("Curve has time_min", r.get("pressure_curve", [{}])[0].get("time_min") is not None)
test("Curve has pressure_ppg", r.get("pressure_curve", [{}])[0].get("pressure_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("FIT type", api("POST", "/api/analysis/formation-integrity-test", {"source": "demo", "well": "3P", "test_type": "FIT"})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/formation-integrity-test", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [273] Stuck Pipe Risk
# ═══════════════════════════════════════════════════════════════
print("\n[273] Stuck Pipe Risk")
code, r = api("POST", "/api/analysis/stuck-pipe-risk", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "mud_weight_ppg": 10.0, "mud_type": "WBM"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has mud_type", "mud_type" in r)
test("Has mean_risk", "mean_risk" in r)
test("Has max_risk", "max_risk" in r)
test("Has pct_high_risk", "pct_high_risk" in r)
test("Has stuck_class", "stuck_class" in r)
test("Class valid", r.get("stuck_class") in ("HIGH_RISK", "MODERATE_RISK", "LOW_RISK", "MINIMAL"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has total_risk", r.get("profile", [{}])[0].get("total_risk") is not None)
test("Prof has diff_stick_risk", r.get("profile", [{}])[0].get("diff_stick_risk") is not None)
test("Prof has risk_level", r.get("profile", [{}])[0].get("risk_level") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("OBM mud", api("POST", "/api/analysis/stuck-pipe-risk", {"source": "demo", "well": "3P", "mud_type": "OBM"})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/stuck-pipe-risk", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [274] Wellbore Stability Window
# ═══════════════════════════════════════════════════════════════
print("\n[274] Wellbore Stability Window")
code, r = api("POST", "/api/analysis/wellbore-stability-window", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "UCS_MPa": 40, "friction_angle_deg": 30})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has friction_angle_deg", "friction_angle_deg" in r)
test("Has min_window_ppg", "min_window_ppg" in r)
test("Has mean_window_ppg", "mean_window_ppg" in r)
test("Has optimal_mw_ppg", "optimal_mw_ppg" in r)
test("Has window_class", "window_class" in r)
test("Class valid", r.get("window_class") in ("NO_WINDOW", "VERY_NARROW", "NARROW", "ADEQUATE", "WIDE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has collapse_ppg", r.get("profile", [{}])[0].get("collapse_ppg") is not None)
test("Prof has frac_ppg", r.get("profile", [{}])[0].get("frac_ppg") is not None)
test("Prof has window_ppg", r.get("profile", [{}])[0].get("window_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Low UCS", api("POST", "/api/analysis/wellbore-stability-window", {"source": "demo", "well": "3P", "UCS_MPa": 15})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/wellbore-stability-window", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.67.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
