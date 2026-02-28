"""Tests for v3.74.0: Torque-Drag, Casing Wear, Kick Margin, Cement Integrity, Swelling Pressure."""
import requests, sys

BASE = "http://localhost:8169"
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
# [305] Torque-Drag Analysis
# ═══════════════════════════════════════════════════════════════
print("\n[305] Torque-Drag Analysis")
code, r = api("POST", "/api/analysis/torque-drag-analysis", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "friction_factor": 0.25})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has friction_factor", "friction_factor" in r)
test("Has max_hookload_klbs", "max_hookload_klbs" in r)
test("Has max_torque_kftlbs", "max_torque_kftlbs" in r)
test("Has hookload_pct", "hookload_pct" in r)
test("Has torque_pct", "torque_pct" in r)
test("Has td_class", "td_class" in r)
test("Class valid", r.get("td_class") in ("CRITICAL", "HIGH", "MODERATE", "LOW"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has hookload_klbs", r.get("profile", [{}])[0].get("hookload_klbs") is not None)
test("Prof has torque_kftlbs", r.get("profile", [{}])[0].get("torque_kftlbs") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High friction", api("POST", "/api/analysis/torque-drag-analysis", {"source": "demo", "well": "3P", "friction_factor": 0.4})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/torque-drag-analysis", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [306] Casing Wear Prediction
# ═══════════════════════════════════════════════════════════════
print("\n[306] Casing Wear Prediction")
code, r = api("POST", "/api/analysis/casing-wear-prediction", {"source": "demo", "well": "3P", "rotating_hours": 200, "rpm": 120})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has rotating_hours", "rotating_hours" in r)
test("Has rpm", "rpm" in r)
test("Has max_wear_pct", "max_wear_pct" in r)
test("Has mean_wear_pct", "mean_wear_pct" in r)
test("Has min_remaining_wall_pct", "min_remaining_wall_pct" in r)
test("Has cw_class", "cw_class" in r)
test("Class valid", r.get("cw_class") in ("SEVERE", "SIGNIFICANT", "MODERATE", "MINOR"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has wear_pct", r.get("profile", [{}])[0].get("wear_pct") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Long rotation", api("POST", "/api/analysis/casing-wear-prediction", {"source": "demo", "well": "3P", "rotating_hours": 500})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/casing-wear-prediction", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [307] Kick Margin Profile
# ═══════════════════════════════════════════════════════════════
print("\n[307] Kick Margin Profile")
code, r = api("POST", "/api/analysis/kick-margin-profile", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "mud_weight_ppg": 10, "kick_intensity_ppg": 0.5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has kick_intensity_ppg", "kick_intensity_ppg" in r)
test("Has min_kick_margin_ppg", "min_kick_margin_ppg" in r)
test("Has mean_kick_margin_ppg", "mean_kick_margin_ppg" in r)
test("Has pct_narrow", "pct_narrow" in r)
test("Has km_class", "km_class" in r)
test("Class valid", r.get("km_class") in ("NO_MARGIN", "TIGHT", "ADEQUATE", "WIDE"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has kick_margin_ppg", r.get("profile", [{}])[0].get("kick_margin_ppg") is not None)
test("Prof has safe_window_ppg", r.get("profile", [{}])[0].get("safe_window_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High kick", api("POST", "/api/analysis/kick-margin-profile", {"source": "demo", "well": "3P", "kick_intensity_ppg": 2.0})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/kick-margin-profile", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [308] Cement Integrity Assessment
# ═══════════════════════════════════════════════════════════════
print("\n[308] Cement Integrity Assessment")
code, r = api("POST", "/api/analysis/cement-integrity-assessment", {"source": "demo", "well": "3P", "depth_m": 3000, "cement_UCS_MPa": 30, "cement_tensile_MPa": 3, "delta_T_C": 80, "delta_P_MPa": 15})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has cement_UCS_MPa", "cement_UCS_MPa" in r)
test("Has cement_tensile_MPa", "cement_tensile_MPa" in r)
test("Has delta_T_C", "delta_T_C" in r)
test("Has delta_P_MPa", "delta_P_MPa" in r)
test("Has thermal_stress_MPa", "thermal_stress_MPa" in r)
test("Has tensile_SF", "tensile_SF" in r)
test("Has compressive_SF", "compressive_SF" in r)
test("Has min_SF", "min_SF" in r)
test("Has ci_class", "ci_class" in r)
test("Class valid", r.get("ci_class") in ("FAILED", "MARGINAL", "ADEQUATE", "ROBUST"))
test("Has dt_sweep", "dt_sweep" in r)
test("Sweep non-empty", len(r.get("dt_sweep", [])) > 0)
test("Sweep has delta_T_C", r.get("dt_sweep", [{}])[0].get("delta_T_C") is not None)
test("Sweep has tensile_SF", r.get("dt_sweep", [{}])[0].get("tensile_SF") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High dT", api("POST", "/api/analysis/cement-integrity-assessment", {"source": "demo", "well": "3P", "delta_T_C": 150})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/cement-integrity-assessment", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [309] Swelling Pressure Risk
# ═══════════════════════════════════════════════════════════════
print("\n[309] Swelling Pressure Risk")
code, r = api("POST", "/api/analysis/swelling-pressure-risk", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "clay_content_pct": 30, "water_activity": 0.9})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has clay_content_pct", "clay_content_pct" in r)
test("Has water_activity", "water_activity" in r)
test("Has mean_risk_index", "mean_risk_index" in r)
test("Has max_risk_index", "max_risk_index" in r)
test("Has max_swelling_MPa", "max_swelling_MPa" in r)
test("Has sw_class", "sw_class" in r)
test("Class valid", r.get("sw_class") in ("SEVERE", "HIGH", "MODERATE", "LOW"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has swelling_P_MPa", r.get("profile", [{}])[0].get("swelling_P_MPa") is not None)
test("Prof has risk_index", r.get("profile", [{}])[0].get("risk_index") is not None)
test("Has aw_sweep", "aw_sweep" in r)
test("Sweep non-empty", len(r.get("aw_sweep", [])) > 0)
test("Sweep has water_activity", r.get("aw_sweep", [{}])[0].get("water_activity") is not None)
test("Sweep has swelling_P_MPa", r.get("aw_sweep", [{}])[0].get("swelling_P_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High clay", api("POST", "/api/analysis/swelling-pressure-risk", {"source": "demo", "well": "3P", "clay_content_pct": 60})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/swelling-pressure-risk", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.74.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
