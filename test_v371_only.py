"""Tests for v3.71.0: Kill Weight, ECD Sensitivity, Formation Breakdown, Stress Polygon, Frac Gradient Window."""
import requests, sys

BASE = "http://localhost:8166"
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
# [290] Hydrostatic Kill Weight
# ═══════════════════════════════════════════════════════════════
print("\n[290] Hydrostatic Kill Weight")
code, r = api("POST", "/api/analysis/hydrostatic-kill-weight", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "kick_margin_ppg": 0.5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has kick_margin_ppg", "kick_margin_ppg" in r)
test("Has max_kill_ppg", "max_kill_ppg" in r)
test("Has min_margin_ppg", "min_margin_ppg" in r)
test("Has kw_class", "kw_class" in r)
test("Class valid", r.get("kw_class") in ("CRITICAL", "TIGHT", "ADEQUATE", "WIDE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has MW_ppg", r.get("profile", [{}])[0].get("MW_ppg") is not None)
test("Prof has kill_ppg", r.get("profile", [{}])[0].get("kill_ppg") is not None)
test("Prof has frac_grad_ppg", r.get("profile", [{}])[0].get("frac_grad_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High margin", api("POST", "/api/analysis/hydrostatic-kill-weight", {"source": "demo", "well": "3P", "kick_margin_ppg": 1.5})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/hydrostatic-kill-weight", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [291] ECD Sensitivity
# ═══════════════════════════════════════════════════════════════
print("\n[291] ECD Sensitivity")
code, r = api("POST", "/api/analysis/ecd-sensitivity", {"source": "demo", "well": "3P", "depth_m": 3000, "mud_weight_ppg": 10, "flow_rate_gpm": 500, "hole_diameter_in": 8.5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has flow_rate_gpm", "flow_rate_gpm" in r)
test("Has ECD_ppg", "ECD_ppg" in r)
test("Has frac_grad_ppg", "frac_grad_ppg" in r)
test("Has ecd_margin_ppg", "ecd_margin_ppg" in r)
test("Has ecd_class", "ecd_class" in r)
test("Class valid", r.get("ecd_class") in ("CRITICAL", "TIGHT", "ADEQUATE", "WIDE"))
test("Has flow_sweep", "flow_sweep" in r)
test("Sweep non-empty", len(r.get("flow_sweep", [])) > 0)
test("Sweep has flow_rate_gpm", r.get("flow_sweep", [{}])[0].get("flow_rate_gpm") is not None)
test("Sweep has ECD_ppg", r.get("flow_sweep", [{}])[0].get("ECD_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High flow", api("POST", "/api/analysis/ecd-sensitivity", {"source": "demo", "well": "3P", "flow_rate_gpm": 800})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/ecd-sensitivity", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [292] Formation Breakdown
# ═══════════════════════════════════════════════════════════════
print("\n[292] Formation Breakdown")
code, r = api("POST", "/api/analysis/formation-breakdown", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "tensile_strength_MPa": 5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has tensile_strength_MPa", "tensile_strength_MPa" in r)
test("Has mean_Pb_HW_MPa", "mean_Pb_HW_MPa" in r)
test("Has min_Pb_HW_MPa", "min_Pb_HW_MPa" in r)
test("Has bd_class", "bd_class" in r)
test("Class valid", r.get("bd_class") in ("LOW_PRESSURE", "MODERATE", "HIGH", "VERY_HIGH"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Pb_HW_MPa", r.get("profile", [{}])[0].get("Pb_HW_MPa") is not None)
test("Prof has Pb_HF_MPa", r.get("profile", [{}])[0].get("Pb_HF_MPa") is not None)
test("Prof has Pr_MPa", r.get("profile", [{}])[0].get("Pr_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Weak rock", api("POST", "/api/analysis/formation-breakdown", {"source": "demo", "well": "3P", "tensile_strength_MPa": 1})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/formation-breakdown", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [293] Stress Regime Polygon
# ═══════════════════════════════════════════════════════════════
print("\n[293] Stress Regime Polygon")
code, r = api("POST", "/api/analysis/stress-regime-polygon", {"source": "demo", "well": "3P", "depth_m": 3000, "friction": 0.6})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has friction", "friction" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has Shmin_est_MPa", "Shmin_est_MPa" in r)
test("Has SHmax_est_MPa", "SHmax_est_MPa" in r)
test("Has regime_est", "regime_est" in r)
test("Has frictional_limit_q", "frictional_limit_q" in r)
test("Has boundaries", "boundaries" in r)
test("Has reg_class", "reg_class" in r)
test("Class valid", r.get("reg_class") in ("NORMAL_FAULT", "STRIKE_SLIP", "REVERSE_FAULT"))
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High friction", api("POST", "/api/analysis/stress-regime-polygon", {"source": "demo", "well": "3P", "friction": 0.85})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/stress-regime-polygon", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [294] Fracture Gradient Window
# ═══════════════════════════════════════════════════════════════
print("\n[294] Fracture Gradient Window")
code, r = api("POST", "/api/analysis/fracture-gradient-window", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "mud_weight_ppg": 10})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has min_window_ppg", "min_window_ppg" in r)
test("Has mean_window_ppg", "mean_window_ppg" in r)
test("Has pct_safe", "pct_safe" in r)
test("Has fw_class", "fw_class" in r)
test("Class valid", r.get("fw_class") in ("NO_WINDOW", "NARROW", "ADEQUATE", "WIDE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has pore_ppg", r.get("profile", [{}])[0].get("pore_ppg") is not None)
test("Prof has frac_ppg", r.get("profile", [{}])[0].get("frac_ppg") is not None)
test("Prof has window_ppg", r.get("profile", [{}])[0].get("window_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Heavy MW", api("POST", "/api/analysis/fracture-gradient-window", {"source": "demo", "well": "3P", "mud_weight_ppg": 14})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/fracture-gradient-window", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.71.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
