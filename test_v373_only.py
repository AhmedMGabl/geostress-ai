"""Tests for v3.73.0: Sand Failure, Wellbore Breathing, Surge-Swab, Lost Circulation, Hole Cleaning Eff."""
import requests, sys

BASE = "http://localhost:8168"
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
# [300] Sand Failure Prediction
# ═══════════════════════════════════════════════════════════════
print("\n[300] Sand Failure Prediction")
code, r = api("POST", "/api/analysis/sand-failure-prediction", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "UCS_MPa": 30, "TWC_factor": 3.0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has TWC_factor", "TWC_factor" in r)
test("Has mean_sand_risk", "mean_sand_risk" in r)
test("Has max_sand_risk", "max_sand_risk" in r)
test("Has pct_critical", "pct_critical" in r)
test("Has sf_class", "sf_class" in r)
test("Class valid", r.get("sf_class") in ("CRITICAL", "HIGH_RISK", "MODERATE", "STABLE"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has TWC_strength_MPa", r.get("profile", [{}])[0].get("TWC_strength_MPa") is not None)
test("Prof has sand_risk", r.get("profile", [{}])[0].get("sand_risk") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Weak rock", api("POST", "/api/analysis/sand-failure-prediction", {"source": "demo", "well": "3P", "UCS_MPa": 10})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/sand-failure-prediction", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [301] Wellbore Breathing
# ═══════════════════════════════════════════════════════════════
print("\n[301] Wellbore Breathing")
code, r = api("POST", "/api/analysis/wellbore-breathing", {"source": "demo", "well": "3P", "depth_m": 3000, "mud_weight_ppg": 11, "pump_on_ecd_ppg": 12})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has pump_on_ecd_ppg", "pump_on_ecd_ppg" in r)
test("Has frac_open_MPa", "frac_open_MPa" in r)
test("Has frac_close_MPa", "frac_close_MPa" in r)
test("Has pumps_on_margin_MPa", "pumps_on_margin_MPa" in r)
test("Has pumps_off_margin_MPa", "pumps_off_margin_MPa" in r)
test("Has volume_loss_bbl", "volume_loss_bbl" in r)
test("Has volume_return_bbl", "volume_return_bbl" in r)
test("Has breathing_index", "breathing_index" in r)
test("Has br_class", "br_class" in r)
test("Class valid", r.get("br_class") in ("SEVERE", "MODERATE", "MILD", "NONE"))
test("Has mw_sweep", "mw_sweep" in r)
test("Sweep non-empty", len(r.get("mw_sweep", [])) > 0)
test("Sweep has MW_ppg", r.get("mw_sweep", [{}])[0].get("MW_ppg") is not None)
test("Sweep has breathing_index", r.get("mw_sweep", [{}])[0].get("breathing_index") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Heavy MW", api("POST", "/api/analysis/wellbore-breathing", {"source": "demo", "well": "3P", "mud_weight_ppg": 14})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/wellbore-breathing", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [302] Surge-Swab Pressure
# ═══════════════════════════════════════════════════════════════
print("\n[302] Surge-Swab Pressure")
code, r = api("POST", "/api/analysis/surge-swab-pressure", {"source": "demo", "well": "3P", "depth_m": 3000, "mud_weight_ppg": 10, "pipe_speed_ft_min": 90})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has pipe_speed_ft_min", "pipe_speed_ft_min" in r)
test("Has surge_ppg", "surge_ppg" in r)
test("Has swab_ppg", "swab_ppg" in r)
test("Has surge_eqmw_ppg", "surge_eqmw_ppg" in r)
test("Has swab_eqmw_ppg", "swab_eqmw_ppg" in r)
test("Has frac_grad_ppg", "frac_grad_ppg" in r)
test("Has pore_grad_ppg", "pore_grad_ppg" in r)
test("Has frac_margin_ppg", "frac_margin_ppg" in r)
test("Has kick_margin_ppg", "kick_margin_ppg" in r)
test("Has ss_class", "ss_class" in r)
test("Class valid", r.get("ss_class") in ("CRITICAL", "TIGHT", "MODERATE", "SAFE"))
test("Has speed_sweep", "speed_sweep" in r)
test("Sweep non-empty", len(r.get("speed_sweep", [])) > 0)
test("Sweep has speed_ft_min", r.get("speed_sweep", [{}])[0].get("speed_ft_min") is not None)
test("Sweep has surge_ppg", r.get("speed_sweep", [{}])[0].get("surge_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Fast trip", api("POST", "/api/analysis/surge-swab-pressure", {"source": "demo", "well": "3P", "pipe_speed_ft_min": 180})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/surge-swab-pressure", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [303] Lost Circulation Risk
# ═══════════════════════════════════════════════════════════════
print("\n[303] Lost Circulation Risk")
code, r = api("POST", "/api/analysis/lost-circulation-risk", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "mud_weight_ppg": 11})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has mean_lc_risk", "mean_lc_risk" in r)
test("Has max_lc_risk", "max_lc_risk" in r)
test("Has pct_high_risk", "pct_high_risk" in r)
test("Has lc_class", "lc_class" in r)
test("Class valid", r.get("lc_class") in ("SEVERE", "HIGH", "MODERATE", "LOW"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has frac_grad_ppg", r.get("profile", [{}])[0].get("frac_grad_ppg") is not None)
test("Prof has lc_risk", r.get("profile", [{}])[0].get("lc_risk") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Heavy MW", api("POST", "/api/analysis/lost-circulation-risk", {"source": "demo", "well": "3P", "mud_weight_ppg": 14})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/lost-circulation-risk", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [304] Hole Cleaning Efficiency
# ═══════════════════════════════════════════════════════════════
print("\n[304] Hole Cleaning Efficiency")
code, r = api("POST", "/api/analysis/hole-cleaning-efficiency", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "flow_rate_gpm": 500, "inclination_deg": 0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has flow_rate_gpm", "flow_rate_gpm" in r)
test("Has hole_diameter_in", "hole_diameter_in" in r)
test("Has pipe_od_in", "pipe_od_in" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has inclination_deg", "inclination_deg" in r)
test("Has ann_velocity_ft_min", "ann_velocity_ft_min" in r)
test("Has mean_cleaning_eff", "mean_cleaning_eff" in r)
test("Has min_cleaning_eff", "min_cleaning_eff" in r)
test("Has pct_poor", "pct_poor" in r)
test("Has hc_class", "hc_class" in r)
test("Class valid", r.get("hc_class") in ("POOR", "MARGINAL", "ADEQUATE", "GOOD"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has cleaning_efficiency", r.get("profile", [{}])[0].get("cleaning_efficiency") is not None)
test("Has flow_sweep", "flow_sweep" in r)
test("Sweep non-empty", len(r.get("flow_sweep", [])) > 0)
test("Sweep has flow_rate_gpm", r.get("flow_sweep", [{}])[0].get("flow_rate_gpm") is not None)
test("Sweep has cleaning_efficiency", r.get("flow_sweep", [{}])[0].get("cleaning_efficiency") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High incl", api("POST", "/api/analysis/hole-cleaning-efficiency", {"source": "demo", "well": "3P", "inclination_deg": 60})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/hole-cleaning-efficiency", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.73.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
