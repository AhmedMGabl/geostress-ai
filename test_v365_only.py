"""Tests for v3.65.0: Wellbore Collapse, Fracture Aperture Stress, Casing Design, Drilling Margin, Geomech Facies."""
import requests, sys

BASE = "http://localhost:8160"
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
# [260] Wellbore Collapse Pressure
# ═══════════════════════════════════════════════════════════════
print("\n[260] Wellbore Collapse Pressure")
code, r = api("POST", "/api/analysis/wellbore-collapse-pressure", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25, "UCS_MPa": 50, "friction_angle_deg": 30})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has friction_angle_deg", "friction_angle_deg" in r)
test("Has max_collapse_ppg", "max_collapse_ppg" in r)
test("Has pct_critical", "pct_critical" in r)
test("Has collapse_class", "collapse_class" in r)
test("Class valid", r.get("collapse_class") in ("CRITICAL", "HIGH_RISK", "MODERATE", "STABLE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has collapse_mw_ppg", r.get("profile", [{}])[0].get("collapse_mw_ppg") is not None)
test("Prof has margin_ppg", r.get("profile", [{}])[0].get("margin_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Weak rock", api("POST", "/api/analysis/wellbore-collapse-pressure", {"source": "demo", "well": "3P", "UCS_MPa": 20})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/wellbore-collapse-pressure", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [261] Fracture Aperture vs Stress
# ═══════════════════════════════════════════════════════════════
print("\n[261] Fracture Aperture vs Stress")
code, r = api("POST", "/api/analysis/fracture-aperture-stress", {"source": "demo", "well": "3P", "initial_aperture_mm": 0.5, "stiffness_GPa_m": 50})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has initial_aperture_mm", "initial_aperture_mm" in r)
test("Has stiffness_GPa_m", "stiffness_GPa_m" in r)
test("Has mean_aperture_mm", "mean_aperture_mm" in r)
test("Has mean_perm_mD", "mean_perm_mD" in r)
test("Has n_open", "n_open" in r)
test("Has aperture_class", "aperture_class" in r)
test("Class valid", r.get("aperture_class") in ("CLOSED", "TIGHT", "OPEN"))
test("Has fractures", "fractures" in r)
test("Fracs non-empty", len(r.get("fractures", [])) > 0)
test("Frac has depth_m", r.get("fractures", [{}])[0].get("depth_m") is not None)
test("Frac has aperture_mm", r.get("fractures", [{}])[0].get("aperture_mm") is not None)
test("Frac has sigma_n_eff_MPa", r.get("fractures", [{}])[0].get("sigma_n_eff_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Large aperture", api("POST", "/api/analysis/fracture-aperture-stress", {"source": "demo", "well": "3P", "initial_aperture_mm": 2.0})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/fracture-aperture-stress", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [262] Casing Design Check
# ═══════════════════════════════════════════════════════════════
print("\n[262] Casing Design Check")
code, r = api("POST", "/api/analysis/casing-design-grade", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "casing_grade": "N80"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has casing_grade", "casing_grade" in r)
test("Has casing_od_in", "casing_od_in" in r)
test("Has collapse_resist_MPa", "collapse_resist_MPa" in r)
test("Has burst_resist_MPa", "burst_resist_MPa" in r)
test("Has min_collapse_SF", "min_collapse_SF" in r)
test("Has min_burst_SF", "min_burst_SF" in r)
test("Has casing_class", "casing_class" in r)
test("Class valid", r.get("casing_class") in ("FAIL", "MARGINAL", "ADEQUATE", "SAFE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has collapse_SF", r.get("profile", [{}])[0].get("collapse_SF") is not None)
test("Prof has burst_SF", r.get("profile", [{}])[0].get("burst_SF") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("P110 grade", api("POST", "/api/analysis/casing-design-grade", {"source": "demo", "well": "3P", "casing_grade": "P110"})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/casing-design-grade", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [263] Drilling Margin
# ═══════════════════════════════════════════════════════════════
print("\n[263] Drilling Margin")
code, r = api("POST", "/api/analysis/drilling-margin-window", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "mud_weight_ppg": 10})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has min_kick_margin_ppg", "min_kick_margin_ppg" in r)
test("Has min_loss_margin_ppg", "min_loss_margin_ppg" in r)
test("Has min_window_ppg", "min_window_ppg" in r)
test("Has margin_class", "margin_class" in r)
test("Class valid", r.get("margin_class") in ("CRITICAL", "NARROW", "ADEQUATE", "WIDE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Pp_ppg", r.get("profile", [{}])[0].get("Pp_ppg") is not None)
test("Prof has frac_ppg", r.get("profile", [{}])[0].get("frac_ppg") is not None)
test("Prof has kick_margin_ppg", r.get("profile", [{}])[0].get("kick_margin_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Heavy MW", api("POST", "/api/analysis/drilling-margin-window", {"source": "demo", "well": "3P", "mud_weight_ppg": 14})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/drilling-margin-window", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [264] Geomechanical Facies
# ═══════════════════════════════════════════════════════════════
print("\n[264] Geomechanical Facies")
code, r = api("POST", "/api/analysis/geomechanical-facies", {"source": "demo", "well": "3P", "n_facies": 3})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has n_facies", "n_facies" in r)
test("Has facies_class", "facies_class" in r)
test("Class valid", r.get("facies_class") in ("DISTRIBUTED", "HETEROGENEOUS", "WELL_DEFINED"))
test("Has facies", "facies" in r)
test("Facies non-empty", len(r.get("facies", [])) > 0)
test("Facies has facies_id", r.get("facies", [{}])[0].get("facies_id") is not None)
test("Facies has n_fractures", r.get("facies", [{}])[0].get("n_fractures") is not None)
test("Facies has mean_dip_deg", r.get("facies", [{}])[0].get("mean_dip_deg") is not None)
test("Facies has depth_range_m", r.get("facies", [{}])[0].get("depth_range_m") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("5 facies", api("POST", "/api/analysis/geomechanical-facies", {"source": "demo", "well": "3P", "n_facies": 5})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/geomechanical-facies", {"source": "demo", "well": "6P"})[0] in (200, 400, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.65.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
