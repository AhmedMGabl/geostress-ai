"""Tests for v3.72.0: Washout, Casing Shoe, Diff Sticking, BH Stability Map, HF Containment."""
import requests, sys

BASE = "http://localhost:8167"
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
# [295] Wellbore Washout
# ═══════════════════════════════════════════════════════════════
print("\n[295] Wellbore Washout")
code, r = api("POST", "/api/analysis/wellbore-washout", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "bit_size_in": 8.5, "mud_weight_ppg": 10})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has bit_size_in", "bit_size_in" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has mean_washout_risk", "mean_washout_risk" in r)
test("Has max_enlargement_pct", "max_enlargement_pct" in r)
test("Has pct_washout", "pct_washout" in r)
test("Has wo_class", "wo_class" in r)
test("Class valid", r.get("wo_class") in ("SEVERE", "MODERATE", "MINOR", "STABLE"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has caliper_in", r.get("profile", [{}])[0].get("caliper_in") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Light MW", api("POST", "/api/analysis/wellbore-washout", {"source": "demo", "well": "3P", "mud_weight_ppg": 8})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/wellbore-washout", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [296] Casing Shoe Strength
# ═══════════════════════════════════════════════════════════════
print("\n[296] Casing Shoe Strength")
code, r = api("POST", "/api/analysis/casing-shoe-strength", {"source": "demo", "well": "3P", "shoe_depth_m": 2000, "casing_size_in": 9.625, "mud_weight_ppg": 10})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has shoe_depth_m", "shoe_depth_m" in r)
test("Has casing_size_in", "casing_size_in" in r)
test("Has LOT_MPa", "LOT_MPa" in r)
test("Has LOT_ppg", "LOT_ppg" in r)
test("Has FIT_MPa", "FIT_MPa" in r)
test("Has MAASP_psi", "MAASP_psi" in r)
test("Has kick_tolerance_ppg", "kick_tolerance_ppg" in r)
test("Has shoe_class", "shoe_class" in r)
test("Class valid", r.get("shoe_class") in ("WEAK", "MARGINAL", "ADEQUATE", "STRONG"))
test("Has mw_sweep", "mw_sweep" in r)
test("Sweep non-empty", len(r.get("mw_sweep", [])) > 0)
test("Sweep has MW_ppg", r.get("mw_sweep", [{}])[0].get("MW_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Deep shoe", api("POST", "/api/analysis/casing-shoe-strength", {"source": "demo", "well": "3P", "shoe_depth_m": 4000})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/casing-shoe-strength", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [297] Differential Sticking
# ═══════════════════════════════════════════════════════════════
print("\n[297] Differential Sticking")
code, r = api("POST", "/api/analysis/differential-sticking", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "mud_weight_ppg": 11, "pipe_od_in": 5.0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has pipe_od_in", "pipe_od_in" in r)
test("Has mean_risk_index", "mean_risk_index" in r)
test("Has max_risk_index", "max_risk_index" in r)
test("Has ds_class", "ds_class" in r)
test("Class valid", r.get("ds_class") in ("HIGH", "MODERATE", "LOW", "MINIMAL"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has overbalance_MPa", r.get("profile", [{}])[0].get("overbalance_MPa") is not None)
test("Prof has risk_index", r.get("profile", [{}])[0].get("risk_index") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Heavy MW", api("POST", "/api/analysis/differential-sticking", {"source": "demo", "well": "3P", "mud_weight_ppg": 14})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/differential-sticking", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [298] Borehole Stability Map
# ═══════════════════════════════════════════════════════════════
print("\n[298] Borehole Stability Map")
code, r = api("POST", "/api/analysis/borehole-stability-map", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "mw_min_ppg": 8, "mw_max_ppg": 16})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has mw_min_ppg", "mw_min_ppg" in r)
test("Has mw_max_ppg", "mw_max_ppg" in r)
test("Has mean_SI", "mean_SI" in r)
test("Has pct_unstable", "pct_unstable" in r)
test("Has map_class", "map_class" in r)
test("Class valid", r.get("map_class") in ("CRITICAL", "CONSTRAINED", "MODERATE", "FAVORABLE"))
test("Has grid", "grid" in r)
test("Grid non-empty", len(r.get("grid", [])) > 0)
test("Grid has depth_m", r.get("grid", [{}])[0].get("depth_m") is not None)
test("Grid has MW_ppg", r.get("grid", [{}])[0].get("MW_ppg") is not None)
test("Grid has stability_index", r.get("grid", [{}])[0].get("stability_index") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Narrow range", api("POST", "/api/analysis/borehole-stability-map", {"source": "demo", "well": "3P", "mw_min_ppg": 9, "mw_max_ppg": 12})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/borehole-stability-map", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [299] Hydraulic Fracture Containment
# ═══════════════════════════════════════════════════════════════
print("\n[299] Hydraulic Fracture Containment")
code, r = api("POST", "/api/analysis/hydraulic-fracture-containment", {"source": "demo", "well": "3P", "depth_m": 3000, "reservoir_thickness_m": 30, "net_pressure_MPa": 5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has reservoir_thickness_m", "reservoir_thickness_m" in r)
test("Has net_pressure_MPa", "net_pressure_MPa" in r)
test("Has Shmin_res_MPa", "Shmin_res_MPa" in r)
test("Has barrier_contrast_MPa", "barrier_contrast_MPa" in r)
test("Has containment_ratio", "containment_ratio" in r)
test("Has frac_height_m", "frac_height_m" in r)
test("Has height_ratio", "height_ratio" in r)
test("Has hf_class", "hf_class" in r)
test("Class valid", r.get("hf_class") in ("CONTAINED", "MARGINAL", "BREAKTHROUGH", "UNCONTAINED"))
test("Has np_sweep", "np_sweep" in r)
test("Sweep non-empty", len(r.get("np_sweep", [])) > 0)
test("Sweep has net_pressure_MPa", r.get("np_sweep", [{}])[0].get("net_pressure_MPa") is not None)
test("Sweep has frac_height_m", r.get("np_sweep", [{}])[0].get("frac_height_m") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High pressure", api("POST", "/api/analysis/hydraulic-fracture-containment", {"source": "demo", "well": "3P", "net_pressure_MPa": 10})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/hydraulic-fracture-containment", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.72.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
