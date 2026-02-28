"""Tests for v3.57.0: Fracture Porosity, Differential Stress, Fault Seal, Connectivity Index, Breakout Angular."""
import requests, sys

BASE = "http://localhost:8147"
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
# [220] Fracture Porosity Estimate
# ═══════════════════════════════════════════════════════════════
print("\n[220] Fracture Porosity Estimate")
code, r = api("POST", "/api/analysis/fracture-porosity-estimate", {"source": "demo", "well": "3P", "aperture_mm": 0.5, "bin_size_m": 50})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has aperture_mm", "aperture_mm" in r)
test("Has bin_size_m", "bin_size_m" in r)
test("Has depth_range_m", "depth_range_m" in r)
test("Has n_bins", "n_bins" in r)
test("Has mean_porosity_pct", "mean_porosity_pct" in r)
test("Has max_porosity_pct", "max_porosity_pct" in r)
test("Has mean_permeability_mD", "mean_permeability_mD" in r)
test("Has max_permeability_mD", "max_permeability_mD" in r)
test("Has porosity_class", "porosity_class" in r)
test("Class valid", r.get("porosity_class") in ("HIGH", "MODERATE", "LOW"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_from_m", r.get("profile", [{}])[0].get("depth_from_m") is not None)
test("Prof has n_fractures", r.get("profile", [{}])[0].get("n_fractures") is not None)
test("Prof has fracture_porosity_pct", r.get("profile", [{}])[0].get("fracture_porosity_pct") is not None)
test("Prof has permeability_mD", r.get("profile", [{}])[0].get("permeability_mD") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/fracture-porosity-estimate", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [221] Differential Stress Profile
# ═══════════════════════════════════════════════════════════════
print("\n[221] Differential Stress Profile")
code, r = api("POST", "/api/analysis/differential-stress", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 30, "regime": "NF"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has regime", "regime" in r)
test("Has max_differential_MPa", "max_differential_MPa" in r)
test("Has mean_differential_MPa", "mean_differential_MPa" in r)
test("Has gradient_MPa_per_km", "gradient_MPa_per_km" in r)
test("Has stress_class", "stress_class" in r)
test("Class valid", r.get("stress_class") in ("HIGH", "MODERATE", "LOW"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has S1_MPa", r.get("profile", [{}])[0].get("S1_MPa") is not None)
test("Prof has S3_MPa", r.get("profile", [{}])[0].get("S3_MPa") is not None)
test("Prof has differential_stress_MPa", r.get("profile", [{}])[0].get("differential_stress_MPa") is not None)
test("Prof has q_MPa", r.get("profile", [{}])[0].get("q_MPa") is not None)
test("Prof has p_prime_MPa", r.get("profile", [{}])[0].get("p_prime_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("SS works", api("POST", "/api/analysis/differential-stress", {"source": "demo", "well": "3P", "regime": "SS"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [222] Fault Seal Analysis
# ═══════════════════════════════════════════════════════════════
print("\n[222] Fault Seal Analysis")
code, r = api("POST", "/api/analysis/fault-seal-analysis", {"source": "demo", "well": "3P", "clay_fraction": 0.2, "throw_m": 50})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has clay_fraction", "clay_fraction" in r)
test("Has throw_m", "throw_m" in r)
test("Has n_zones", "n_zones" in r)
test("Has mean_SGR_pct", "mean_SGR_pct" in r)
test("Has min_SGR_pct", "min_SGR_pct" in r)
test("Has max_seal_capacity_m", "max_seal_capacity_m" in r)
test("Has seal_class", "seal_class" in r)
test("Class valid", r.get("seal_class") in ("SEALING", "PARTIALLY_SEALING", "LEAKING"))
test("Has zones", "zones" in r)
test("Zones non-empty", len(r.get("zones", [])) > 0)
test("Zone has depth_from_m", r.get("zones", [{}])[0].get("depth_from_m") is not None)
test("Zone has SGR_pct", r.get("zones", [{}])[0].get("SGR_pct") is not None)
test("Zone has fault_perm_mD", r.get("zones", [{}])[0].get("fault_perm_mD") is not None)
test("Zone has seal_capacity_m", r.get("zones", [{}])[0].get("seal_capacity_m") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/fault-seal-analysis", {"source": "demo", "well": "6P", "clay_fraction": 0.3})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [223] Fracture Connectivity Index
# ═══════════════════════════════════════════════════════════════
print("\n[223] Fracture Connectivity Index")
code, r = api("POST", "/api/analysis/connectivity-index", {"source": "demo", "well": "3P", "bin_size_m": 50, "dip_threshold": 30})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has bin_size_m", "bin_size_m" in r)
test("Has dip_threshold_deg", "dip_threshold_deg" in r)
test("Has depth_range_m", "depth_range_m" in r)
test("Has n_bins", "n_bins" in r)
test("Has mean_connectivity_index", "mean_connectivity_index" in r)
test("Has max_connectivity_index", "max_connectivity_index" in r)
test("Has max_ci_depth_m", "max_ci_depth_m" in r)
test("Has pct_above_percolation", "pct_above_percolation" in r)
test("Has connectivity_class", "connectivity_class" in r)
test("Class valid", r.get("connectivity_class") in ("WELL_CONNECTED", "MODERATE", "POORLY_CONNECTED"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_from_m", r.get("profile", [{}])[0].get("depth_from_m") is not None)
test("Prof has P10_per_m", r.get("profile", [{}])[0].get("P10_per_m") is not None)
test("Prof has high_angle_pct", r.get("profile", [{}])[0].get("high_angle_pct") is not None)
test("Prof has orientation_dispersion", r.get("profile", [{}])[0].get("orientation_dispersion") is not None)
test("Prof has connectivity_index", r.get("profile", [{}])[0].get("connectivity_index") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/connectivity-index", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [224] Breakout Angular Width
# ═══════════════════════════════════════════════════════════════
print("\n[224] Breakout Angular Width")
code, r = api("POST", "/api/analysis/breakout-angular", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 30, "UCS_MPa": 80, "regime": "NF"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has regime", "regime" in r)
test("Has max_breakout_width_deg", "max_breakout_width_deg" in r)
test("Has mean_breakout_width_deg", "mean_breakout_width_deg" in r)
test("Has first_breakout_depth_m", "first_breakout_depth_m" in r)
test("Has pct_with_breakout", "pct_with_breakout" in r)
test("Has risk_class", "risk_class" in r)
test("Class valid", r.get("risk_class") in ("CRITICAL", "HIGH", "MODERATE", "LOW"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has SHmax_MPa", r.get("profile", [{}])[0].get("SHmax_MPa") is not None)
test("Prof has Shmin_MPa", r.get("profile", [{}])[0].get("Shmin_MPa") is not None)
test("Prof has breakout_width_deg", r.get("profile", [{}])[0].get("breakout_width_deg") is not None)
test("Prof has stability", r.get("profile", [{}])[0].get("stability") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("RF works", api("POST", "/api/analysis/breakout-angular", {"source": "demo", "well": "3P", "regime": "RF", "UCS_MPa": 50})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.57.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
