"""Tests for v3.55.0: Fracture Intersection, Stress Polygon, Mud Weight Window, Fracture Spacing, Stress Ratio."""
import requests, sys

BASE = "http://localhost:8145"
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
# [210] Fracture Intersection Density
# ═══════════════════════════════════════════════════════════════
print("\n[210] Fracture Intersection Density")
code, r = api("POST", "/api/analysis/fracture-intersection", {"source": "demo", "well": "3P", "bin_size_m": 50})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has bin_size_m", "bin_size_m" in r)
test("Has depth_range_m", "depth_range_m" in r)
test("Has total_intersections", "total_intersections" in r)
test("Has overall_density_per_m", "overall_density_per_m" in r)
test("Has connectivity_class", "connectivity_class" in r)
test("Class valid", r.get("connectivity_class") in ("HIGH", "MODERATE", "LOW"))
test("Has n_bins", "n_bins" in r)
test("Has max_intersection_bin", "max_intersection_bin" in r)
test("Has bins", "bins" in r)
test("Bins non-empty", len(r.get("bins", [])) > 0)
test("Bin has depth_from_m", r.get("bins", [{}])[0].get("depth_from_m") is not None)
test("Bin has n_fractures", r.get("bins", [{}])[0].get("n_fractures") is not None)
test("Bin has n_intersections", r.get("bins", [{}])[0].get("n_intersections") is not None)
test("Bin has intersection_density_per_m", r.get("bins", [{}])[0].get("intersection_density_per_m") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/fracture-intersection", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [211] Stress Polygon
# ═══════════════════════════════════════════════════════════════
print("\n[211] Stress Polygon")
code, r = api("POST", "/api/analysis/stress-polygon-zoback", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has friction", "friction" in r)
test("Has Sv_MPa", "Sv_MPa" in r)
test("Has Pp_MPa", "Pp_MPa" in r)
test("Has SHmax_est_MPa", "SHmax_est_MPa" in r)
test("Has Shmin_est_MPa", "Shmin_est_MPa" in r)
test("Has current_regime", "current_regime" in r)
test("Regime valid", r.get("current_regime") in ("NORMAL_FAULT", "STRIKE_SLIP", "REVERSE_FAULT"))
test("Has frictional_limit_q", "frictional_limit_q" in r)
test("Has K_Hmin", "K_Hmin" in r)
test("Has K_Hmax", "K_Hmax" in r)
test("Has nf_shmin_min_MPa", "nf_shmin_min_MPa" in r)
test("Has rf_shmax_max_MPa", "rf_shmax_max_MPa" in r)
test("Has polygon_points", "polygon_points" in r)
test("Points non-empty", len(r.get("polygon_points", [])) > 0)
test("Point has Shmin_MPa", r.get("polygon_points", [{}])[0].get("Shmin_MPa") is not None)
test("Point has SHmax_MPa", r.get("polygon_points", [{}])[0].get("SHmax_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/stress-polygon-zoback", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [212] Mud Weight Window
# ═══════════════════════════════════════════════════════════════
print("\n[212] Mud Weight Window")
code, r = api("POST", "/api/analysis/mud-weight-profile", {"source": "demo", "well": "3P", "depth_from": 1000, "depth_to": 5000, "n_points": 30})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has min_window_ppg", "min_window_ppg" in r)
test("Has window_class", "window_class" in r)
test("Class valid", r.get("window_class") in ("WIDE", "ADEQUATE", "NARROW", "INVERTED"))
test("Has pct_safe", "pct_safe" in r)
test("Has narrowest_point", "narrowest_point" in r)
test("Narrowest has depth_m", r.get("narrowest_point", {}).get("depth_m") is not None)
test("Narrowest has window_ppg", r.get("narrowest_point", {}).get("window_ppg") is not None)
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Pp_ppg", r.get("profile", [{}])[0].get("Pp_ppg") is not None)
test("Prof has collapse_ppg", r.get("profile", [{}])[0].get("collapse_ppg") is not None)
test("Prof has frac_ppg", r.get("profile", [{}])[0].get("frac_ppg") is not None)
test("Prof has overburden_ppg", r.get("profile", [{}])[0].get("overburden_ppg") is not None)
test("Prof has window_ppg", r.get("profile", [{}])[0].get("window_ppg") is not None)
test("Prof has safe", "safe" in r.get("profile", [{}])[0])
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/mud-weight-profile", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [213] Fracture Spacing Statistics
# ═══════════════════════════════════════════════════════════════
print("\n[213] Fracture Spacing Statistics")
code, r = api("POST", "/api/analysis/fracture-spacing-stats", {"source": "demo", "well": "3P"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has n_spacings", "n_spacings" in r)
test("Has mean_spacing_m", "mean_spacing_m" in r)
test("Has median_spacing_m", "median_spacing_m" in r)
test("Has std_spacing_m", "std_spacing_m" in r)
test("Has min_spacing_m", "min_spacing_m" in r)
test("Has max_spacing_m", "max_spacing_m" in r)
test("Has cv", "cv" in r)
test("Has P10_m", "P10_m" in r)
test("Has P50_m", "P50_m" in r)
test("Has P90_m", "P90_m" in r)
test("Has clustering_class", "clustering_class" in r)
test("Class valid", r.get("clustering_class") in ("CLUSTERED", "RANDOM", "REGULAR"))
test("Has best_fit_distribution", "best_fit_distribution" in r)
test("Fit valid", r.get("best_fit_distribution") in ("exponential", "lognormal"))
test("Has exponential_fit", "exponential_fit" in r)
test("Exp has p_value", "p_value" in r.get("exponential_fit", {}))
test("Has lognormal_fit", "lognormal_fit" in r)
test("LN has p_value", "p_value" in r.get("lognormal_fit", {}))
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/fracture-spacing-stats", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [214] In-Situ Stress Ratio Profile
# ═══════════════════════════════════════════════════════════════
print("\n[214] In-Situ Stress Ratio Profile")
code, r = api("POST", "/api/analysis/stress-ratio-profile", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 30})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has mean_K0", "mean_K0" in r)
test("Has mean_A_value", "mean_A_value" in r)
test("Has K0_class", "K0_class" in r)
test("Class valid", r.get("K0_class") in ("HIGH", "MODERATE", "LOW"))
test("Has dominant_regime", "dominant_regime" in r)
test("Regime valid", r.get("dominant_regime") in ("NF", "SS", "RF"))
test("Has regime_counts", "regime_counts" in r)
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Sv_MPa", r.get("profile", [{}])[0].get("Sv_MPa") is not None)
test("Prof has SHmax_MPa", r.get("profile", [{}])[0].get("SHmax_MPa") is not None)
test("Prof has Shmin_MPa", r.get("profile", [{}])[0].get("Shmin_MPa") is not None)
test("Prof has K0", r.get("profile", [{}])[0].get("K0") is not None)
test("Prof has A_value", r.get("profile", [{}])[0].get("A_value") is not None)
test("Prof has R_ratio", r.get("profile", [{}])[0].get("R_ratio") is not None)
test("Prof has regime", r.get("profile", [{}])[0].get("regime") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/stress-ratio-profile", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.55.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
