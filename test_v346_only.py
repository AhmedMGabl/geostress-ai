"""Focused test for v3.46.0 endpoints: Stereonet Density + Stress Path + Aperture Distribution + Stability Window + Orientation Stats."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8133"


def api(method, path, body=None, timeout=300):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {label}" + (f" ({detail})" if detail else ""))
        passed += 1
    else:
        print(f"  FAIL: {label}" + (f" ({detail})" if detail else ""))
        failed += 1


# ── [165] Stereonet Density ──────────────────────────────
print("\n[165] Stereonet Density")
sd = api("POST", "/api/analysis/stereonet-density", {"source": "demo", "well": "3P", "grid_resolution": 100, "sigma": 3.0})
check("Status 200", sd is not None)
check("Has well", sd.get("well") == "3P")
check("Has n_poles", isinstance(sd.get("n_poles"), int) and sd["n_poles"] > 0)
check("Has grid_resolution", isinstance(sd.get("grid_resolution"), int))
check("Has sigma", isinstance(sd.get("sigma"), (int, float)))
check("Has counting_angle_deg", isinstance(sd.get("counting_angle_deg"), (int, float)))
check("Has max_density", isinstance(sd.get("max_density"), (int, float)))
check("Has mean_density", isinstance(sd.get("mean_density"), (int, float)))
check("Has peak_pole_trend", isinstance(sd.get("peak_pole_trend"), (int, float)))
check("Has peak_pole_plunge", isinstance(sd.get("peak_pole_plunge"), (int, float)))
check("Has fisher_kappa", isinstance(sd.get("fisher_kappa"), (int, float)))
check("Has R_bar", isinstance(sd.get("R_bar"), (int, float)))
check("Has clustering", sd.get("clustering") in ("STRONG", "MODERATE", "WEAK"))
check("Has contour_levels", isinstance(sd.get("contour_levels"), dict))
cl = sd["contour_levels"]
check("Contour has p50", isinstance(cl.get("p50"), (int, float)))
check("Contour has p95", isinstance(cl.get("p95"), (int, float)))
check("Has recommendations", isinstance(sd.get("recommendations"), list) and len(sd["recommendations"]) > 0)
check("Has plot", isinstance(sd.get("plot"), str) and len(sd["plot"]) > 100)
check("Has stakeholder_brief", isinstance(sd.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(sd["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(sd.get("elapsed_s"), (int, float)))

# 6P test
sd2 = api("POST", "/api/analysis/stereonet-density", {"source": "demo", "well": "6P"})
check("6P works", sd2 is not None and sd2.get("well") == "6P")


# ── [166] Stress Path ────────────────────────────────────
print("\n[166] Stress Path")
sp = api("POST", "/api/analysis/stress-path", {"source": "demo", "well": "3P", "depth_start": 1000, "depth_end": 5000, "n_points": 20, "friction": 0.6, "pp_gradient": 10.0, "sv_gradient": 25.0})
check("Status 200", sp is not None)
check("Has well", sp.get("well") == "3P")
check("Has depth_range_m", isinstance(sp.get("depth_range_m"), list) and len(sp["depth_range_m"]) == 2)
check("Has n_points", isinstance(sp.get("n_points"), int) and sp["n_points"] >= 2)
check("Has friction", isinstance(sp.get("friction"), (int, float)))
check("Has frictional_limit_q", isinstance(sp.get("frictional_limit_q"), (int, float)) and sp["frictional_limit_q"] > 1)
check("Has pp_gradient_MPa_per_km", isinstance(sp.get("pp_gradient_MPa_per_km"), (int, float)))
check("Has sv_gradient_MPa_per_km", isinstance(sp.get("sv_gradient_MPa_per_km"), (int, float)))
check("Has path_points", isinstance(sp.get("path_points"), list) and len(sp["path_points"]) >= 2)
pp0 = sp["path_points"][0]
check("Point has depth_m", isinstance(pp0.get("depth_m"), (int, float)))
check("Point has Sv_MPa", isinstance(pp0.get("Sv_MPa"), (int, float)))
check("Point has Pp_MPa", isinstance(pp0.get("Pp_MPa"), (int, float)))
check("Point has Sv_eff_MPa", isinstance(pp0.get("Sv_eff_MPa"), (int, float)))
check("Point has normal_fault", isinstance(pp0.get("normal_fault"), dict))
check("NF has SHmax_MPa", isinstance(pp0["normal_fault"].get("SHmax_MPa"), (int, float)))
check("NF has Shmin_MPa", isinstance(pp0["normal_fault"].get("Shmin_MPa"), (int, float)))
check("Point has strike_slip", isinstance(pp0.get("strike_slip"), dict))
check("Point has reverse_fault", isinstance(pp0.get("reverse_fault"), dict))
check("Has mid_depth_summary", isinstance(sp.get("mid_depth_summary"), dict))
ms = sp["mid_depth_summary"]
check("Mid has depth_m", isinstance(ms.get("depth_m"), (int, float)))
check("Mid has stress_ratio_NF", isinstance(ms.get("stress_ratio_NF"), (int, float)))
check("Has recommendations", isinstance(sp.get("recommendations"), list) and len(sp["recommendations"]) > 0)
check("Has plot", isinstance(sp.get("plot"), str) and len(sp["plot"]) > 100)
check("Has stakeholder_brief", isinstance(sp.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(sp["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(sp.get("elapsed_s"), (int, float)))

# 6P test
sp2 = api("POST", "/api/analysis/stress-path", {"source": "demo", "well": "6P"})
check("6P works", sp2 is not None and sp2.get("well") == "6P")


# ── [167] Aperture Distribution ──────────────────────────
print("\n[167] Aperture Distribution")
ad = api("POST", "/api/analysis/aperture-distribution", {"source": "demo", "well": "3P", "model": "power_law", "reference_aperture_mm": 0.5})
check("Status 200", ad is not None)
check("Has well", ad.get("well") == "3P")
check("Has model", isinstance(ad.get("model"), str))
check("Has n_fractures", isinstance(ad.get("n_fractures"), int) and ad["n_fractures"] > 0)
check("Has n_spacings", isinstance(ad.get("n_spacings"), int) and ad["n_spacings"] > 0)
check("Has statistics", isinstance(ad.get("statistics"), dict))
stats = ad["statistics"]
check("Stats has mean_mm", isinstance(stats.get("mean_mm"), (int, float)))
check("Stats has median_mm", isinstance(stats.get("median_mm"), (int, float)))
check("Stats has std_mm", isinstance(stats.get("std_mm"), (int, float)))
check("Stats has p10_mm", isinstance(stats.get("p10_mm"), (int, float)))
check("Stats has p90_mm", isinstance(stats.get("p90_mm"), (int, float)))
check("Has lognormal_mu", isinstance(ad.get("lognormal_mu"), (int, float)))
check("Has lognormal_sigma", isinstance(ad.get("lognormal_sigma"), (int, float)))
check("Has permeability_darcy", isinstance(ad.get("permeability_darcy"), (int, float)))
check("Has size_distribution", isinstance(ad.get("size_distribution"), list) and len(ad["size_distribution"]) >= 3)
sd0 = ad["size_distribution"][0]
check("SizeDist has range", isinstance(sd0.get("range"), str))
check("SizeDist has count", isinstance(sd0.get("count"), int))
check("SizeDist has pct", isinstance(sd0.get("pct"), (int, float)))
check("Has recommendations", isinstance(ad.get("recommendations"), list) and len(ad["recommendations"]) > 0)
check("Has plot", isinstance(ad.get("plot"), str) and len(ad["plot"]) > 100)
check("Has stakeholder_brief", isinstance(ad.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(ad["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(ad.get("elapsed_s"), (int, float)))

# 6P test - may fail due to null depths
try:
    ad2 = api("POST", "/api/analysis/aperture-distribution", {"source": "demo", "well": "6P"})
    check("6P works or graceful error", ad2 is not None)
except Exception as e:
    check("6P works or graceful error", "400" in str(e) or "404" in str(e), f"Expected: {e}")


# ── [168] Stability Window ───────────────────────────────
print("\n[168] Stability Window")
sw = api("POST", "/api/analysis/stability-window", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6, "ucs_mpa": 80, "tensile_strength_mpa": 8, "pp_gradient": 10.0, "sv_gradient": 25.0})
check("Status 200", sw is not None)
check("Has well", sw.get("well") == "3P")
check("Has depth_m", isinstance(sw.get("depth_m"), (int, float)))
check("Has friction", isinstance(sw.get("friction"), (int, float)))
check("Has UCS_MPa", isinstance(sw.get("UCS_MPa"), (int, float)))
check("Has tensile_strength_MPa", isinstance(sw.get("tensile_strength_MPa"), (int, float)))
check("Has Sv_MPa", isinstance(sw.get("Sv_MPa"), (int, float)))
check("Has SHmax_MPa", isinstance(sw.get("SHmax_MPa"), (int, float)))
check("Has Shmin_MPa", isinstance(sw.get("Shmin_MPa"), (int, float)))
check("Has Pp_MPa", isinstance(sw.get("Pp_MPa"), (int, float)))
check("Has collapse_pressure_MPa", isinstance(sw.get("collapse_pressure_MPa"), (int, float)))
check("Has fracture_pressure_MPa", isinstance(sw.get("fracture_pressure_MPa"), (int, float)))
check("Has loss_circulation_MPa", isinstance(sw.get("loss_circulation_MPa"), (int, float)))
check("Has mud_weight_collapse_SG", isinstance(sw.get("mud_weight_collapse_SG"), (int, float)))
check("Has mud_weight_kick_SG", isinstance(sw.get("mud_weight_kick_SG"), (int, float)))
check("Has mud_weight_fracture_SG", isinstance(sw.get("mud_weight_fracture_SG"), (int, float)))
check("Has mud_weight_loss_SG", isinstance(sw.get("mud_weight_loss_SG"), (int, float)))
check("Has safe_window_lower_SG", isinstance(sw.get("safe_window_lower_SG"), (int, float)))
check("Has safe_window_upper_SG", isinstance(sw.get("safe_window_upper_SG"), (int, float)))
check("Has window_width_SG", isinstance(sw.get("window_width_SG"), (int, float)))
check("Has window_status", sw.get("window_status") in ("SAFE", "NARROW", "NO_WINDOW"))
check("Has recommendations", isinstance(sw.get("recommendations"), list) and len(sw["recommendations"]) > 0)
check("Has plot", isinstance(sw.get("plot"), str) and len(sw["plot"]) > 100)
check("Has stakeholder_brief", isinstance(sw.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(sw["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(sw.get("elapsed_s"), (int, float)))

# 6P test
sw2 = api("POST", "/api/analysis/stability-window", {"source": "demo", "well": "6P", "depth": 2500})
check("6P works", sw2 is not None and sw2.get("well") == "6P")


# ── [169] Orientation Statistics ─────────────────────────
print("\n[169] Orientation Statistics")
os_ = api("POST", "/api/analysis/orientation-stats", {"source": "demo", "well": "3P"})
check("Status 200", os_ is not None)
check("Has well", os_.get("well") == "3P")
check("Has n_azimuths", isinstance(os_.get("n_azimuths"), int) and os_["n_azimuths"] > 0)
check("Has n_dips", isinstance(os_.get("n_dips"), int) and os_["n_dips"] > 0)
check("Has azimuth_stats", isinstance(os_.get("azimuth_stats"), dict))
az = os_["azimuth_stats"]
check("Az has mean_deg", isinstance(az.get("mean_deg"), (int, float)))
check("Az has circular_variance", isinstance(az.get("circular_variance"), (int, float)))
check("Az has circular_std_deg", isinstance(az.get("circular_std_deg"), (int, float)))
check("Az has resultant_length", isinstance(az.get("resultant_length"), (int, float)))
check("Az has von_mises_kappa", isinstance(az.get("von_mises_kappa"), (int, float)))
check("Az has quartiles", isinstance(az.get("quartiles"), dict))
check("Has dip_stats", isinstance(os_.get("dip_stats"), dict))
ds = os_["dip_stats"]
check("Dip has mean_deg", isinstance(ds.get("mean_deg"), (int, float)))
check("Dip has std_deg", isinstance(ds.get("std_deg"), (int, float)))
check("Dip has median_deg", isinstance(ds.get("median_deg"), (int, float)))
check("Has rayleigh_test", isinstance(os_.get("rayleigh_test"), dict))
rt = os_["rayleigh_test"]
check("Rayleigh has z_statistic", isinstance(rt.get("z_statistic"), (int, float)))
check("Rayleigh has p_value", isinstance(rt.get("p_value"), (int, float)))
check("Rayleigh has is_uniform", isinstance(rt.get("is_uniform"), bool))
check("Rayleigh has interpretation", isinstance(rt.get("interpretation"), str))
check("Has kuiper_V", isinstance(os_.get("kuiper_V"), (int, float)))
check("Has orientation_tensor", isinstance(os_.get("orientation_tensor"), dict))
ot = os_["orientation_tensor"]
check("Tensor has eigenvalues", isinstance(ot.get("eigenvalues"), list) and len(ot["eigenvalues"]) == 3)
check("Tensor has fabric_type", ot.get("fabric_type") in ("CLUSTER", "GIRDLE", "RANDOM"))
check("Tensor has fabric_strength", ot.get("fabric_strength") in ("STRONG", "MODERATE", "WEAK"))
check("Tensor has woodcock_C", isinstance(ot.get("woodcock_C"), (int, float)))
check("Tensor has woodcock_K", isinstance(ot.get("woodcock_K"), (int, float)))
check("Has recommendations", isinstance(os_.get("recommendations"), list) and len(os_["recommendations"]) > 0)
check("Has plot", isinstance(os_.get("plot"), str) and len(os_["plot"]) > 100)
check("Has stakeholder_brief", isinstance(os_.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(os_["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(os_.get("elapsed_s"), (int, float)))

# 6P test
os2 = api("POST", "/api/analysis/orientation-stats", {"source": "demo", "well": "6P"})
check("6P works", os2 is not None and os2.get("well") == "6P")


# ── Summary ──────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.46.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
