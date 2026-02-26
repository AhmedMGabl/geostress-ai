"""Focused test for v3.43.0 endpoints: Fracture Spacing + Stress Polygon + Orientation Clustering + Depth Trends + Confidence Map."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8129"


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


# ── [150] Fracture Spacing Analysis ───────────────────
print("\n[150] Fracture Spacing Analysis")
fs = api("POST", "/api/analysis/fracture-spacing", {"source": "demo", "well": "3P"})
check("Status 200", fs is not None)
check("Has well", fs.get("well") == "3P")
check("Has n_fractures", isinstance(fs.get("n_fractures"), int) and fs["n_fractures"] > 0)
check("Has n_spacings", isinstance(fs.get("n_spacings"), int))
check("Has mean_spacing_m", isinstance(fs.get("mean_spacing_m"), (int, float)))
check("Has median_spacing_m", isinstance(fs.get("median_spacing_m"), (int, float)))
check("Has std_spacing_m", isinstance(fs.get("std_spacing_m"), (int, float)))
check("Has cv", isinstance(fs.get("cv"), (int, float)))
check("Has pattern", fs.get("pattern") in ("CLUSTERED", "REGULAR", "RANDOM"))
check("Has lognormal_mu", isinstance(fs.get("lognormal_mu"), (int, float)))
check("Has lognormal_sigma", isinstance(fs.get("lognormal_sigma"), (int, float)))
check("Has percentiles", isinstance(fs.get("percentiles"), dict) and "p50" in fs["percentiles"])
check("Has depth_zones", isinstance(fs.get("depth_zones"), list) and len(fs["depth_zones"]) >= 2)
z0 = fs["depth_zones"][0]
check("Zone has zone", isinstance(z0.get("zone"), str))
check("Zone has n_fractures", isinstance(z0.get("n_fractures"), int))
check("Zone has density_per_m", isinstance(z0.get("density_per_m"), (int, float)))
check("Has recommendations", isinstance(fs.get("recommendations"), list) and len(fs["recommendations"]) > 0)
check("Has plot", isinstance(fs.get("plot"), str) and len(fs["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fs.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(fs["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(fs.get("elapsed_s"), (int, float)))

# 6P test
fs2 = api("POST", "/api/analysis/fracture-spacing", {"source": "demo", "well": "6P"})
check("6P works", fs2 is not None and fs2.get("well") == "6P")


# ── [151] Stress Polygon ─────────────────────────────
print("\n[151] Stress Polygon")
sp = api("POST", "/api/analysis/stress-polygon-extended", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6})
check("Status 200", sp is not None)
check("Has well", sp.get("well") == "3P")
check("Has depth_m", isinstance(sp.get("depth_m"), (int, float)) and sp["depth_m"] == 3000)
check("Has friction", isinstance(sp.get("friction"), (int, float)) and sp["friction"] == 0.6)
check("Has Pp_MPa", isinstance(sp.get("Pp_MPa"), (int, float)))
check("Has Sv_MPa", isinstance(sp.get("Sv_MPa"), (int, float)) and sp["Sv_MPa"] > 0)
check("Has frictional_limit_q", isinstance(sp.get("frictional_limit_q"), (int, float)) and sp["frictional_limit_q"] > 1)
check("Has regimes", isinstance(sp.get("regimes"), list) and len(sp["regimes"]) == 3)
r0 = sp["regimes"][0]
check("Regime has regime", isinstance(r0.get("regime"), str))
check("Regime has Sv_MPa", isinstance(r0.get("Sv_MPa"), (int, float)))
check("Regime has SHmax_range", isinstance(r0.get("SHmax_range"), list) and len(r0["SHmax_range"]) == 2)
check("Regime has Shmin_range", isinstance(r0.get("Shmin_range"), list) and len(r0["Shmin_range"]) == 2)
check("Regime has description", isinstance(r0.get("description"), str))
check("Has recommendations", isinstance(sp.get("recommendations"), list) and len(sp["recommendations"]) > 0)
check("Has plot", isinstance(sp.get("plot"), str) and len(sp["plot"]) > 100)
check("Has stakeholder_brief", isinstance(sp.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(sp["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(sp.get("elapsed_s"), (int, float)))

# 6P test
sp2 = api("POST", "/api/analysis/stress-polygon-extended", {"source": "demo", "well": "6P", "depth": 2500, "friction": 0.7})
check("6P works", sp2 is not None and sp2.get("well") == "6P")


# ── [152] Orientation Clustering ──────────────────────
print("\n[152] Orientation Clustering")
oc = api("POST", "/api/analysis/orientation-clustering", {"source": "demo", "well": "3P", "max_k": 6})
check("Status 200", oc is not None)
check("Has well", oc.get("well") == "3P")
check("Has n_fractures", isinstance(oc.get("n_fractures"), int) and oc["n_fractures"] > 0)
check("Has best_k", isinstance(oc.get("best_k"), int) and oc["best_k"] >= 2)
check("Has best_silhouette", isinstance(oc.get("best_silhouette"), (int, float)))
check("Has quality", oc.get("quality") in ("EXCELLENT", "GOOD", "FAIR", "POOR"))
check("Has k_analysis", isinstance(oc.get("k_analysis"), list) and len(oc["k_analysis"]) >= 2)
ka0 = oc["k_analysis"][0]
check("k_analysis has k", isinstance(ka0.get("k"), int))
check("k_analysis has silhouette", isinstance(ka0.get("silhouette"), (int, float)))
check("k_analysis has inertia", isinstance(ka0.get("inertia"), (int, float)))
check("Has clusters", isinstance(oc.get("clusters"), list) and len(oc["clusters"]) >= 2)
cl0 = oc["clusters"][0]
check("Cluster has cluster", isinstance(cl0.get("cluster"), int))
check("Cluster has n_fractures", isinstance(cl0.get("n_fractures"), int))
check("Cluster has pct", isinstance(cl0.get("pct"), (int, float)))
check("Cluster has mean_azimuth", isinstance(cl0.get("mean_azimuth"), (int, float)))
check("Cluster has mean_dip", isinstance(cl0.get("mean_dip"), (int, float)))
check("Cluster has std_azimuth", isinstance(cl0.get("std_azimuth"), (int, float)))
check("Cluster has std_dip", isinstance(cl0.get("std_dip"), (int, float)))
check("Has recommendations", isinstance(oc.get("recommendations"), list) and len(oc["recommendations"]) > 0)
check("Has plot", isinstance(oc.get("plot"), str) and len(oc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(oc.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(oc["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(oc.get("elapsed_s"), (int, float)))

# 6P test
oc2 = api("POST", "/api/analysis/orientation-clustering", {"source": "demo", "well": "6P"})
check("6P works", oc2 is not None and oc2.get("well") == "6P")


# ── [153] Depth Trend Analysis ────────────────────────
print("\n[153] Depth Trend Analysis")
dt = api("POST", "/api/analysis/depth-trend", {"source": "demo", "well": "3P", "window": 20})
check("Status 200", dt is not None)
check("Has well", dt.get("well") == "3P")
check("Has n_samples", isinstance(dt.get("n_samples"), int) and dt["n_samples"] > 0)
check("Has depth_range", isinstance(dt.get("depth_range"), list) and len(dt["depth_range"]) == 2)
check("Has window_size", isinstance(dt.get("window_size"), int))
check("Has dip_trend", dt.get("dip_trend") in ("STEEPENING", "SHALLOWING", "STABLE"))
check("Has dip_slope_deg_per_m", isinstance(dt.get("dip_slope_deg_per_m"), (int, float)))
check("Has azimuth_trend", dt.get("azimuth_trend") in ("ROTATING", "STABLE"))
check("Has azimuth_rotation_rate_per_km", isinstance(dt.get("azimuth_rotation_rate_per_km"), (int, float)))
check("Has breakpoints", isinstance(dt.get("breakpoints"), list))
check("Has rolling_stats", isinstance(dt.get("rolling_stats"), list) and len(dt["rolling_stats"]) > 0)
rs0 = dt["rolling_stats"][0]
check("Rolling has depth_center", isinstance(rs0.get("depth_center"), (int, float)))
check("Rolling has mean_azimuth", isinstance(rs0.get("mean_azimuth"), (int, float)))
check("Rolling has mean_dip", isinstance(rs0.get("mean_dip"), (int, float)))
check("Rolling has std_azimuth", isinstance(rs0.get("std_azimuth"), (int, float)))
check("Rolling has n_fractures", isinstance(rs0.get("n_fractures"), int))
check("Has recommendations", isinstance(dt.get("recommendations"), list) and len(dt["recommendations"]) > 0)
check("Has plot", isinstance(dt.get("plot"), str) and len(dt["plot"]) > 100)
check("Has stakeholder_brief", isinstance(dt.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(dt["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(dt.get("elapsed_s"), (int, float)))

# 6P test -- 6P has null depths so this tests graceful handling
try:
    dt2 = api("POST", "/api/analysis/depth-trend", {"source": "demo", "well": "6P"})
    check("6P works or gracefully handles null depths", dt2 is not None)
except Exception as e:
    # 6P may fail if too few non-null depths -- that's acceptable
    check("6P works or gracefully handles null depths", "400" in str(e) or "404" in str(e), f"Expected error: {e}")


# ── [154] Classification Confidence Map ───────────────
print("\n[154] Classification Confidence Map")
cm = api("POST", "/api/analysis/confidence-map", {"source": "demo", "well": "3P"})
check("Status 200", cm is not None)
check("Has well", cm.get("well") == "3P")
check("Has n_samples", isinstance(cm.get("n_samples"), int) and cm["n_samples"] > 0)
check("Has n_classes", isinstance(cm.get("n_classes"), int) and cm["n_classes"] >= 2)
check("Has classes", isinstance(cm.get("classes"), list) and len(cm["classes"]) >= 2)
check("Has overall_accuracy", isinstance(cm.get("overall_accuracy"), (int, float)))
check("Has mean_confidence", isinstance(cm.get("mean_confidence"), (int, float)))
check("Has calibration_gap", isinstance(cm.get("calibration_gap"), (int, float)))
check("Has n_uncertain", isinstance(cm.get("n_uncertain"), int))
check("Has pct_uncertain", isinstance(cm.get("pct_uncertain"), (int, float)))
check("Has confidence_distribution", isinstance(cm.get("confidence_distribution"), list) and len(cm["confidence_distribution"]) == 5)
cd0 = cm["confidence_distribution"][0]
check("ConfDist has range", isinstance(cd0.get("range"), str))
check("ConfDist has label", isinstance(cd0.get("label"), str))
check("ConfDist has n_samples", isinstance(cd0.get("n_samples"), int))
check("ConfDist has pct", isinstance(cd0.get("pct"), (int, float)))
check("ConfDist has accuracy", isinstance(cd0.get("accuracy"), (int, float)))
check("Has per_class", isinstance(cm.get("per_class"), list) and len(cm["per_class"]) >= 2)
pc0 = cm["per_class"][0]
check("PerClass has class", isinstance(pc0.get("class"), str))
check("PerClass has n_samples", isinstance(pc0.get("n_samples"), int))
check("PerClass has mean_confidence", isinstance(pc0.get("mean_confidence"), (int, float)))
check("PerClass has accuracy", isinstance(pc0.get("accuracy"), (int, float)))
check("PerClass has n_low_confidence", isinstance(pc0.get("n_low_confidence"), int))
check("Has uncertain_samples", isinstance(cm.get("uncertain_samples"), list))
check("Has recommendations", isinstance(cm.get("recommendations"), list) and len(cm["recommendations"]) > 0)
check("Has plot", isinstance(cm.get("plot"), str) and len(cm["plot"]) > 100)
check("Has stakeholder_brief", isinstance(cm.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(cm["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(cm.get("elapsed_s"), (int, float)))

# 6P test
cm2 = api("POST", "/api/analysis/confidence-map", {"source": "demo", "well": "6P"})
check("6P works", cm2 is not None and cm2.get("well") == "6P")


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.43.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
