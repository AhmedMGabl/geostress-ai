"""Focused test for v3.36.0 endpoints: Uncertainty Zonation + Aperture-Permeability + Well Correlation."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8114"


def api(method, path, body=None, timeout=120):
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


# ── [117] Uncertainty-Aware Zonation ─────────────────────
print("\n[117] Uncertainty-Aware Zonation")
uz = api("POST", "/api/analysis/uncertainty-zonation", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", uz is not None)
check("Has well", uz.get("well") == "3P")
check("Has n_samples", isinstance(uz.get("n_samples"), int) and uz["n_samples"] > 0)
check("Has n_confident", isinstance(uz.get("n_confident"), int))
check("Has n_uncertain", isinstance(uz.get("n_uncertain"), int))
check("Has n_unreliable", isinstance(uz.get("n_unreliable"), int))
check("Counts add up", uz["n_confident"] + uz["n_uncertain"] + uz["n_unreliable"] == uz["n_samples"])
check("Has pct_confident", isinstance(uz.get("pct_confident"), (int, float)))
check("Has pct_uncertain", isinstance(uz.get("pct_uncertain"), (int, float)))
check("Has pct_unreliable", isinstance(uz.get("pct_unreliable"), (int, float)))
check("Has mean_certainty", isinstance(uz.get("mean_certainty"), (int, float)))
check("Has n_zones", isinstance(uz.get("n_zones"), int) and uz["n_zones"] >= 1)
check("Has continuous_zones", isinstance(uz.get("continuous_zones"), list) and len(uz["continuous_zones"]) >= 1)

cz0 = uz["continuous_zones"][0]
check("CZ has zone", cz0.get("zone") in ("CONFIDENT", "UNCERTAIN", "UNRELIABLE"))
check("CZ has top_m", isinstance(cz0.get("top_m"), (int, float)))
check("CZ has bottom_m", isinstance(cz0.get("bottom_m"), (int, float)))
check("CZ has thickness_m", isinstance(cz0.get("thickness_m"), (int, float)))
check("CZ has n_samples", isinstance(cz0.get("n_samples"), int))
check("CZ has mean_certainty", isinstance(cz0.get("mean_certainty"), (int, float)))

check("Has recommendations", isinstance(uz.get("recommendations"), list) and len(uz["recommendations"]) > 0)
check("Has plot", isinstance(uz.get("plot"), str) and len(uz["plot"]) > 100)
check("Has stakeholder_brief", isinstance(uz.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(uz["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", uz["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(uz["stakeholder_brief"].get("what_this_means"), str))

# 6P test
uz2 = api("POST", "/api/analysis/uncertainty-zonation", {"source": "demo", "well": "6P"}, timeout=300)
check("6P works", uz2 is not None and uz2.get("well") == "6P")


# ── [118] Fracture Aperture & Permeability ─────────────────
print("\n[118] Fracture Aperture & Permeability")
ap = api("POST", "/api/analysis/aperture-permeability", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", ap is not None)
check("Has well", ap.get("well") == "3P")
check("Has n_fractures", isinstance(ap.get("n_fractures"), int) and ap["n_fractures"] > 0)
check("Has mean_aperture_mm", isinstance(ap.get("mean_aperture_mm"), (int, float)) and ap["mean_aperture_mm"] > 0)
check("Has std_aperture_mm", isinstance(ap.get("std_aperture_mm"), (int, float)))
check("Has min_aperture_mm", isinstance(ap.get("min_aperture_mm"), (int, float)) and ap["min_aperture_mm"] > 0)
check("Has max_aperture_mm", isinstance(ap.get("max_aperture_mm"), (int, float)))
check("Has bulk_permeability_darcy", isinstance(ap.get("bulk_permeability_darcy"), (int, float)))

check("Has per_class_stats", isinstance(ap.get("per_class_stats"), list) and len(ap["per_class_stats"]) >= 2)
pcs0 = ap["per_class_stats"][0]
check("PCS has class", isinstance(pcs0.get("class"), str))
check("PCS has n_fractures", isinstance(pcs0.get("n_fractures"), int))
check("PCS has mean_aperture_mm", isinstance(pcs0.get("mean_aperture_mm"), (int, float)))
check("PCS has std_aperture_mm", isinstance(pcs0.get("std_aperture_mm"), (int, float)))
check("PCS has mean_permeability_darcy", isinstance(pcs0.get("mean_permeability_darcy"), (int, float)))
check("PCS has max_permeability_darcy", isinstance(pcs0.get("max_permeability_darcy"), (int, float)))
check("PCS has type_multiplier", isinstance(pcs0.get("type_multiplier"), (int, float)))

check("Has fracture_data", isinstance(ap.get("fracture_data"), list) and len(ap["fracture_data"]) > 0)
fd0 = ap["fracture_data"][0]
check("FD has index", isinstance(fd0.get("index"), int))
check("FD has depth_m", isinstance(fd0.get("depth_m"), (int, float)))
check("FD has dip_deg", isinstance(fd0.get("dip_deg"), (int, float)))
check("FD has fracture_type", isinstance(fd0.get("fracture_type"), str))
check("FD has aperture_mm", isinstance(fd0.get("aperture_mm"), (int, float)) and fd0["aperture_mm"] > 0)
check("FD has permeability_darcy", isinstance(fd0.get("permeability_darcy"), (int, float)))
check("FD has normal_stress_MPa", isinstance(fd0.get("normal_stress_MPa"), (int, float)))

check("Has recommendations", isinstance(ap.get("recommendations"), list) and len(ap["recommendations"]) > 0)
check("Has plot", isinstance(ap.get("plot"), str) and len(ap["plot"]) > 100)
check("Has stakeholder_brief", isinstance(ap.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(ap["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", ap["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# 6P test
ap2 = api("POST", "/api/analysis/aperture-permeability", {"source": "demo", "well": "6P"}, timeout=300)
check("6P works", ap2 is not None and ap2.get("well") == "6P")


# ── [119] Well Correlation Analysis ─────────────────────
print("\n[119] Well Correlation Analysis")
wc = api("POST", "/api/analysis/well-correlation", {"source": "demo"}, timeout=300)
check("Status 200", wc is not None)
check("Has n_wells", isinstance(wc.get("n_wells"), int) and wc["n_wells"] >= 2)

check("Has well_summaries", isinstance(wc.get("well_summaries"), list) and len(wc["well_summaries"]) >= 2)
ws0 = wc["well_summaries"][0]
check("WS has well", isinstance(ws0.get("well"), str))
check("WS has n_fractures", isinstance(ws0.get("n_fractures"), int))
check("WS has n_classes", isinstance(ws0.get("n_classes"), int))
check("WS has mean_azimuth", isinstance(ws0.get("mean_azimuth"), (int, float)))
check("WS has mean_dip", isinstance(ws0.get("mean_dip"), (int, float)))
check("WS has depth_range_m", isinstance(ws0.get("depth_range_m"), dict))

check("Has correlations", isinstance(wc.get("correlations"), list) and len(wc["correlations"]) >= 1)
c0 = wc["correlations"][0]
check("C has well_a", isinstance(c0.get("well_a"), str))
check("C has well_b", isinstance(c0.get("well_b"), str))
check("C has orientation_similarity", isinstance(c0.get("orientation_similarity"), (int, float)) and 0 <= c0["orientation_similarity"] <= 1)
check("C has dip_similarity", isinstance(c0.get("dip_similarity"), (int, float)) and 0 <= c0["dip_similarity"] <= 1)
check("C has type_overlap", isinstance(c0.get("type_overlap"), (int, float)))
check("C has distribution_similarity", isinstance(c0.get("distribution_similarity"), (int, float)))
check("C has depth_overlap", isinstance(c0.get("depth_overlap"), (int, float)))
check("C has overall_correlation", isinstance(c0.get("overall_correlation"), (int, float)) and 0 <= c0["overall_correlation"] <= 1)
check("C has correlation_level", c0.get("correlation_level") in ("HIGH", "MODERATE", "LOW"))
check("C has common_classes", isinstance(c0.get("common_classes"), list))

check("Has recommendations", isinstance(wc.get("recommendations"), list) and len(wc["recommendations"]) > 0)
check("Has plot", isinstance(wc.get("plot"), str) and len(wc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(wc.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(wc["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", wc["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has what_this_means", isinstance(wc["stakeholder_brief"].get("what_this_means"), str))


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.36.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
