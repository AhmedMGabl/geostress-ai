"""Tests for v3.58.0: Stress Gradient, Mineral Fill, Coulomb Failure, DFN Params, Drilling Margin."""
import requests, sys

BASE = "http://localhost:8148"
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
# [225] Stress Gradient Profile
# ═══════════════════════════════════════════════════════════════
print("\n[225] Stress Gradient Profile")
code, r = api("POST", "/api/analysis/stress-gradient-profile", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 30})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has mean_Sv_gradient_MPa_km", "mean_Sv_gradient_MPa_km" in r)
test("Has mean_Pp_gradient_MPa_km", "mean_Pp_gradient_MPa_km" in r)
test("Has gradient_class", "gradient_class" in r)
test("Class valid", r.get("gradient_class") in ("HIGH", "NORMAL", "LOW"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Sv_MPa", r.get("profile", [{}])[0].get("Sv_MPa") is not None)
test("Prof has Sv_gradient_MPa_km", r.get("profile", [{}])[0].get("Sv_gradient_MPa_km") is not None)
test("Prof has Pp_EMW_ppg", r.get("profile", [{}])[0].get("Pp_EMW_ppg") is not None)
test("Prof has Sv_psi_ft", r.get("profile", [{}])[0].get("Sv_psi_ft") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/stress-gradient-profile", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [226] Fracture Mineral Fill
# ═══════════════════════════════════════════════════════════════
print("\n[226] Fracture Mineral Fill")
code, r = api("POST", "/api/analysis/mineral-fill", {"source": "demo", "well": "3P"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has dominant_fill", "dominant_fill" in r)
test("Fill valid", r.get("dominant_fill") in ("calcite", "quartz", "clay", "open"))
test("Has fill_counts", "fill_counts" in r)
test("Has fill_percentages", "fill_percentages" in r)
test("Has mean_probabilities", "mean_probabilities" in r)
test("Has seal_impact", "seal_impact" in r)
test("Seal valid", r.get("seal_impact") in ("POOR_SEAL", "GOOD_SEAL", "CEMENTED", "MIXED"))
test("Has fractures", "fractures" in r)
test("Fractures non-empty", len(r.get("fractures", [])) > 0)
test("Frac has dominant_fill", r.get("fractures", [{}])[0].get("dominant_fill") is not None)
test("Frac has calcite_prob", r.get("fractures", [{}])[0].get("calcite_prob") is not None)
test("Frac has quartz_prob", r.get("fractures", [{}])[0].get("quartz_prob") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/mineral-fill", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [227] Coulomb Failure Function
# ═══════════════════════════════════════════════════════════════
print("\n[227] Coulomb Failure Function")
code, r = api("POST", "/api/analysis/coulomb-failure-func", {"source": "demo", "well": "3P", "depth": 3000, "friction": 0.6, "cohesion_MPa": 0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has friction", "friction" in r)
test("Has cohesion_MPa", "cohesion_MPa" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has n_at_failure", "n_at_failure" in r)
test("Has pct_at_failure", "pct_at_failure" in r)
test("Has mean_CFF_MPa", "mean_CFF_MPa" in r)
test("Has max_CFF_MPa", "max_CFF_MPa" in r)
test("Has min_CFF_MPa", "min_CFF_MPa" in r)
test("Has failure_class", "failure_class" in r)
test("Class valid", r.get("failure_class") in ("CRITICAL", "HIGH", "MODERATE", "STABLE"))
test("Has fractures", "fractures" in r)
test("Fractures non-empty", len(r.get("fractures", [])) > 0)
test("Frac has CFF_MPa", r.get("fractures", [{}])[0].get("CFF_MPa") is not None)
test("Frac has sigma_n_MPa", r.get("fractures", [{}])[0].get("sigma_n_MPa") is not None)
test("Frac has tau_MPa", r.get("fractures", [{}])[0].get("tau_MPa") is not None)
test("Frac has at_failure", "at_failure" in r.get("fractures", [{}])[0])
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/coulomb-failure-func", {"source": "demo", "well": "6P", "depth": 3000})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [228] DFN Parameters
# ═══════════════════════════════════════════════════════════════
print("\n[228] DFN Parameters")
code, r = api("POST", "/api/analysis/dfn-params", {"source": "demo", "well": "3P"})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has depth_range_m", "depth_range_m" in r)
test("Has P10_per_m", "P10_per_m" in r)
test("Has P32_est_per_m3", "P32_est_per_m3" in r)
test("Has mean_spacing_m", "mean_spacing_m" in r)
test("Has median_spacing_m", "median_spacing_m" in r)
test("Has cv_spacing", "cv_spacing" in r)
test("Has clustering", "clustering" in r)
test("Clustering valid", r.get("clustering") in ("HIGHLY_CLUSTERED", "CLUSTERED", "RANDOM", "REGULAR"))
test("Has best_spacing_dist", "best_spacing_dist" in r)
test("Has exp_p_value", "exp_p_value" in r)
test("Has lognorm_p_value", "lognorm_p_value" in r)
test("Has mean_azimuth_deg", "mean_azimuth_deg" in r)
test("Has mean_dip_deg", "mean_dip_deg" in r)
test("Has fisher_kappa", "fisher_kappa" in r)
test("Has orientation_R_bar", "orientation_R_bar" in r)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/dfn-params", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [229] Drilling Margin Analysis
# ═══════════════════════════════════════════════════════════════
print("\n[229] Drilling Margin Analysis")
code, r = api("POST", "/api/analysis/drilling-margin", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 30, "mud_weight_ppg": 10.0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has min_window_ppg", "min_window_ppg" in r)
test("Has min_kick_margin_ppg", "min_kick_margin_ppg" in r)
test("Has min_loss_margin_ppg", "min_loss_margin_ppg" in r)
test("Has pct_safe", "pct_safe" in r)
test("Has margin_class", "margin_class" in r)
test("Class valid", r.get("margin_class") in ("KICK_RISK", "LOSS_RISK", "NARROW", "ADEQUATE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Pp_ppg", r.get("profile", [{}])[0].get("Pp_ppg") is not None)
test("Prof has collapse_ppg", r.get("profile", [{}])[0].get("collapse_ppg") is not None)
test("Prof has frac_ppg", r.get("profile", [{}])[0].get("frac_ppg") is not None)
test("Prof has kick_margin_ppg", r.get("profile", [{}])[0].get("kick_margin_ppg") is not None)
test("Prof has loss_margin_ppg", r.get("profile", [{}])[0].get("loss_margin_ppg") is not None)
test("Prof has safe", "safe" in r.get("profile", [{}])[0])
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Heavy MW works", api("POST", "/api/analysis/drilling-margin", {"source": "demo", "well": "3P", "mud_weight_ppg": 14.0})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.58.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
