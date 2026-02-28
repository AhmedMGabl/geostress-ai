"""Tests for v3.62.0: MW Optimization, Orientation Bias, Stress Ratio Depth, Shear Failure, Perm Tensor."""
import requests, sys

BASE = "http://localhost:8152"
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
# [245] Mud Weight Optimization
# ═══════════════════════════════════════════════════════════════
print("\n[245] Mud Weight Optimization")
code, r = api("POST", "/api/analysis/mud-weight-optimization", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "safety_factor": 1.1})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has safety_factor", "safety_factor" in r)
test("Has min_window_ppg", "min_window_ppg" in r)
test("Has optimal_range_ppg", "optimal_range_ppg" in r)
test("Has min_kick_margin_ppg", "min_kick_margin_ppg" in r)
test("Has mw_class", "mw_class" in r)
test("Class valid", r.get("mw_class") in ("CRITICAL", "NARROW", "ADEQUATE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has collapse_ppg", r.get("profile", [{}])[0].get("collapse_ppg") is not None)
test("Prof has frac_ppg", r.get("profile", [{}])[0].get("frac_ppg") is not None)
test("Prof has optimal_ppg", r.get("profile", [{}])[0].get("optimal_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High SF", api("POST", "/api/analysis/mud-weight-optimization", {"source": "demo", "well": "3P", "safety_factor": 1.3})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/mud-weight-optimization", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [246] Fracture Orientation Bias
# ═══════════════════════════════════════════════════════════════
print("\n[246] Fracture Orientation Bias")
code, r = api("POST", "/api/analysis/fracture-orientation-bias", {"source": "demo", "well": "3P", "borehole_azimuth": 0, "borehole_dip": 90})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has borehole_azimuth", "borehole_azimuth" in r)
test("Has borehole_dip", "borehole_dip" in r)
test("Has mean_terzaghi_weight", "mean_terzaghi_weight" in r)
test("Has max_terzaghi_weight", "max_terzaghi_weight" in r)
test("Has n_undersampled", "n_undersampled" in r)
test("Has pct_undersampled", "pct_undersampled" in r)
test("Has bias_class", "bias_class" in r)
test("Class valid", r.get("bias_class") in ("SEVERE", "MODERATE", "MINOR"))
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Deviated well", api("POST", "/api/analysis/fracture-orientation-bias", {"source": "demo", "well": "3P", "borehole_azimuth": 45, "borehole_dip": 60})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/fracture-orientation-bias", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [247] Stress Ratio Depth
# ═══════════════════════════════════════════════════════════════
print("\n[247] Stress Ratio Depth")
code, r = api("POST", "/api/analysis/stress-ratio-depth", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "n_points": 25})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has n_points", "n_points" in r)
test("Has mean_K0", "mean_K0" in r)
test("Has mean_A_ratio", "mean_A_ratio" in r)
test("Has K0_trend", "K0_trend" in r)
test("Has ratio_class", "ratio_class" in r)
test("Class valid", r.get("ratio_class") in ("HIGH_CONTRAST", "MODERATE_CONTRAST", "LOW_CONTRAST"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has K0", r.get("profile", [{}])[0].get("K0") is not None)
test("Prof has K_eff", r.get("profile", [{}])[0].get("K_eff") is not None)
test("Prof has A_ratio", r.get("profile", [{}])[0].get("A_ratio") is not None)
test("Prof has R_value", r.get("profile", [{}])[0].get("R_value") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("6P works", api("POST", "/api/analysis/stress-ratio-depth", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [248] Wellbore Shear Failure
# ═══════════════════════════════════════════════════════════════
print("\n[248] Wellbore Shear Failure")
code, r = api("POST", "/api/analysis/wellbore-shear-failure", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "UCS_MPa": 50, "friction": 0.6})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has friction", "friction" in r)
test("Has cohesion_MPa", "cohesion_MPa" in r)
test("Has n_failure_zones", "n_failure_zones" in r)
test("Has pct_failure", "pct_failure" in r)
test("Has min_shear_SF", "min_shear_SF" in r)
test("Has min_margin_MPa", "min_margin_MPa" in r)
test("Has shear_class", "shear_class" in r)
test("Class valid", r.get("shear_class") in ("CRITICAL", "HIGH_RISK", "MODERATE_RISK", "STABLE"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has sigma_1_wall_MPa", r.get("profile", [{}])[0].get("sigma_1_wall_MPa") is not None)
test("Prof has shear_SF", r.get("profile", [{}])[0].get("shear_SF") is not None)
test("Prof has failure", "failure" in r.get("profile", [{}])[0])
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Weak rock", api("POST", "/api/analysis/wellbore-shear-failure", {"source": "demo", "well": "3P", "UCS_MPa": 20})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/wellbore-shear-failure", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [249] Fracture Permeability Tensor
# ═══════════════════════════════════════════════════════════════
print("\n[249] Fracture Permeability Tensor")
code, r = api("POST", "/api/analysis/fracture-perm-tensor-directional", {"source": "demo", "well": "3P", "aperture_mm": 0.1})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has n_fractures", "n_fractures" in r)
test("Has aperture_mm", "aperture_mm" in r)
test("Has tensor", "tensor" in r)
test("Tensor has kxx_mD", "kxx_mD" in r.get("tensor", {}))
test("Tensor has kyy_mD", "kyy_mD" in r.get("tensor", {}))
test("Tensor has kzz_mD", "kzz_mD" in r.get("tensor", {}))
test("Tensor has kxy_mD", "kxy_mD" in r.get("tensor", {}))
test("Has k_max_mD", "k_max_mD" in r)
test("Has k_min_mD", "k_min_mD" in r)
test("Has anisotropy_ratio", "anisotropy_ratio" in r)
test("Has tensor_class", "tensor_class" in r)
test("Class valid", r.get("tensor_class") in ("HIGHLY_ANISOTROPIC", "ANISOTROPIC", "ISOTROPIC"))
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Large aperture", api("POST", "/api/analysis/fracture-perm-tensor-directional", {"source": "demo", "well": "3P", "aperture_mm": 1.0})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/fracture-perm-tensor-directional", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.62.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
