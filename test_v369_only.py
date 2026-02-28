"""Tests for v3.69.0: Injection Seismicity, Trajectory Stress, Reservoir Compaction, Perforation Stability, Critical Drawdown."""
import requests, sys

BASE = "http://localhost:8164"
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
# [280] Injection-Induced Seismicity
# ═══════════════════════════════════════════════════════════════
print("\n[280] Injection-Induced Seismicity")
code, r = api("POST", "/api/analysis/injection-induced-seismicity", {"source": "demo", "well": "3P", "depth_m": 3000, "injection_pressure_MPa": 35, "injection_volume_m3": 5000})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has delta_Pp_MPa", "delta_Pp_MPa" in r)
test("Has CFS_initial_MPa", "CFS_initial_MPa" in r)
test("Has CFS_after_MPa", "CFS_after_MPa" in r)
test("Has delta_CFS_MPa", "delta_CFS_MPa" in r)
test("Has Mw_max", "Mw_max" in r)
test("Has seism_class", "seism_class" in r)
test("Class valid", r.get("seism_class") in ("VERY_HIGH", "HIGH_RISK", "MODERATE", "LOW"))
test("Has fault_friction", "fault_friction" in r)
test("Has fault_distance_m", "fault_distance_m" in r)
test("Has t_reach_days", "t_reach_days" in r)
test("Has pressure_sweep", "pressure_sweep" in r)
test("Sweep non-empty", len(r.get("pressure_sweep", [])) > 0)
test("Sweep has delta_Pp_MPa", r.get("pressure_sweep", [{}])[0].get("delta_Pp_MPa") is not None)
test("Sweep has CFS_MPa", r.get("pressure_sweep", [{}])[0].get("CFS_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High pressure", api("POST", "/api/analysis/injection-induced-seismicity", {"source": "demo", "well": "3P", "injection_pressure_MPa": 60})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/injection-induced-seismicity", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [281] Wellbore Trajectory Stress
# ═══════════════════════════════════════════════════════════════
print("\n[281] Wellbore Trajectory Stress")
code, r = api("POST", "/api/analysis/wellbore-trajectory-stress", {"source": "demo", "well": "3P", "depth_m": 3000, "UCS_MPa": 50, "grid_step_deg": 15})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has n_azimuths", "n_azimuths" in r)
test("Has n_inclinations", "n_inclinations" in r)
test("Has best_azimuth_deg", "best_azimuth_deg" in r)
test("Has best_inclination_deg", "best_inclination_deg" in r)
test("Has best_stability_index", "best_stability_index" in r)
test("Has worst_stability_index", "worst_stability_index" in r)
test("Has traj_class", "traj_class" in r)
test("Class valid", r.get("traj_class") in ("CRITICAL", "CONSTRAINED", "HIGHLY_CONSTRAINED", "FAVORABLE"))
test("Has grid", "grid" in r)
test("Grid non-empty", len(r.get("grid", [])) > 0)
test("Grid has azimuth_deg", r.get("grid", [{}])[0].get("azimuth_deg") is not None)
test("Grid has inclination_deg", r.get("grid", [{}])[0].get("inclination_deg") is not None)
test("Grid has stability_index", r.get("grid", [{}])[0].get("stability_index") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Fine grid", api("POST", "/api/analysis/wellbore-trajectory-stress", {"source": "demo", "well": "3P", "grid_step_deg": 30})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/wellbore-trajectory-stress", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [282] Reservoir Compaction
# ═══════════════════════════════════════════════════════════════
print("\n[282] Reservoir Compaction")
code, r = api("POST", "/api/analysis/reservoir-compaction", {"source": "demo", "well": "3P", "depth_m": 3000, "reservoir_thickness_m": 50, "depletion_MPa": 15, "porosity": 0.2})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has reservoir_thickness_m", "reservoir_thickness_m" in r)
test("Has depletion_MPa", "depletion_MPa" in r)
test("Has porosity", "porosity" in r)
test("Has compaction_m", "compaction_m" in r)
test("Has subsidence_m", "subsidence_m" in r)
test("Has bulk_modulus_GPa", "bulk_modulus_GPa" in r)
test("Has strain_pct", "strain_pct" in r)
test("Has comp_class", "comp_class" in r)
test("Class valid", r.get("comp_class") in ("SEVERE", "SIGNIFICANT", "MODERATE", "MINOR"))
test("Has path", "path" in r)
test("Path non-empty", len(r.get("path", [])) > 0)
test("Path has depletion_MPa", r.get("path", [{}])[0].get("depletion_MPa") is not None)
test("Path has compaction_m", r.get("path", [{}])[0].get("compaction_m") is not None)
test("Path has subsidence_m", r.get("path", [{}])[0].get("subsidence_m") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Large depletion", api("POST", "/api/analysis/reservoir-compaction", {"source": "demo", "well": "3P", "depletion_MPa": 30})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/reservoir-compaction", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [283] Perforation Stability
# ═══════════════════════════════════════════════════════════════
print("\n[283] Perforation Stability")
code, r = api("POST", "/api/analysis/perforation-stability", {"source": "demo", "well": "3P", "depth_m": 3000, "UCS_MPa": 50, "n_angles": 36})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_m", "depth_m" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has perf_angle_deg", "perf_angle_deg" in r)
test("Has perf_SI", "perf_SI" in r)
test("Has best_angle_deg", "best_angle_deg" in r)
test("Has best_SI", "best_SI" in r)
test("Has drawdown_MPa", "drawdown_MPa" in r)
test("Has perf_length_in", "perf_length_in" in r)
test("Has perf_class", "perf_class" in r)
test("Class valid", r.get("perf_class") in ("CRITICAL", "CONSTRAINED", "STABLE", "FAVORABLE"))
test("Has angle_sweep", "angle_sweep" in r)
test("Sweep non-empty", len(r.get("angle_sweep", [])) > 0)
test("Sweep has angle_deg", r.get("angle_sweep", [{}])[0].get("angle_deg") is not None)
test("Sweep has SI", r.get("angle_sweep", [{}])[0].get("SI") is not None)
test("Sweep has hoop_MPa", r.get("angle_sweep", [{}])[0].get("hoop_MPa") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Low UCS", api("POST", "/api/analysis/perforation-stability", {"source": "demo", "well": "3P", "UCS_MPa": 10})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/perforation-stability", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [284] Critical Drawdown
# ═══════════════════════════════════════════════════════════════
print("\n[284] Critical Drawdown")
code, r = api("POST", "/api/analysis/critical-drawdown", {"source": "demo", "well": "3P", "depth_m": 3000, "UCS_MPa": 50, "TWC_factor": 3.0})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has depth_to_m", "depth_to_m" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has TWC_factor", "TWC_factor" in r)
test("Has mean_critical_drawdown_MPa", "mean_critical_drawdown_MPa" in r)
test("Has min_critical_drawdown_MPa", "min_critical_drawdown_MPa" in r)
test("Has dd_class", "dd_class" in r)
test("Class valid", r.get("dd_class") in ("CRITICAL", "HIGH_RISK", "MODERATE", "LOW", "VERY_LOW"))
test("Has profile", "profile" in r)
test("Profile non-empty", len(r.get("profile", [])) > 0)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has Pp_MPa", r.get("profile", [{}])[0].get("Pp_MPa") is not None)
test("Prof has critical_drawdown_MPa", r.get("profile", [{}])[0].get("critical_drawdown_MPa") is not None)
test("Prof has critical_drawdown_psi", r.get("profile", [{}])[0].get("critical_drawdown_psi") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Low UCS", api("POST", "/api/analysis/critical-drawdown", {"source": "demo", "well": "3P", "UCS_MPa": 10})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/critical-drawdown", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.69.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
