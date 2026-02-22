"""Test suite for GeoStress AI v3.5.0 / v3.6.0 / v3.7.0 / v3.8.0 / v3.9.0.

Tests: input validation, field calibration, error boundaries,
uncertainty quantification, decision matrix.
"""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8099"


def api(method, path, body=None, timeout=60):
    """Call API and return parsed JSON."""
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def api_expect_error(method, path, body=None, expected_status=400):
    """Call API expecting an HTTP error. Returns True if expected status received."""
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return False  # Should have raised
    except urllib.error.HTTPError as e:
        return e.code == expected_status


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


# ── 1. Input Validation: Classifier ──────────────────

print("\n[1] Classifier Validation")

check("Invalid classifier returns 400",
      api_expect_error("POST", "/api/analysis/classify",
                       {"classifier": "nonexistent_model"}))

check("Invalid classifier on SHAP returns 400",
      api_expect_error("POST", "/api/analysis/shap",
                       {"classifier": "bad_model"}))

check("Invalid classifier on deep-ensemble returns 400",
      api_expect_error("POST", "/api/analysis/deep-ensemble",
                       {"classifier": "bad_model"}))

check("Invalid classifier on transfer-learning returns 400",
      api_expect_error("POST", "/api/analysis/transfer-learning",
                       {"classifier": "bad_model"}))

check("Invalid classifier on retrain returns 400",
      api_expect_error("POST", "/api/feedback/retrain",
                       {"classifier": "bad_model"}))

check("Valid classifier works",
      api("POST", "/api/analysis/classify",
          {"classifier": "random_forest"}).get("cv_mean_accuracy", 0) > 0)


# ── 2. Input Validation: Regime ──────────────────────

print("\n[2] Regime Validation")

check("Invalid regime on sensitivity returns 400",
      api_expect_error("POST", "/api/analysis/sensitivity",
                       {"regime": "invalid_regime"}))

check("Invalid regime on bayesian returns 400",
      api_expect_error("POST", "/api/analysis/bayesian",
                       {"regime": "bogus"}))

check("Invalid regime on risk-matrix returns 400",
      api_expect_error("POST", "/api/analysis/risk-matrix",
                       {"regime": "fake"}))

check("Valid regime works",
      not api_expect_error("POST", "/api/analysis/sensitivity",
                           {"regime": "strike_slip"}))


# ── 3. Input Validation: Numeric Ranges ──────────────

print("\n[3] Numeric Range Validation")

check("Depth out of range returns 400",
      api_expect_error("POST", "/api/analysis/inversion",
                       {"depth_m": -100}))

check("Depth too large returns 400",
      api_expect_error("POST", "/api/analysis/inversion",
                       {"depth_m": 99999}))

check("Cohesion negative returns 400",
      api_expect_error("POST", "/api/analysis/inversion",
                       {"cohesion": -5}))

check("Pore pressure out of range returns 400",
      api_expect_error("POST", "/api/analysis/inversion",
                       {"pore_pressure": 999}))

check("Rating out of range returns 400",
      api_expect_error("POST", "/api/feedback/submit",
                       {"rating": 99}))

check("Friction out of range on what-if returns 400",
      api_expect_error("POST", "/api/analysis/what-if",
                       {"friction": 10.0}))

check("n_clusters out of range returns 400",
      api_expect_error("POST", "/api/analysis/cluster",
                       {"n_clusters": 0}))

check("fine_tune_fraction out of range returns 400",
      api_expect_error("POST", "/api/analysis/transfer-learning",
                       {"fine_tune_fraction": 5.0}))


# ── 4. Input Validation: Well Not Found ──────────────

print("\n[4] Well Not Found Validation")

check("Unknown well on cluster returns 404",
      api_expect_error("POST", "/api/analysis/cluster",
                       {"well": "NONEXISTENT_WELL"}, expected_status=404))

check("Unknown well on inversion returns 404",
      api_expect_error("POST", "/api/analysis/inversion",
                       {"well": "FAKE_WELL"}, expected_status=404))


# ── 5. Error Boundaries ─────────────────────────────

print("\n[5] Global Error Boundaries")

d = api("GET", "/api/system/health")
check("Health endpoint works", d.get("status") == "HEALTHY")
check("Version is 3.x",
      d.get("app_version", "").startswith("3."))


# ── 6. Field Calibration: Add Measurement ────────────

print("\n[6] Field Calibration - Add Measurement")

m1 = api("POST", "/api/calibration/add-measurement", {
    "well": "3P",
    "test_type": "LOT",
    "depth_m": 3200,
    "measured_stress_mpa": 48.5,
    "stress_direction": "Shmin",
    "notes": "Test LOT at reservoir top"
})
check("Add LOT measurement", m1.get("status") == "ok")
check("Measurement has ID", len(m1.get("measurement", {}).get("id", "")) > 0)

m2 = api("POST", "/api/calibration/add-measurement", {
    "well": "3P",
    "test_type": "XLOT",
    "depth_m": 3350,
    "measured_stress_mpa": 62.0,
    "stress_direction": "SHmax",
    "azimuth_deg": 145,
    "notes": "XLOT in pay zone"
})
check("Add XLOT measurement with azimuth", m2.get("status") == "ok")
check("Total count incremented", m2.get("total_for_well", 0) >= 2)


# ── 7. Field Calibration: Validation Errors ──────────

print("\n[7] Field Calibration - Input Validation")

check("Invalid test type returns 400",
      api_expect_error("POST", "/api/calibration/add-measurement",
                       {"well": "3P", "test_type": "invalid_test",
                        "depth_m": 3000, "measured_stress_mpa": 50,
                        "stress_direction": "Shmin"}))

check("Invalid stress direction returns 400",
      api_expect_error("POST", "/api/calibration/add-measurement",
                       {"well": "3P", "test_type": "LOT",
                        "depth_m": 3000, "measured_stress_mpa": 50,
                        "stress_direction": "bad_dir"}))

check("Missing well returns 400",
      api_expect_error("POST", "/api/calibration/add-measurement",
                       {"well": "", "test_type": "LOT",
                        "depth_m": 3000, "measured_stress_mpa": 50,
                        "stress_direction": "Shmin"}))


# ── 8. Field Calibration: Validate ───────────────────

print("\n[8] Field Calibration - Validate Against Model")

v = api("POST", "/api/calibration/validate", {
    "well": "3P", "source": "demo", "depth_m": 3300, "pp_mpa": 30
})
check("Validation returns comparisons", len(v.get("comparisons", [])) > 0)
check("Has calibration score", 0 <= v.get("calibration_score", -1) <= 100)
check("Has overall rating",
      v.get("overall_rating") in ["CALIBRATED", "ACCEPTABLE",
                                   "NEEDS_RECALIBRATION", "UNRELIABLE"])
check("Has model predictions",
      "sigma1_mpa" in v.get("model_predictions", {}))
check("Has recommendations", len(v.get("recommendations", [])) > 0)
check("Has industry context", "oil industry" in v.get("industry_context", "").lower())

# Check individual comparison structure
comp = v["comparisons"][0]
check("Comparison has test_type", "test_type" in comp)
check("Comparison has measured_mpa", "measured_mpa" in comp)
check("Comparison has predicted_mpa", "predicted_mpa" in comp)


# ── 9. Field Calibration: No Measurements ────────────

print("\n[9] Field Calibration - Empty Well")

v_empty = api("POST", "/api/calibration/validate", {
    "well": "6P", "source": "demo", "depth_m": 3000, "pp_mpa": 30
})
check("Empty well returns no_measurements",
      v_empty.get("status") == "no_measurements")
check("Has helpful message", "measurement" in v_empty.get("message", "").lower())


# ── 10. Get Measurements ─────────────────────────────

print("\n[10] Get Measurements")

meas = api("GET", "/api/calibration/measurements?well=3P")
check("Get measurements returns list",
      isinstance(meas.get("measurements"), list) and len(meas["measurements"]) > 0)

meas_all = api("GET", "/api/calibration/measurements")
check("Get all measurements returns dict",
      isinstance(meas_all.get("measurements"), dict))


# ── 11. Existing Endpoints Still Work ────────────────

print("\n[11] Regression - Core Endpoints")

d = api("GET", "/api/data/summary?source=demo")
check("Data summary works", d.get("total_fractures", 0) > 0)

inv = api("POST", "/api/analysis/inversion", {
    "well": "3P", "regime": "auto", "source": "demo"
})
check("Inversion works", "shmax_azimuth_deg" in inv)

clf = api("POST", "/api/analysis/classify", {
    "classifier": "random_forest", "source": "demo"
})
check("Classification works", clf.get("cv_mean_accuracy", 0) > 0)


# ── 12. v3.6.0: Inversion Uncertainty ─────────────────

print("\n[12] Inversion Uncertainty (v3.6.0)")

inv2 = api("POST", "/api/analysis/inversion", {
    "well": "3P", "regime": "strike_slip", "source": "demo"
})
unc = inv2.get("uncertainty", {})
check("Has uncertainty block", len(unc) > 0)
check("Has shmax_ci_90", isinstance(unc.get("shmax_ci_90"), list) and len(unc["shmax_ci_90"]) == 2)
check("Has shmax_std_deg", isinstance(unc.get("shmax_std_deg"), (int, float)))
check("Has quality assessment", unc.get("quality") in ["WELL_CONSTRAINED", "MODERATELY_CONSTRAINED", "POORLY_CONSTRAINED"])
check("Has sigma1_ci_90", isinstance(unc.get("sigma1_ci_90"), list) and len(unc["sigma1_ci_90"]) == 2)

cs_range = inv2.get("critically_stressed_range", {})
check("Has CS% range", "best_estimate" in cs_range and "low_friction" in cs_range)
check("CS% range has note", "note" in cs_range)


# ── 13. v3.6.0: Classification Confidence ──────────────

print("\n[13] Classification Confidence (v3.6.0)")

clf2 = api("POST", "/api/analysis/classify", {
    "classifier": "random_forest", "source": "demo"
})
conf = clf2.get("confidence", {})
check("Has confidence block", len(conf) > 0)
check("Has mean_prediction_confidence", isinstance(conf.get("mean_prediction_confidence"), (int, float)))
check("Mean confidence > 0.5", (conf.get("mean_prediction_confidence") or 0) > 0.5)
check("Has per_class_confidence", len(conf.get("per_class_confidence", {})) > 0)
check("Has accuracy_range", isinstance(conf.get("accuracy_range"), list) and len(conf["accuracy_range"]) == 2)


# ── 14. v3.6.0: Executive Decision Matrix ─────────────

print("\n[14] Executive Decision Matrix (v3.6.0)")

exec_r = api("POST", "/api/analysis/executive-summary", {
    "source": "demo", "well": "3P", "depth": 3000
}, timeout=120)
dm = exec_r.get("decision_matrix", {})
check("Has decision_matrix", len(dm) > 0)
check("Has verdict", dm.get("verdict") in ["GO", "CONDITIONAL GO", "NO-GO"])
check("Has verdict_note", isinstance(dm.get("verdict_note"), str) and len(dm["verdict_note"]) > 0)
check("Has 6 factors", len(dm.get("factors", [])) == 6)
factor_names = [f["factor"] for f in dm.get("factors", [])]
check("Has Data Sufficiency factor", "Data Sufficiency" in factor_names)
check("Has Safety Margin factor", "Safety Margin" in factor_names)
check("Has WSM Quality factor", "WSM Quality" in factor_names)
check("Has Data Quality factor", "Data Quality" in factor_names)
for f in dm.get("factors", []):
    check(f'Factor "{f["factor"]}" has valid status',
          f.get("status") in ["GREEN", "AMBER", "RED"],
          f.get("status"))


# ── 15. v3.7.0: WSM Quality Ranking ──────────────────

print("\n[15] WSM Quality Ranking (v3.7.0)")

inv3 = api("POST", "/api/analysis/inversion", {
    "well": "3P", "regime": "strike_slip", "source": "demo"
})
unc3 = inv3.get("uncertainty", {})
check("Has WSM quality rank", unc3.get("wsm_quality_rank") in ["A", "B", "C", "D", "E"])
check("WSM rank detail is string", isinstance(unc3.get("wsm_quality_detail"), str) and
      len(unc3["wsm_quality_detail"]) > 10)
check("WSM rank A or B for 3P", unc3.get("wsm_quality_rank") in ["A", "B"],
      unc3.get("wsm_quality_rank"))


# ── 16. v3.7.0: Calibration Warning ─────────────────

print("\n[16] Calibration Warning (v3.7.0)")

cw = inv3.get("calibration_warning", {})
check("Has calibration warning", cw.get("requires_calibration") is True)
check("Warning mentions LOT/XLOT", "LOT" in cw.get("message", ""))
check("Lists reliable outputs", "shmax_azimuth_deg" in cw.get("reliable_outputs", []))
check("Lists outputs needing validation", "sigma1" in cw.get("requires_validation", []))


# ── 17. v3.7.0: Multi-Criteria CS% ──────────────────

print("\n[17] Multi-Criteria CS% (v3.7.0)")

mc = inv3.get("multi_criteria_cs", {})
check("Has Mohr-Coulomb CS%", isinstance(mc.get("mohr_coulomb_pct"), (int, float)))
check("Has Mogi-Coulomb CS%", isinstance(mc.get("mogi_coulomb_pct"), (int, float)))
check("Has Drucker-Prager CS%", isinstance(mc.get("drucker_prager_pct"), (int, float)))
check("Multi-criteria has note", len(mc.get("note", "")) > 20)
check("Different criteria give different values",
      mc.get("mohr_coulomb_pct") != mc.get("drucker_prager_pct"))


# ── 18. v3.7.0: Stress Polygon ──────────────────────

print("\n[18] Stress Polygon (v3.7.0)")

sp = inv3.get("stress_polygon", {})
check("Has stress polygon", len(sp) > 0)
check("Has Sv", isinstance(sp.get("sv_mpa"), (int, float)))
check("Has frictional limit ratio", sp.get("frictional_limit_ratio", 0) > 1)
check("Has normal fault bounds", "shmin_range_mpa" in sp.get("normal_fault", {}))
check("Has strike-slip bounds", "shmax_range_mpa" in sp.get("strike_slip", {}))
check("Has thrust fault bounds", "shmax_range_mpa" in sp.get("thrust_fault", {}))

# Stress polygon endpoint
sp2 = api("POST", "/api/analysis/stress-polygon", {"depth_m": 3300, "mu": 0.6})
check("Stress polygon endpoint works", sp2.get("frictional_limit_ratio", 0) > 1)
check("Has friction sensitivity", len(sp2.get("friction_sensitivity", {})) == 3)

# Stress polygon validation on inversion
sp_val = inv3.get("stress_polygon", {}).get("validation", {})
check("Has stress polygon validation",
      sp_val.get("status") in ["WITHIN_BOUNDS", "NEAR_LIMIT", "EXCEEDS_BOUNDS"])
check("Has effective stress ratio", isinstance(sp_val.get("effective_ratio"), (int, float)))

# Multi-criteria CS% ordering: Mogi > MC > DP (DP is most conservative)
mc_mc = inv3.get("multi_criteria_cs", {}).get("mohr_coulomb_pct", -1)
mc_dp = inv3.get("multi_criteria_cs", {}).get("drucker_prager_pct", -1)
mc_mg = inv3.get("multi_criteria_cs", {}).get("mogi_coulomb_pct", -1)
check("Mogi-Coulomb CS% >= MC CS% (sigma2 correction)",
      mc_mg >= mc_mc, f"Mogi={mc_mg}% >= MC={mc_mc}%")
check("Drucker-Prager CS% <= MC CS% (conservative smooth yield)",
      mc_dp <= mc_mc, f"DP={mc_dp}% <= MC={mc_mc}%")


# ── 19. v3.7.0: Mud Weight Window ───────────────────

print("\n[19] Mud Weight Window (v3.7.0)")

mw = inv3.get("mud_weight_window", {})
check("Inversion includes mud weight", len(mw) > 0)
check("Has safe window", "safe_window" in mw)
check("Has pore pressure in ppg",
      isinstance(mw.get("pore_pressure", {}).get("ppg"), (int, float)))
check("Has status", mw.get("status") in ["SAFE", "NARROW", "IMPOSSIBLE"])

# Dedicated endpoint
mw2 = api("POST", "/api/analysis/mud-weight-window", {
    "well": "3P", "source": "demo", "depth_m": 3300
}, timeout=120)
check("MWW endpoint has safe window", "safe_window" in mw2)
check("MWW has depth profile", len(mw2.get("depth_profile", [])) > 0)
check("MWW profile has status", mw2["depth_profile"][0].get("status") is not None)


# ── 20. v3.7.0: Fracture QC ─────────────────────────

print("\n[20] Fracture QC (v3.7.0)")

qc = api("GET", "/api/data/qc?source=demo")
check("QC has total", qc.get("total", 0) > 0)
check("QC has pass rate", 0 <= qc.get("pass_rate", -1) <= 1)
check("QC has flags", isinstance(qc.get("flags"), dict))
check("QC has azimuth quality", isinstance(qc.get("azimuth_quality"), dict))
check("QC has WSM note", "WSM" in qc.get("wsm_note", ""))

# Per-well QC
qc_3p = api("GET", "/api/data/qc?source=demo&well=3P")
check("Per-well QC works", qc_3p.get("total", 0) > 0)
check("Well 3P has better pass rate", qc_3p.get("pass_rate", 0) > qc.get("pass_rate", 0))


# ── 21. v3.7.0: Spatial (Depth-Blocked) CV ──────────

print("\n[21] Spatial (Depth-Blocked) CV (v3.7.0)")

clf3 = api("POST", "/api/analysis/classify", {
    "classifier": "random_forest", "source": "demo"
})
sp_cv = clf3.get("spatial_cv", {})
check("Has spatial CV results", len(sp_cv) > 0)
check("Has spatial CV accuracy",
      isinstance(sp_cv.get("spatial_cv_accuracy"), (int, float)))
check("Has spatial CV note", len(sp_cv.get("note", "")) > 20)
check("Spatial CV < random CV (spatial autocorrelation)",
      sp_cv.get("spatial_cv_accuracy", 1) < clf3.get("cv_mean_accuracy", 0),
      f"spatial={sp_cv.get('spatial_cv_accuracy')}, random={clf3.get('cv_mean_accuracy')}")


# ── [22] 1D Stress Profile ─────────────────────────────────────
print("\n[22] 1D Stress Profile")
sp = api("POST", "/api/analysis/stress-profile", {
    "well": "3P", "depth_min": 1000, "depth_max": 5000, "n_points": 10
})
check("Has profile array", isinstance(sp.get("profile"), list) and len(sp["profile"]) > 0)
check("Has plot image", isinstance(sp.get("plot_img"), str) and len(sp["plot_img"]) > 100)
check("Has regime", sp.get("regime") in ["normal", "thrust", "strike_slip"])
check("Has SHmax azimuth", isinstance(sp.get("shmax_azimuth_deg"), (int, float)))
check("Has R ratio", isinstance(sp.get("R"), (int, float)) and 0 <= sp["R"] <= 1)
check("Has reference depth", isinstance(sp.get("reference_depth_m"), (int, float)))
check("Has note with caveats", "hydrostatic" in sp.get("note", "").lower())

# Verify depth ordering
if sp.get("profile"):
    depths = [p["depth_m"] for p in sp["profile"]]
    check("Depths are monotonically increasing", depths == sorted(depths))
    # Sv should increase with depth
    sv_vals = [p["sv_mpa"] for p in sp["profile"]]
    check("Sv increases with depth", sv_vals == sorted(sv_vals))
    # All stresses should be positive
    all_pos = all(p["sv_mpa"] > 0 and p["shmax_mpa"] > 0 and p["shmin_mpa"] > 0 and p["pp_mpa"] > 0 for p in sp["profile"])
    check("All stresses positive", all_pos)
    # Sv > Pp at all depths (physical requirement)
    sv_gt_pp = all(p["sv_mpa"] > p["pp_mpa"] for p in sp["profile"])
    check("Sv > Pp at all depths", sv_gt_pp)

# Test with well 6P
sp2 = api("POST", "/api/analysis/stress-profile", {
    "well": "6P", "depth_min": 2000, "depth_max": 4000, "n_points": 5
})
check("6P stress profile works", len(sp2.get("profile", [])) == 5)

# Test invalid well
sp3_ok = api_expect_error("POST", "/api/analysis/stress-profile",
                          {"well": "INVALID_WELL"}, expected_status=404)
check("Invalid well returns 404", sp3_ok)

# ── [23] QC in Overview ────────────────────────────────────────
print("\n[23] QC in Overview")
ov = api("POST", "/api/analysis/overview", {"well": "3P"})
qc_s = ov.get("qc_summary")
check("Overview has QC summary", qc_s is not None)
if qc_s:
    check("QC has total", isinstance(qc_s.get("total"), int) and qc_s["total"] > 0)
    check("QC has passed count", isinstance(qc_s.get("passed"), int))
    check("QC has pass rate pct", isinstance(qc_s.get("pass_rate_pct"), (int, float)))
    check("QC pass rate is percentage", 0 <= qc_s["pass_rate_pct"] <= 100)
    check("QC has WSM note", len(qc_s.get("wsm_note", "")) > 10)
    check("3P pass rate > 90%", qc_s["pass_rate_pct"] > 90,
          f"pass_rate={qc_s['pass_rate_pct']}%")

# ── [24] Conformal Prediction (v3.8.0) ─────────────────────────
print("\n[24] Conformal Prediction")
clf4 = api("POST", "/api/analysis/classify", {
    "well": "3P", "classifier": "xgboost", "source": "demo"
})
cp = clf4.get("conformal_prediction")
check("Has conformal prediction", cp is not None)
if cp:
    check("Has coverage target", cp.get("coverage_target") == 0.9)
    check("Empirical coverage >= target",
          cp.get("empirical_coverage", 0) >= cp.get("coverage_target", 1))
    check("Has avg prediction set size",
          isinstance(cp.get("avg_prediction_set_size"), (int, float)))
    check("Set size <= n_classes",
          cp["avg_prediction_set_size"] <= cp.get("n_classes", 999))
    check("Has precision ratio",
          isinstance(cp.get("precision_ratio"), (int, float)))
    check("Precision > 0.5 (model is useful)", cp["precision_ratio"] > 0.5,
          f"precision={cp['precision_ratio']}")
    check("Has explanatory note", len(cp.get("note", "")) > 20)


# ── [25] Cost-Sensitive Learning (v3.8.0) ──────────────────────
print("\n[25] Cost-Sensitive Learning")
cs = api("POST", "/api/analysis/cost-sensitive", {
    "classifier": "xgboost", "false_negative_cost": 10
})
check("Has high-risk classes", isinstance(cs.get("high_risk_classes"), list) and len(cs["high_risk_classes"]) > 0)
check("Has standard accuracy", isinstance(cs.get("standard_accuracy"), (int, float)))
check("Has cost-sensitive accuracy", isinstance(cs.get("cost_sensitive_accuracy"), (int, float)))
check("Has per-class comparison", isinstance(cs.get("per_class_comparison"), list))
check("Has interpretation", len(cs.get("interpretation", "")) > 30)
check("Has note about 2025 literature", "2025" in cs.get("note", ""))
# Check that high-risk classes have recall data
for cc in cs.get("per_class_comparison", []):
    if cc.get("high_risk"):
        check(f"High-risk {cc['class']} has recall data",
              isinstance(cc.get("cost_sensitive_recall"), (int, float)))
        break

# ── [26] Physics Check (v3.8.0) ────────────────────────────────
print("\n[26] Physics Check")
pc = api("POST", "/api/analysis/physics-check", {
    "well": "3P", "depth": 3000
})
check("Physics check returns result", isinstance(pc, dict))
check("Has checks or constraints", len(pc) > 0)

# ── [27] Physics-Constrained Prediction ────────────────────────
print("\n[27] Physics-Constrained Prediction")
pp = api("POST", "/api/analysis/physics-predict", {
    "well": "3P", "depth": 3000, "fast": True
})
check("Physics prediction returns result", isinstance(pp, dict))

# ══════════════════════════════════════════════════════
# v3.9.0 — Stakeholder Intelligence
# ══════════════════════════════════════════════════════

# ── [28] Inversion Stakeholder Brief ─────────────────
print("\n[28] Inversion Stakeholder Brief")
inv = api("POST", "/api/analysis/inversion",
    {"well": "3P", "regime": "auto", "depth_m": 3000})
sb = inv.get("stakeholder_brief", {})
check("Has stakeholder_brief", bool(sb))
check("Has headline", "headline" in sb and len(sb["headline"]) > 10)
check("Has risk_level", sb.get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Has confidence_sentence", "confidence_sentence" in sb and len(sb["confidence_sentence"]) > 10)
check("Has next_action", "next_action" in sb and len(sb["next_action"]) > 5)
check("Has suitable_for list", isinstance(sb.get("suitable_for"), list) and len(sb["suitable_for"]) > 0)
check("Has not_suitable_for list", isinstance(sb.get("not_suitable_for"), list) and len(sb["not_suitable_for"]) > 0)
check("Has critically_stressed_plain", "critically_stressed_plain" in sb)
check("Has feedback_note", "feedback_note" in sb)
check("Headline mentions well", "3P" in sb.get("headline", ""))

# ── [29] Classification Stakeholder Brief ────────────
print("\n[29] Classification Stakeholder Brief")
clf = api("POST", "/api/analysis/classify",
    {"classifier": "random_forest"})
cb = clf.get("stakeholder_brief", {})
check("Has stakeholder_brief", bool(cb))
check("Has headline", "headline" in cb and len(cb["headline"]) > 5)
check("Has verdict", "verdict" in cb)
check("Has what_it_means", "what_it_means" in cb)
check("Has confidence_sentence", "confidence_sentence" in cb)
check("Has action recommendation", "action" in cb)
check("Has limiting_class info", "limiting_class" in cb)

# ── [30] Model Comparison Stakeholder Brief ──────────
print("\n[30] Model Comparison Stakeholder Brief")
mc = api("POST", "/api/analysis/compare-models",
    {"fast": True}, timeout=120)
mb = mc.get("stakeholder_brief", {})
check("Has stakeholder_brief", bool(mb))
check("Has headline", "headline" in mb and len(mb["headline"]) > 5)
check("Has what_agreement_means", "what_agreement_means" in mb)
check("Has model_to_use", "model_to_use" in mb)
check("Has caution", "caution" in mb)

# ── [31] Cost-Sensitive Stakeholder Brief ────────────
print("\n[31] Cost-Sensitive Stakeholder Brief")
cs = api("POST", "/api/analysis/cost-sensitive",
    {"classifier": "xgboost"}, timeout=120)
csb = cs.get("stakeholder_brief", {})
check("Has stakeholder_brief", bool(csb))
check("Has headline", "headline" in csb and len(csb["headline"]) > 5)
check("Has tradeoff_explained", "tradeoff_explained" in csb)
check("Has high_risk_classes", isinstance(csb.get("high_risk_classes"), list))
check("Has recommended_use", "recommended_use" in csb)

# ── [32] Overview Stakeholder Brief ──────────────────
print("\n[32] Overview Stakeholder Brief")
ov = api("POST", "/api/analysis/overview",
    {"well": "3P", "regime": "auto"})
ob = ov.get("stakeholder_brief", {})
check("Has stakeholder_brief", bool(ob))
check("Has headline", "headline" in ob and len(ob["headline"]) > 5)
check("Has confidence_sentence", "confidence_sentence" in ob)
check("Has feedback_note", "feedback_note" in ob)
check("Headline mentions well", "3P" in ob.get("headline", ""))

# ── [33] RLHF Stakeholder Brief ─────────────────────
print("\n[33] RLHF Stakeholder Brief")
rq = api("POST", "/api/rlhf/review-queue",
    {"well": "3P", "n_samples": 5})
rb = rq.get("stakeholder_brief", {})
check("Has stakeholder_brief", bool(rb))
check("Has why_these_samples", "why_these_samples" in rb)
check("Has what_to_look_for", "what_to_look_for" in rb)
check("Has what_happens_next", "what_happens_next" in rb)
check("Has progress", "progress" in rb)

# ── [34] Feedback Receipt ────────────────────────────
print("\n[34] Feedback Receipt")
fb = api("POST", "/api/feedback/submit", {
    "well": "3P", "analysis_type": "inversion",
    "rating": 4, "comment": "Test feedback from v3.9 tests"
})
check("Has feedback_receipt", "feedback_receipt" in fb)
fr = fb.get("feedback_receipt", {})
check("Has recorded_at", "recorded_at" in fr)
check("Has what_happens_next", "what_happens_next" in fr)
check("Has current_average_rating", "current_average_rating" in fr)
check("Has n_ratings_total", "n_ratings_total" in fr)

# ── [35] Correction Receipt ──────────────────────────
print("\n[35] Correction Receipt")
cr = api("POST", "/api/feedback/correct-label", {
    "well": "3P", "fracture_idx": 0,
    "original_type": "Continuous", "corrected_type": "Discontinuous"
})
check("Has correction_receipt", "correction_receipt" in cr)
crr = cr.get("correction_receipt", {})
check("Has recorded_at", "recorded_at" in crr)
check("Has what_happens_next", "what_happens_next" in crr)
check("Has corrections_pending", "corrections_pending" in crr)
check("Has ready_to_retrain", "ready_to_retrain" in crr)
check("Has expected_improvement", "expected_improvement" in crr)

# ── [36] RLHF Accept Receipt ────────────────────────
print("\n[36] RLHF Accept Receipt")
ar = api("POST", "/api/rlhf/accept-reject", {
    "well": "3P", "verdict": "accept",
    "sample_index": 0, "predicted_type": "Continuous"
})
check("Has receipt", "receipt" in ar)
rr = ar.get("receipt", {})
check("Has verdict_recorded", rr.get("verdict_recorded") == "accept")
check("Has impact", "impact" in rr)
check("Has reviewed_this_session", "reviewed_this_session" in rr)

# ── [37] Verdict Consistency ─────────────────────────
print("\n[37] Verdict Consistency")
# Overview risk should match inversion risk assessment
check("Overview has risk level",
      ov.get("risk", {}).get("level") in ("LOW", "MODERATE", "HIGH", "UNKNOWN"))
# Inversion brief risk should match CS% assessment
inv_cs_pct = inv.get("critically_stressed_pct", 0)
inv_risk = sb.get("risk_level", "")
if inv_cs_pct < 10:
    check("CS<10% = GREEN risk", inv_risk == "GREEN",
          f"cs={inv_cs_pct}% risk={inv_risk}")
elif inv_cs_pct <= 30:
    check("CS 10-30% = AMBER risk", inv_risk == "AMBER",
          f"cs={inv_cs_pct}% risk={inv_risk}")
else:
    check("CS>30% = RED risk", inv_risk == "RED",
          f"cs={inv_cs_pct}% risk={inv_risk}")

# ── [38] A/B Test Endpoint ─────────────────────────
print("\n[38] A/B Test Endpoint")
ab = api("POST", "/api/models/ab-test", {"source": "demo", "model_a": "gradient_boosting", "model_b": "random_forest"})
check("Has model_a", "model_a" in ab and "name" in ab["model_a"])
check("Has model_b", "model_b" in ab and "name" in ab["model_b"])
check("Has agreement", "agreement" in ab)
agr = ab.get("agreement", {})
check("Agreement total > 0", agr.get("total", 0) > 0)
check("Agreement pct in range", 0 <= agr.get("agreement_pct", -1) <= 100)
check("Has verdict", ab.get("verdict") in ("EQUIVALENT", "MODEL_A_BETTER", "MODEL_B_BETTER"))
check("Has disagreements list", isinstance(ab.get("disagreements"), list))

# ── [39] A/B Test Stakeholder Brief ─────────────────
print("\n[39] A/B Test Stakeholder Brief")
ab_sb = ab.get("stakeholder_brief", {})
check("Has headline", len(ab_sb.get("headline", "")) > 10)
check("Has risk_level", ab_sb.get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Has winner", "winner" in ab_sb)
check("Has confidence_sentence", "accuracy" in ab_sb.get("confidence_sentence", ""))
check("Has action", len(ab_sb.get("action", "")) > 5)
check("Has disagreement_note", len(ab_sb.get("disagreement_note", "")) > 5)

# ── [40] Version Compare Stakeholder Brief ──────────
print("\n[40] Version Compare Stakeholder Brief")
vc = api("POST", "/api/models/compare-versions", {"well": "3P"})
if vc.get("verdict"):
    vc_sb = vc.get("stakeholder_brief", {})
    check("Has headline", len(vc_sb.get("headline", "")) > 10)
    check("Has risk_level", vc_sb.get("risk_level") in ("GREEN", "AMBER", "RED"))
    check("Has action", len(vc_sb.get("action", "")) > 5)
    check("Has what_changed", "Accuracy" in vc_sb.get("what_changed", ""))
    check("Has suitable_for", isinstance(vc_sb.get("suitable_for"), list))
else:
    check("Version compare needs 2+ versions (ok)", "message" in vc)

# ── Summary ──────────────────────────────────────────

print(f"\n{'='*50}")
print(f"v3.9.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
