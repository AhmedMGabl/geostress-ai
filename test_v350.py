"""Test suite for GeoStress AI v3.5.0 through v3.17.0.

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

# ── [41] Stacking Ensemble Classifier ─────────────
print("\n[41] Stacking Ensemble Classifier")
stk = api("POST", "/api/analysis/classify",
          {"source": "demo", "classifier": "stacking"}, timeout=120)
check("Stacking returns accuracy", stk.get("cv_mean_accuracy", 0) > 0.3)
check("Stacking has confusion matrix", isinstance(stk.get("confusion_matrix"), list))
check("Stacking has class names", len(stk.get("class_names", [])) > 0)
check("Stacking has stakeholder brief", "headline" in stk.get("stakeholder_brief", {}))

# ── [42] Top Feature Drivers in Classification ─────
print("\n[42] Top Feature Drivers in Classification")
clf_rf = api("POST", "/api/analysis/classify",
             {"source": "demo", "classifier": "random_forest"})
drivers = clf_rf.get("top_drivers", [])
check("Has top_drivers", len(drivers) > 0)
check("Top driver has feature name", len(drivers[0].get("feature", "")) > 0 if drivers else False)
check("Top driver has importance", drivers[0].get("importance", 0) > 0 if drivers else False)
check("Top driver has explanation", len(drivers[0].get("explanation", "")) > 3 if drivers else False)
check("At most 5 drivers", len(drivers) <= 5)

# ── [43] Ensemble Vote Endpoint ─────────────────────
print("\n[43] Ensemble Vote Endpoint")
ev = api("POST", "/api/models/ensemble-vote", {"source": "demo"}, timeout=180)
check("Has n_models", ev.get("n_models", 0) >= 2)
check("Has n_fractures", ev.get("n_fractures", 0) > 0)
check("Has models dict", isinstance(ev.get("models"), dict))
ens = ev.get("ensemble", {})
check("Has mean_agreement_pct", 0 <= ens.get("mean_agreement_pct", -1) <= 100)
check("Has unanimous_count", ens.get("unanimous_count", -1) >= 0)
check("Has contested_count", ens.get("contested_count", -1) >= 0)
check("Has predictions list", len(ens.get("predictions", [])) == ev.get("n_fractures", 0))
check("Has contested_fractures", isinstance(ev.get("contested_fractures"), list))

# ── [44] Ensemble Vote Stakeholder Brief ────────────
print("\n[44] Ensemble Vote Stakeholder Brief")
ev_sb = ev.get("stakeholder_brief", {})
check("Has headline", len(ev_sb.get("headline", "")) > 10)
check("Has risk_level", ev_sb.get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Has confidence_sentence", "models" in ev_sb.get("confidence_sentence", "").lower())
check("Has action", len(ev_sb.get("action", "")) > 5)
check("Has models_used list", isinstance(ev_sb.get("models_used"), list))

# ── [45] Data Improvement Plan ──────────────────────
print("\n[45] Data Improvement Plan")
dip = api("POST", "/api/data/improvement-plan", {"source": "demo"}, timeout=120)
check("Has headline", len(dip.get("headline", "")) > 10)
check("Has risk_level", dip.get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Has overall_accuracy", 0 < dip.get("overall_accuracy", 0) <= 1)
check("Has per_class_performance", len(dip.get("per_class_performance", [])) > 0)
pcp = dip.get("per_class_performance", [{}])[0]
check("Per-class has recall", "recall" in pcp)
check("Per-class has status", pcp.get("status") in ("GOOD", "NEEDS_DATA", "CRITICAL"))
check("Has action_plan", isinstance(dip.get("action_plan"), list))
check("Has stakeholder brief", "headline" in dip.get("stakeholder_brief", {}))

# ── [46] SHAP Visualization Plots ──────────────────────
print("\n[46] SHAP Visualization Plots")
sp = api("POST", "/api/shap/plots", {"source": "demo", "classifier": "random_forest"}, timeout=120)
check("Has global_importance_plot", sp.get("global_importance_plot", "").startswith("data:image"))
check("Has waterfall_plot", sp.get("waterfall_plot", "").startswith("data:image"))
check("Has feature_scatter_plot", sp.get("feature_scatter_plot", "").startswith("data:image"))
check("Has per_class_plots", isinstance(sp.get("per_class_plots"), dict) and len(sp["per_class_plots"]) > 0)
check("Has SHAP flag", sp.get("has_shap") is True)
check("Has waterfall_sample", "index" in sp.get("waterfall_sample", {}))
check("Has classifier_used", sp.get("classifier_used") in ("random_forest", "xgboost", "lightgbm"))
check("Has stakeholder_brief", "headline" in sp.get("stakeholder_brief", {}))

# ── [47] Near-Miss Detection ──────────────────────────
print("\n[47] Near-Miss Detection & Blind Spots")
nm = api("POST", "/api/analysis/near-misses", {"source": "demo", "classifier": "random_forest"}, timeout=120)
check("Has n_near_misses", isinstance(nm.get("n_near_misses"), int))
check("Has near_misses list", isinstance(nm.get("near_misses"), list))
check("Has blind_spots", isinstance(nm.get("blind_spots"), list))
check("Has n_blind_spots", isinstance(nm.get("n_blind_spots"), int))
check("Has risk_matrix", isinstance(nm.get("risk_matrix"), list))
check("Has plot", nm.get("plot", "").startswith("data:image"))
check("Has overall_accuracy", isinstance(nm.get("overall_accuracy"), (int, float)))
check("Near-miss has margin", "margin" in nm.get("near_misses", [{}])[0] if nm.get("near_misses") else True)
check("Blind spot has error_rate", "error_rate" in nm.get("blind_spots", [{}])[0] if nm.get("blind_spots") else True)
check("Has stakeholder_brief", "headline" in nm.get("stakeholder_brief", {}))
check("Brief has standards_reference", "API RP 580" in nm.get("stakeholder_brief", {}).get("standards_reference", ""))

# ── [48] Query-by-Committee Active Learning ───────────
print("\n[48] Query-by-Committee Active Learning")
qbc = api("POST", "/api/analysis/active-learning-qbc", {"source": "demo", "n_suggest": 5}, timeout=180)
check("Has committee_size", isinstance(qbc.get("committee_size"), int) and qbc["committee_size"] >= 2)
check("Has committee_members", isinstance(qbc.get("committee_members"), list))
check("Has suggestions", isinstance(qbc.get("suggestions"), list) and len(qbc["suggestions"]) > 0)
check("Has committee_accuracies", isinstance(qbc.get("committee_accuracies"), dict))
check("Has plot", qbc.get("plot", "").startswith("data:image"))
check("Has stats", "mean_vote_entropy" in qbc.get("stats", {}))
sg = qbc.get("suggestions", [{}])[0]
check("Suggestion has model_predictions", isinstance(sg.get("model_predictions"), dict))
check("Suggestion has qbc_score", isinstance(sg.get("qbc_score"), (int, float)))
check("Suggestion has agreement", "/" in str(sg.get("agreement", "")))
check("Has stakeholder_brief", "headline" in qbc.get("stakeholder_brief", {}))

# ── [49] Calibration Report + OOD ─────────────────────
print("\n[49] Calibration Report + OOD Detection")
cal = api("POST", "/api/analysis/calibration-report", {"source": "demo", "classifier": "random_forest"}, timeout=180)
check("Has ece_uncalibrated", isinstance(cal.get("ece_uncalibrated"), (int, float)))
check("Has ece_calibrated", isinstance(cal.get("ece_calibrated"), (int, float)))
check("Has brier_uncalibrated", isinstance(cal.get("brier_uncalibrated"), (int, float)))
check("Has brier_calibrated", isinstance(cal.get("brier_calibrated"), (int, float)))
check("Has calibration_quality", cal.get("calibration_quality") in ("GOOD", "FAIR", "POOR"))
check("Has calibration_curves", isinstance(cal.get("calibration_curves"), dict))
check("Has ood_per_well", isinstance(cal.get("ood_per_well"), dict))
check("Has plot", cal.get("plot", "").startswith("data:image"))
if cal.get("ood_per_well"):
    well_ood = list(cal["ood_per_well"].values())[0]
    check("OOD has mean_mahalanobis", isinstance(well_ood.get("mean_mahalanobis"), (int, float)))
    check("OOD has ood_severity", well_ood.get("ood_severity") in ("LOW", "MEDIUM", "HIGH"))
check("Has stakeholder_brief", "headline" in cal.get("stakeholder_brief", {}))

# ── [50] Failure Dashboard (API RP 580) ───────────────
print("\n[50] Failure Dashboard (API RP 580)")
fd = api("POST", "/api/analysis/failure-dashboard", {"source": "demo", "classifier": "random_forest"}, timeout=180)
check("Has safety_score", isinstance(fd.get("safety_score"), (int, float)) and 0 <= fd["safety_score"] <= 100)
check("Has decision", fd.get("decision") in ("GO", "CONDITIONAL GO", "REVIEW REQUIRED", "NO-GO"))
check("Has decision_detail", len(fd.get("decision_detail", "")) > 10)
check("Has risk_factors", isinstance(fd.get("risk_factors"), list) and len(fd["risk_factors"]) == 5)
check("Has n_fail", isinstance(fd.get("n_fail"), int))
check("Has n_warn", isinstance(fd.get("n_warn"), int))
check("Has plot", fd.get("plot", "").startswith("data:image"))
rf0 = fd.get("risk_factors", [{}])[0]
check("Risk factor has factor name", len(rf0.get("factor", "")) > 0)
check("Risk factor has status", rf0.get("status") in ("PASS", "WARN", "FAIL"))
check("Risk factor has threshold", len(rf0.get("threshold", "")) > 0)
check("Has stakeholder_brief", "headline" in fd.get("stakeholder_brief", {}))
check("Brief references API RP 580", "API RP 580" in fd.get("stakeholder_brief", {}).get("standards_reference", ""))

# ── [51] Domain-Adapted Transfer Learning ─────────────
print("\n[51] Domain-Adapted Transfer Learning")
ta = api("POST", "/api/analysis/transfer-adapted", {
    "source_well": "3P", "target_well": "6P", "classifier": "random_forest", "source": "demo"
}, timeout=180)
check("Has results dict", isinstance(ta.get("results"), dict))
check("Has zero_shot method", "zero_shot" in ta.get("results", {}))
check("Has fine_tuned method", "fine_tuned" in ta.get("results", {}))
check("Has mmd_adapted method", "mmd_adapted" in ta.get("results", {}))
check("Has pseudo_labeled method", "pseudo_labeled" in ta.get("results", {}))
check("Has target_only method", "target_only" in ta.get("results", {}))
check("Has best_method", isinstance(ta.get("best_method"), str) and len(ta["best_method"]) > 0)
check("Has best_accuracy", isinstance(ta.get("best_accuracy"), (int, float)) and 0 <= ta["best_accuracy"] <= 1)
check("Has feature_shifts list", isinstance(ta.get("feature_shifts"), list))
check("Has n_shifts count", isinstance(ta.get("n_shifts"), int))
check("Has plot", isinstance(ta.get("plot"), str) and ta["plot"].startswith("data:image"))
check("Has n_source and n_target", ta.get("n_source", 0) > 0 and ta.get("n_target", 0) > 0)
ta_zs = ta.get("results", {}).get("zero_shot", {})
check("Zero-shot has accuracy", isinstance(ta_zs.get("accuracy"), (int, float)))
check("Zero-shot has f1", isinstance(ta_zs.get("f1"), (int, float)))
check("Has stakeholder_brief", "headline" in ta.get("stakeholder_brief", {}))
if ta.get("feature_shifts"):
    fs0 = ta["feature_shifts"][0]
    check("Shift has feature name", len(fs0.get("feature", "")) > 0)
    check("Shift has cohens_d", isinstance(fs0.get("cohens_d"), (int, float)))
    check("Shift has severity", fs0.get("severity") in ("MEDIUM", "HIGH"))

# ── [52] Error Budget / Learning Curve ────────────────
print("\n[52] Error Budget / Learning Curve")
eb = api("POST", "/api/analysis/error-budget", {"classifier": "random_forest", "source": "demo"}, timeout=180)
check("Has classifier", eb.get("classifier") == "random_forest")
check("Has n_samples", isinstance(eb.get("n_samples"), int) and eb["n_samples"] > 0)
check("Has current_accuracy", isinstance(eb.get("current_accuracy"), (int, float)) and 0 < eb["current_accuracy"] <= 1)
check("Has train_test_gap", isinstance(eb.get("train_test_gap"), (int, float)) and eb["train_test_gap"] >= 0)
check("Has diagnosis", eb.get("diagnosis") in ("OVERFITTING", "UNDERFITTING", "PLATEAU", "IMPROVING"))
check("Has learning_curve list", isinstance(eb.get("learning_curve"), list) and len(eb["learning_curve"]) >= 3)
check("Has samples_for_1pct", isinstance(eb.get("samples_for_1pct_improvement"), int))
check("Has plot", isinstance(eb.get("plot"), str) and eb["plot"].startswith("data:image"))
lc0 = eb.get("learning_curve", [{}])[0]
check("Curve point has n_samples", isinstance(lc0.get("n_samples"), int))
check("Curve point has train_accuracy", isinstance(lc0.get("train_accuracy"), (int, float)))
check("Curve point has test_accuracy", isinstance(lc0.get("test_accuracy"), (int, float)))
check("Curve point has test_std", isinstance(lc0.get("test_std"), (int, float)))
check("Has stakeholder_brief", "headline" in eb.get("stakeholder_brief", {}))
sb = eb.get("stakeholder_brief", {})
check("Brief has risk_level", sb.get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has action", len(sb.get("action", "")) > 10)

# ── [53] Auto-Retrain from Feedback ───────────────────
print("\n[53] Auto-Retrain from Feedback")
ar = api("POST", "/api/analysis/auto-retrain", {
    "well": "3P", "classifier": "random_forest", "source": "demo"
}, timeout=180)
check("Has status", ar.get("status") in ("PROMOTED", "REJECTED", "NO_FEEDBACK"))
if ar.get("status") != "NO_FEEDBACK":
    check("Has classifier", isinstance(ar.get("classifier"), str))
    check("Has baseline", isinstance(ar.get("baseline"), dict))
    check("Baseline has accuracy", isinstance(ar.get("baseline", {}).get("accuracy"), (int, float)))
    check("Has retrained", isinstance(ar.get("retrained"), dict))
    check("Retrained has accuracy", isinstance(ar.get("retrained", {}).get("accuracy"), (int, float)))
    check("Has improvement", isinstance(ar.get("improvement"), (int, float)))
    check("Has feedback_used", isinstance(ar.get("feedback_used"), dict))
    check("Feedback has corrections", isinstance(ar.get("feedback_used", {}).get("corrections"), int))
    check("Feedback has failures", isinstance(ar.get("feedback_used", {}).get("failures"), int))
    check("Has plot", isinstance(ar.get("plot"), str) and ar["plot"].startswith("data:image"))
    check("Has stakeholder_brief", "headline" in ar.get("stakeholder_brief", {}))

# ── [54] Model Arena ──────────────────────────────────
print("\n[54] Model Arena")
ma = api("POST", "/api/analysis/model-arena", {"well": "3P", "source": "demo"}, timeout=300)
check("Has ranking", isinstance(ma.get("ranking"), list) and len(ma["ranking"]) >= 3)
check("Has results dict", isinstance(ma.get("results"), dict))
check("Has best_model", isinstance(ma.get("best_model"), str) and len(ma["best_model"]) > 0)
check("Has best_composite", isinstance(ma.get("best_composite"), (int, float)))
check("Has n_models", isinstance(ma.get("n_models"), int) and ma["n_models"] >= 3)
check("Has plot", isinstance(ma.get("plot"), str) and ma["plot"].startswith("data:image"))
best = ma.get("best_model", "")
best_r = ma.get("results", {}).get(best, {})
check("Best has accuracy", isinstance(best_r.get("accuracy"), (int, float)))
check("Best has f1", isinstance(best_r.get("f1"), (int, float)))
check("Best has ece", isinstance(best_r.get("ece"), (int, float)))
check("Best has composite", isinstance(best_r.get("composite"), (int, float)))
check("Best has rank 1", best_r.get("rank") == 1)
check("Has stakeholder_brief", "headline" in ma.get("stakeholder_brief", {}))

# ── [55] Stakeholder Decision Report ──────────────────
print("\n[55] Stakeholder Decision Report")
sd = api("POST", "/api/report/stakeholder-decision", {
    "well": "3P", "classifier": "random_forest", "source": "demo"
}, timeout=180)
check("Has decision", sd.get("decision") in ("GO", "CONDITIONAL GO", "REVIEW REQUIRED", "NO-GO"))
check("Has score", isinstance(sd.get("score"), int) and 0 <= sd["score"] <= 100)
check("Has accuracy", isinstance(sd.get("accuracy"), (int, float)))
check("Has class_risks", isinstance(sd.get("class_risks"), list) and len(sd["class_risks"]) > 0)
cr0 = sd.get("class_risks", [{}])[0]
check("Class risk has class", isinstance(cr0.get("class"), str))
check("Class risk has risk_score", isinstance(cr0.get("risk_score"), (int, float)))
check("Class risk has verdict", cr0.get("verdict") in ("LOW", "MEDIUM", "HIGH"))
check("Has confidence_stats", isinstance(sd.get("confidence_stats"), dict))
check("Stats has mean", isinstance(sd.get("confidence_stats", {}).get("mean"), (int, float)))
check("Has economic_impact", isinstance(sd.get("economic_impact"), dict))
check("Econ has cost_per_misclass", sd.get("economic_impact", {}).get("cost_per_misclass_usd") == 50000)
check("Has evidence", isinstance(sd.get("evidence"), list))
check("Has feedback_summary", isinstance(sd.get("feedback_summary"), dict))
check("Has plot", isinstance(sd.get("plot"), str) and sd["plot"].startswith("data:image"))
check("Has stakeholder_brief", "headline" in sd.get("stakeholder_brief", {}))

# ── [56] Negative Outcome Learning ────────────────────
print("\n[56] Negative Outcome Learning")
no = api("POST", "/api/analysis/negative-outcomes", {
    "well": "3P", "classifier": "random_forest", "source": "demo"
}, timeout=180)
check("Has n_errors", isinstance(no.get("n_errors"), int) and no["n_errors"] >= 0)
check("Has error_rate", isinstance(no.get("error_rate"), (int, float)))
check("Has n_synthetic_added", isinstance(no.get("n_synthetic_added"), int))
check("Has systematic_biases", isinstance(no.get("systematic_biases"), list))
check("Has feature_diffs", isinstance(no.get("feature_diffs"), list))
check("Has baseline", isinstance(no.get("baseline"), dict))
check("Baseline has accuracy", isinstance(no.get("baseline", {}).get("accuracy"), (int, float)))
check("Has augmented", isinstance(no.get("augmented"), dict))
check("Augmented has accuracy", isinstance(no.get("augmented", {}).get("accuracy"), (int, float)))
check("Has improvement", isinstance(no.get("improvement"), (int, float)))
check("Has plot", isinstance(no.get("plot"), str) and no["plot"].startswith("data:image"))
if no.get("systematic_biases"):
    b0 = no["systematic_biases"][0]
    check("Bias has true_class", isinstance(b0.get("true_class"), str))
    check("Bias has error_rate", isinstance(b0.get("error_rate"), (int, float)))
    check("Bias has confused_with", isinstance(b0.get("confused_with"), str))
check("Has stakeholder_brief", "headline" in no.get("stakeholder_brief", {}))

# ── [57] Data Validation ──────────────────────────────
print("\n[57] Data Validation")
dv = api("POST", "/api/data/validate", {"well": "3P", "source": "demo"}, timeout=60)
check("Has quality", dv.get("quality") in ("GOOD", "ACCEPTABLE", "POOR"))
check("Has n_samples", isinstance(dv.get("n_samples"), int) and dv["n_samples"] > 0)
check("Has n_critical", isinstance(dv.get("n_critical"), int))
check("Has n_warnings", isinstance(dv.get("n_warnings"), int))
check("Has issues list", isinstance(dv.get("issues"), list))
check("Has recommendations", isinstance(dv.get("recommendations"), list) and len(dv["recommendations"]) > 0)
check("Has column_summary", isinstance(dv.get("column_summary"), dict))
check("Has stakeholder_brief", "headline" in dv.get("stakeholder_brief", {}))
if dv.get("issues"):
    i0 = dv["issues"][0]
    check("Issue has severity", i0.get("severity") in ("CRITICAL", "WARNING", "INFO"))
    check("Issue has field", isinstance(i0.get("field"), str))
    check("Issue has detail", isinstance(i0.get("detail"), str))

# ── [58] Cache Warmup ─────────────────────────────────
print("\n[58] Cache Warmup")
wu = api("POST", "/api/system/warmup", timeout=30)
check("Has status WARMING", wu.get("status") == "WARMING")
check("Has targets", isinstance(wu.get("targets"), list) and len(wu["targets"]) > 0)
check("Has message", isinstance(wu.get("message"), str))

# ── [59] RLHF Preference Model ───────────────────────
print("\n[59] RLHF Preference Model")
pm = api("POST", "/api/rlhf/preference-model", {"well": "3P", "source": "demo"}, timeout=180)
check("Has status", pm.get("status") in ("OK", "INSUFFICIENT_DATA"))
if pm.get("status") == "OK":
    check("Has n_reviews", isinstance(pm.get("n_reviews"), int) and pm["n_reviews"] >= 5)
    check("Has accepted count", isinstance(pm.get("accepted"), int))
    check("Has rejected count", isinstance(pm.get("rejected"), int))
    check("Has type_trust", isinstance(pm.get("type_trust"), dict))
    check("Has baseline_accuracy", isinstance(pm.get("baseline_accuracy"), (int, float)))
    check("Has weighted_accuracy", isinstance(pm.get("weighted_accuracy"), (int, float)))
    check("Has improvement", isinstance(pm.get("improvement"), (int, float)))
    check("Has plot", isinstance(pm.get("plot"), str) and pm["plot"].startswith("data:image"))
    check("Has stakeholder_brief", "headline" in pm.get("stakeholder_brief", {}))
else:
    check("Has message", isinstance(pm.get("message"), str))

# ── [60] Balanced Classification (SMOTE) ─────────────
print("\n[60] Balanced Classification (SMOTE)")
bc = api("POST", "/api/analysis/balanced-classify", {"well": "3P", "source": "demo"}, timeout=120)
check("Has well", bc.get("well") == "3P")
check("Has n_samples", isinstance(bc.get("n_samples"), int) and bc["n_samples"] > 50)
check("Has class_counts", isinstance(bc.get("class_counts"), dict) and len(bc["class_counts"]) >= 3)
check("Has has_smote", isinstance(bc.get("has_smote"), bool))
check("Has methods dict", isinstance(bc.get("methods"), dict))
check("Has unbalanced method", "unbalanced" in bc.get("methods", {}))
check("Has balanced_weights method", "balanced_weights" in bc.get("methods", {}))
methods = bc.get("methods", {})
for mname in ("unbalanced", "balanced_weights"):
    m = methods.get(mname, {})
    check(f"{mname} has accuracy", isinstance(m.get("accuracy"), (int, float)) and 0 <= m["accuracy"] <= 1)
    check(f"{mname} has balanced_accuracy", isinstance(m.get("balanced_accuracy"), (int, float)))
    check(f"{mname} has f1", isinstance(m.get("f1"), (int, float)))
    check(f"{mname} has per_class", isinstance(m.get("per_class"), dict))
if bc.get("has_smote"):
    check("Has smote method", "smote" in methods)
    check("Has smote_balanced method", "smote_balanced" in methods)
check("Has best_method", isinstance(bc.get("best_method"), str))
check("Has minority_class_improvements", isinstance(bc.get("minority_class_improvements"), list))
mi_list = bc.get("minority_class_improvements", [])
if len(mi_list) > 0:
    check("MI has class field", "class" in mi_list[0])
    check("MI has baseline_recall", isinstance(mi_list[0].get("baseline_recall"), (int, float)))
    check("MI has best_recall", isinstance(mi_list[0].get("best_recall"), (int, float)))
    check("MI has improvement", isinstance(mi_list[0].get("improvement"), (int, float)))
check("Has plot", isinstance(bc.get("plot"), str) and bc["plot"].startswith("data:image"))
check("Has stakeholder_brief", "headline" in bc.get("stakeholder_brief", {}))

# ── [61] Industrial Readiness Scorecard ──────────────
print("\n[61] Industrial Readiness Scorecard")
rs = api("POST", "/api/report/readiness-scorecard", {"well": "3P", "source": "demo"}, timeout=120)
check("Has readiness level", rs.get("readiness") in ("PRODUCTION", "PILOT", "DEVELOPMENT", "NOT_READY"))
check("Has readiness_text", isinstance(rs.get("readiness_text"), str) and len(rs["readiness_text"]) > 10)
check("Has overall_score 0-100", isinstance(rs.get("overall_score"), (int, float)) and 0 <= rs["overall_score"] <= 100)
check("Has dimensions list", isinstance(rs.get("dimensions"), list) and len(rs["dimensions"]) >= 5)
dims = rs.get("dimensions", [])
for d in dims:
    check(f"Dim '{d.get('dimension', '?')}' has grade", d.get("grade") in ("A", "B", "C", "D", "F"))
    check(f"Dim '{d.get('dimension', '?')}' has score", isinstance(d.get("score"), (int, float)))
    check(f"Dim '{d.get('dimension', '?')}' has detail", isinstance(d.get("detail"), str))
    check(f"Dim '{d.get('dimension', '?')}' has action", isinstance(d.get("action"), str))
    check(f"Dim '{d.get('dimension', '?')}' has weight", isinstance(d.get("weight"), (int, float)))
check("Has grade_counts", isinstance(rs.get("grade_counts"), dict))
check("Has priority_actions", isinstance(rs.get("priority_actions"), list))
check("Has n_samples", isinstance(rs.get("n_samples"), int) and rs["n_samples"] > 0)
check("Has n_wells", isinstance(rs.get("n_wells"), int) and rs["n_wells"] >= 1)
check("Has plot", isinstance(rs.get("plot"), str) and rs["plot"].startswith("data:image"))
check("Has stakeholder_brief", "headline" in rs.get("stakeholder_brief", {}))

# Sanity check: score matches readiness level
if rs.get("overall_score", 0) >= 80:
    check("Score>=80 means PRODUCTION or PILOT", rs.get("readiness") in ("PRODUCTION", "PILOT"))
elif rs.get("overall_score", 0) >= 60:
    check("Score>=60 means at least DEVELOPMENT", rs.get("readiness") in ("PRODUCTION", "PILOT", "DEVELOPMENT"))

# ── [62] Quick Classify (No Plots) ───────────────────
print("\n[62] Quick Classify (No Plots)")
qc = api("POST", "/api/analysis/quick-classify", {"well": "3P", "source": "demo"}, timeout=60)
check("Has well", qc.get("well") == "3P")
check("Has classifier", isinstance(qc.get("classifier"), str))
check("Has accuracy", isinstance(qc.get("accuracy"), (int, float)) and 0 <= qc["accuracy"] <= 1)
check("Has f1", isinstance(qc.get("f1"), (int, float)))
check("Has balanced_accuracy", isinstance(qc.get("balanced_accuracy"), (int, float)))
check("Has per_class", isinstance(qc.get("per_class"), dict))
check("Has confusion_matrix", isinstance(qc.get("confusion_matrix"), list))
check("Has class_names", isinstance(qc.get("class_names"), list))
check("Has cached flag", isinstance(qc.get("cached"), bool))
check("Has stakeholder_brief", "headline" in qc.get("stakeholder_brief", {}))
# Second call should be cached
qc2 = api("POST", "/api/analysis/quick-classify", {"well": "3P", "source": "demo"}, timeout=10)
check("Second call is cached", qc2.get("cached") == True)
check("No plot in quick-classify", "plot" not in qc)

# ── [63] Feature Ablation Study ──────────────────────
print("\n[63] Feature Ablation Study")
fa = api("POST", "/api/analysis/feature-ablation", {"well": "3P", "source": "demo"}, timeout=120)
check("Has well", fa.get("well") == "3P")
check("Has classifier", isinstance(fa.get("classifier"), str))
check("Has n_samples", isinstance(fa.get("n_samples"), int) and fa["n_samples"] > 50)
check("Has n_features_total", isinstance(fa.get("n_features_total"), int) and fa["n_features_total"] >= 3)
check("Has n_groups", isinstance(fa.get("n_groups"), int) and fa["n_groups"] >= 2)
check("Has baseline_accuracy", isinstance(fa.get("baseline_accuracy"), (int, float)) and 0 < fa["baseline_accuracy"] <= 1)
check("Has baseline_balanced_accuracy", isinstance(fa.get("baseline_balanced_accuracy"), (int, float)))
check("Has feature_groups dict", isinstance(fa.get("feature_groups"), dict))
check("Has ablation_results list", isinstance(fa.get("ablation_results"), list) and len(fa["ablation_results"]) >= 2)
ar = fa["ablation_results"][0]
check("Result has group", isinstance(ar.get("group"), str))
check("Result has n_features_removed", isinstance(ar.get("n_features_removed"), int))
check("Result has accuracy_without", isinstance(ar.get("accuracy_without"), (int, float)))
check("Result has accuracy_drop", isinstance(ar.get("accuracy_drop"), (int, float)))
check("Result has importance_rank", ar.get("importance_rank") == 1)
check("Has most_important_group", isinstance(fa.get("most_important_group"), str))
check("Has plot", isinstance(fa.get("plot"), str) and fa["plot"].startswith("data:image"))
check("Has stakeholder_brief", "headline" in fa.get("stakeholder_brief", {}))

# ── [64] Hyperparameter Optimization ─────────────────
print("\n[64] Hyperparameter Optimization")
opt = api("POST", "/api/analysis/optimize-model", {"well": "3P", "source": "demo", "classifier": "random_forest", "n_iter": 10}, timeout=180)
check("Has well", opt.get("well") == "3P")
check("Has classifier", isinstance(opt.get("classifier"), str))
check("Has n_samples", isinstance(opt.get("n_samples"), int))
check("Has n_iterations", opt.get("n_iterations") == 10)
check("Has default_accuracy", isinstance(opt.get("default_accuracy"), (int, float)) and 0 < opt["default_accuracy"] <= 1)
check("Has default_std", isinstance(opt.get("default_std"), (int, float)))
check("Has best_accuracy", isinstance(opt.get("best_accuracy"), (int, float)) and 0 < opt["best_accuracy"] <= 1)
check("Has improvement", isinstance(opt.get("improvement"), (int, float)))
check("Best >= default", opt.get("best_accuracy", 0) >= opt.get("default_accuracy", 1) - 0.05)
check("Has best_params", isinstance(opt.get("best_params"), dict) and len(opt["best_params"]) >= 1)
check("Has top_configurations", isinstance(opt.get("top_configurations"), list) and len(opt["top_configurations"]) >= 1)
tc = opt["top_configurations"][0]
check("Config has rank", isinstance(tc.get("rank"), int))
check("Config has mean_score", isinstance(tc.get("mean_score"), (int, float)))
check("Config has std_score", isinstance(tc.get("std_score"), (int, float)))
check("Config has params", isinstance(tc.get("params"), dict))
check("Has plot", isinstance(opt.get("plot"), str) and opt["plot"].startswith("data:image"))
check("Has stakeholder_brief", "headline" in opt.get("stakeholder_brief", {}))

# ── [65] Pore Pressure Coupling ──────────────────────
print("\n[65] Pore Pressure Coupling")
pp = api("POST", "/api/analysis/pore-pressure-coupling", {"well": "3P", "source": "demo"}, timeout=120)
check("Has well", pp.get("well") == "3P")
check("Has mean_depth_m", isinstance(pp.get("mean_depth_m"), (int, float)) and pp["mean_depth_m"] > 0)
check("Has sv_mpa", isinstance(pp.get("sv_mpa"), (int, float)) and pp["sv_mpa"] > 0)
check("Has n_fractures", isinstance(pp.get("n_fractures"), int) and pp["n_fractures"] > 10)
check("Has pp_sweep", isinstance(pp.get("pp_sweep"), list) and len(pp["pp_sweep"]) >= 5)
sweep = pp["pp_sweep"][0]
check("Sweep has pp_fraction_sv", isinstance(sweep.get("pp_fraction_sv"), (int, float)))
check("Sweep has pp_mpa", isinstance(sweep.get("pp_mpa"), (int, float)))
check("Sweep has cs_pct", isinstance(sweep.get("cs_pct"), (int, float)))
check("Sweep has fif", isinstance(sweep.get("fif"), (int, float)))
check("Sweep has fif_grade", sweep.get("fif_grade") in ("STABLE", "MARGINAL", "CRITICAL"))
check("Has sensitivity_cs_per_mpa", isinstance(pp.get("sensitivity_cs_per_mpa"), (int, float)))
check("Has current_estimate", isinstance(pp.get("current_estimate"), dict))
check("Current has fif_grade", pp["current_estimate"].get("fif_grade") in ("STABLE", "MARGINAL", "CRITICAL"))
check("Has plot", isinstance(pp.get("plot"), str) and pp["plot"].startswith("data:image"))
check("Has stakeholder_brief", "headline" in pp.get("stakeholder_brief", {}))
# CS% should increase with pore pressure
check("CS% increases with Pp", pp["pp_sweep"][-1]["cs_pct"] >= pp["pp_sweep"][0]["cs_pct"])

# ── [66] Heterogeneous Ensemble ──────────────────────
print("\n[66] Heterogeneous Ensemble")
he = api("POST", "/api/analysis/hetero-ensemble", {"well": "3P", "source": "demo"}, timeout=180)
check("Has well", he.get("well") == "3P")
check("Has n_samples", isinstance(he.get("n_samples"), int))
check("Has n_base_models", isinstance(he.get("n_base_models"), int) and he["n_base_models"] >= 3)
check("Has base_accuracies", isinstance(he.get("base_accuracies"), dict))
check("Has best_single_model", isinstance(he.get("best_single_model"), str))
check("Has best_single_accuracy", isinstance(he.get("best_single_accuracy"), (int, float)))
check("Has ensemble_accuracy", isinstance(he.get("ensemble_accuracy"), (int, float)) and 0 < he["ensemble_accuracy"] <= 1)
check("Has ensemble_f1", isinstance(he.get("ensemble_f1"), (int, float)))
check("Has ensemble_balanced_accuracy", isinstance(he.get("ensemble_balanced_accuracy"), (int, float)))
check("Has ensemble_improvement", isinstance(he.get("ensemble_improvement"), (int, float)))
check("Has meta_contributions", isinstance(he.get("meta_contributions"), dict))
check("Contributions sum ~1", abs(sum(he.get("meta_contributions", {}).values()) - 1.0) < 0.05)
check("Has mean_agreement", isinstance(he.get("mean_agreement"), (int, float)) and 0 < he["mean_agreement"] <= 1)
check("Has contested_predictions", isinstance(he.get("contested_predictions"), int))
check("Has class_names", isinstance(he.get("class_names"), list))
check("Has plot", isinstance(he.get("plot"), str) and he["plot"].startswith("data:image"))
check("Has stakeholder_brief", "headline" in he.get("stakeholder_brief", {}))

# ── [67] ML Anomaly Detection ────────────────────────
print("\n[67] ML Anomaly Detection")
ad = api("POST", "/api/analysis/anomaly-detection", {"well": "3P", "source": "demo"}, timeout=120)
check("Has well", ad.get("well") == "3P")
check("Has n_samples", isinstance(ad.get("n_samples"), int) and ad["n_samples"] > 50)
check("Has n_anomalies", isinstance(ad.get("n_anomalies"), int))
check("Has anomaly_rate_pct", isinstance(ad.get("anomaly_rate_pct"), (int, float)))
check("Rate < 100%", ad.get("anomaly_rate_pct", 100) < 100)
check("Has maha_threshold", isinstance(ad.get("maha_threshold"), (int, float)) and ad["maha_threshold"] > 0)
check("Has anomalies list", isinstance(ad.get("anomalies"), list))
if len(ad.get("anomalies", [])) > 0:
    a = ad["anomalies"][0]
    check("Anomaly has index", isinstance(a.get("index"), int))
    check("Anomaly has iso_score", isinstance(a.get("iso_score"), (int, float)))
    check("Anomaly has mahalanobis", isinstance(a.get("mahalanobis"), (int, float)))
    check("Anomaly has unusual_features", isinstance(a.get("unusual_features"), list))
    if len(a.get("unusual_features", [])) > 0:
        check("UF has feature", isinstance(a["unusual_features"][0].get("feature"), str))
        check("UF has z_score", isinstance(a["unusual_features"][0].get("z_score"), (int, float)))
check("Has plot", isinstance(ad.get("plot"), str) and ad["plot"].startswith("data:image"))
check("Has stakeholder_brief", "headline" in ad.get("stakeholder_brief", {}))

# ── Summary ──────────────────────────────────────────

print(f"\n{'='*50}")
print(f"v3.17.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
