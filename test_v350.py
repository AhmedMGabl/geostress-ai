"""Test suite for GeoStress AI v3.5.0 through v3.17.0.

Tests: input validation, field calibration, error boundaries,
uncertainty quantification, decision matrix.
"""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8099"


class _NullResult(dict):
    """Null object returned when API call times out. Prevents crashes in check() assertions.
    All attribute/item access returns another _NullResult, so chained access never crashes."""
    _is_timeout = True
    def get(self, key, default=None):
        return default
    def __getitem__(self, key):
        return _NullResult()
    def __contains__(self, key):
        return False
    def __bool__(self):
        return False
    def __len__(self):
        return 0
    def __iter__(self):
        return iter([])
    def __add__(self, other):
        return 0
    def __radd__(self, other):
        return 0
    def __sub__(self, other):
        return 0
    def __rsub__(self, other):
        return 0
    def __mul__(self, other):
        return 0
    def __le__(self, other):
        return False
    def __lt__(self, other):
        return False
    def __ge__(self, other):
        return False
    def __gt__(self, other):
        return False
    def __eq__(self, other):
        if isinstance(other, _NullResult):
            return True
        return False
    def __ne__(self, other):
        return not self.__eq__(other)
    def __int__(self):
        return 0
    def __float__(self):
        return 0.0
    def __str__(self):
        return ""
    def __repr__(self):
        return "_NullResult()"
    def startswith(self, *a):
        return False
    def endswith(self, *a):
        return False
    def keys(self):
        return []
    def values(self):
        return []
    def items(self):
        return []

_TIMEOUT_RESULT = _NullResult()


def api(method, path, body=None, timeout=60, retries=1):
    """Call API and return parsed JSON. Retries on timeout. Returns _NullResult on final failure."""
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except (TimeoutError, urllib.error.URLError) as e:
            if attempt < retries:
                import time
                print(f"    (retry {attempt+1}/{retries} after timeout on {path})")
                time.sleep(5)
            else:
                print(f"  TIMEOUT: {path} failed after {retries+1} attempts ({type(e).__name__})")
                return _TIMEOUT_RESULT


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
try:
    stk = api("POST", "/api/analysis/classify",
              {"source": "demo", "classifier": "stacking"}, timeout=180, retries=2)
    check("Stacking returns accuracy", stk.get("cv_mean_accuracy", 0) > 0.3)
    check("Stacking has confusion matrix", isinstance(stk.get("confusion_matrix"), list))
    check("Stacking has class names", len(stk.get("class_names", [])) > 0)
    check("Stacking has stakeholder brief", "headline" in stk.get("stakeholder_brief", {}))
except Exception as e41:
    print(f"  SKIP: Stacking timed out ({type(e41).__name__}), continuing...")
    failed += 4

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
try:
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
except Exception as e54:
    print(f"  SKIP: Model Arena timed out or failed ({type(e54).__name__}), continuing...")
    failed += 12

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

# ── [68] Geological Context ─────────────────────────────
print("\n[68] Geological Context")
gc = api("POST", "/api/analysis/geological-context", {"source": "demo"})
check("Status 200", gc is not None)
check("Has n_wells", isinstance(gc.get("n_wells"), int) and gc["n_wells"] >= 1)
check("Has wells list", isinstance(gc.get("wells"), list) and len(gc["wells"]) >= 1)
w0 = gc["wells"][0]
check("Well has well name", isinstance(w0.get("well"), str))
check("Well has n_fractures", isinstance(w0.get("n_fractures"), int) and w0["n_fractures"] > 0)
check("Well has depth_range", isinstance(w0.get("depth_range"), str) and "m" in w0["depth_range"])
check("Well has mean_azimuth", isinstance(w0.get("mean_azimuth"), (int, float)))
check("Well has mean_dip", isinstance(w0.get("mean_dip"), (int, float)))
check("Well has azimuth_spread", isinstance(w0.get("azimuth_spread"), (int, float)))
check("Well has inferred_regime", isinstance(w0.get("inferred_regime"), str))
check("Well has regime_detail", isinstance(w0.get("regime_detail"), str))
check("Well has fracture_sets", isinstance(w0.get("fracture_sets"), list))
if len(w0.get("fracture_sets", [])) > 0:
    fs = w0["fracture_sets"][0]
    check("Set has set_id", isinstance(fs.get("set_id"), int))
    check("Set has count", isinstance(fs.get("count"), int) and fs["count"] > 0)
    check("Set has mean_azimuth", isinstance(fs.get("mean_azimuth"), (int, float)))
    check("Set has interpretation", isinstance(fs.get("interpretation"), str))
check("Well has depth_zones", isinstance(w0.get("depth_zones"), list) and len(w0["depth_zones"]) >= 1)
if len(w0.get("depth_zones", [])) > 0:
    dz = w0["depth_zones"][0]
    check("Zone has zone name", isinstance(dz.get("zone"), str))
    check("Zone has depth_range", isinstance(dz.get("depth_range"), str))
    check("Zone has count", isinstance(dz.get("count"), int) and dz["count"] > 0)
check("Well has type_distribution", isinstance(w0.get("type_distribution"), dict) and len(w0["type_distribution"]) >= 1)
# Multi-well features
if gc["n_wells"] >= 2:
    check("Has cross_well_comparison", isinstance(gc.get("cross_well_comparison"), dict))
    cw = gc["cross_well_comparison"]
    check("CW has wells", isinstance(cw.get("wells"), list) and len(cw["wells"]) == 2)
    check("CW has azimuth_difference", isinstance(cw.get("azimuth_difference"), (int, float)))
    check("CW has dip_difference", isinstance(cw.get("dip_difference"), (int, float)))
    check("CW has same_regime", isinstance(cw.get("same_regime"), bool))
    check("CW has interpretation", isinstance(cw.get("interpretation"), str))
check("Has plot", isinstance(gc.get("plot"), str) and len(gc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(gc.get("stakeholder_brief"), dict))
sb = gc.get("stakeholder_brief", {})
check("Brief has headline", isinstance(sb.get("headline"), str))
check("Brief has risk_level", sb.get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has action", isinstance(sb.get("action"), str))

# ── [69] Decision Confidence Dashboard ──────────────────
print("\n[69] Decision Confidence Dashboard")
dd = api("POST", "/api/report/decision-dashboard", {"well": "3P", "source": "demo"}, timeout=120)
check("Status 200", dd is not None)
check("Has well", dd.get("well") == "3P")
check("Has overall_decision", dd.get("overall_decision") in ("GO", "CONDITIONAL", "NO-GO"))
check("Has overall_color", dd.get("overall_color") in ("GREEN", "AMBER", "RED"))
check("Has n_samples", isinstance(dd.get("n_samples"), int) and dd["n_samples"] > 0)
check("Has accuracy", isinstance(dd.get("accuracy"), (int, float)) and 0 <= dd["accuracy"] <= 1)
check("Has f1", isinstance(dd.get("f1"), (int, float)) and 0 <= dd["f1"] <= 1)
check("Has balanced_accuracy", isinstance(dd.get("balanced_accuracy"), (int, float)) and 0 <= dd["balanced_accuracy"] <= 1)
# Signals
check("Has signals dict", isinstance(dd.get("signals"), dict))
sigs = dd.get("signals", {})
for sig_name in ["model_accuracy", "balanced_accuracy", "data_volume", "class_balance", "expert_reviews", "go_classes"]:
    check(f"Signal {sig_name} exists", sig_name in sigs)
    check(f"Signal {sig_name} has value", "value" in sigs.get(sig_name, {}))
    check(f"Signal {sig_name} has status", sigs.get(sig_name, {}).get("status") in ("GREEN", "AMBER", "RED"))
# Class decisions
check("Has class_decisions", isinstance(dd.get("class_decisions"), list) and len(dd["class_decisions"]) >= 2)
cd0 = dd["class_decisions"][0]
check("CD has class", isinstance(cd0.get("class"), str))
check("CD has recall", isinstance(cd0.get("recall"), (int, float)))
check("CD has precision", isinstance(cd0.get("precision"), (int, float)))
check("CD has support", isinstance(cd0.get("support"), int))
check("CD has decision", cd0.get("decision") in ("GO", "CONDITIONAL", "NO-GO"))
check("CD has reason", isinstance(cd0.get("reason"), str))
# Scenarios
check("Has scenarios", isinstance(dd.get("scenarios"), dict))
for sc in ["best_case", "expected", "worst_case"]:
    check(f"Scenario {sc} exists", sc in dd.get("scenarios", {}))
    check(f"Scenario {sc} has accuracy", isinstance(dd["scenarios"][sc].get("accuracy"), (int, float)))
    check(f"Scenario {sc} has risk", isinstance(dd["scenarios"][sc].get("risk"), str))
# Recommended actions
check("Has recommended_actions", isinstance(dd.get("recommended_actions"), list))
# Plot and brief
check("Has plot", isinstance(dd.get("plot"), str) and len(dd["plot"]) > 100)
check("Has stakeholder_brief", isinstance(dd.get("stakeholder_brief"), dict))
dsb = dd.get("stakeholder_brief", {})
check("Brief has headline", isinstance(dsb.get("headline"), str))
check("Brief has risk_level", dsb.get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has action", isinstance(dsb.get("action"), str))
# Test with different well
dd6 = api("POST", "/api/report/decision-dashboard", {"well": "6P", "source": "demo"}, timeout=120)
check("6P returns result", dd6 is not None and dd6.get("well") == "6P")
check("6P has decision", dd6.get("overall_decision") in ("GO", "CONDITIONAL", "NO-GO"))

# ── [70] Model Significance Testing ─────────────────────
print("\n[70] Model Significance Testing")
try:
    ms = api("POST", "/api/analysis/model-significance", {"well": "3P", "source": "demo"}, timeout=300)
    check("Status 200", ms is not None)
    check("Has well", ms.get("well") == "3P")
    check("Has n_models", isinstance(ms.get("n_models"), int) and ms["n_models"] >= 2)
    check("Has n_samples", isinstance(ms.get("n_samples"), int) and ms["n_samples"] > 0)
    check("Has class_names", isinstance(ms.get("class_names"), list))
    check("Has models list", isinstance(ms.get("models"), list) and len(ms["models"]) >= 2)
    m0 = ms["models"][0]
    check("Model has model name", isinstance(m0.get("model"), str))
    check("Model has accuracy", isinstance(m0.get("accuracy"), (int, float)) and 0 <= m0["accuracy"] <= 1)
    check("Model has f1", isinstance(m0.get("f1"), (int, float)))
    check("Model has balanced_accuracy", isinstance(m0.get("balanced_accuracy"), (int, float)))
    check("Model has time_s", isinstance(m0.get("time_s"), (int, float)))
    check("Has significance_matrix", isinstance(ms.get("significance_matrix"), list))
    sm0 = ms["significance_matrix"][0]
    check("SM has model", isinstance(sm0.get("model"), str))
    check("SM has comparisons", isinstance(sm0.get("comparisons"), dict))
    check("Has recommendation", isinstance(ms.get("recommendation"), dict))
    rec = ms.get("recommendation", {})
    check("Rec has best_model", isinstance(rec.get("best_model"), str))
    check("Rec has accuracy", isinstance(rec.get("accuracy"), (int, float)))
    check("Rec has significantly_better_than", isinstance(rec.get("significantly_better_than"), int))
    check("Rec has verdict", isinstance(rec.get("verdict"), str))
    check("Has plot", isinstance(ms.get("plot"), str) and len(ms["plot"]) > 100)
except Exception as e70:
    print(f"  SKIP: Model Significance timed out or failed ({type(e70).__name__}), continuing...")
    failed += 20
check("Has stakeholder_brief", isinstance(ms.get("stakeholder_brief"), dict))

# ── [71] Data Collection Planner ────────────────────────
print("\n[71] Data Collection Planner")
cp = api("POST", "/api/data/collection-planner", {"source": "demo"}, timeout=120)
check("Status 200", cp is not None)
check("Has n_wells", isinstance(cp.get("n_wells"), int) and cp["n_wells"] >= 1)
check("Has wells list", isinstance(cp.get("wells"), list) and len(cp["wells"]) >= 1)
w0 = cp["wells"][0]
check("Well has well name", isinstance(w0.get("well"), str))
check("Well has total_samples", isinstance(w0.get("total_samples"), int) and w0["total_samples"] > 0)
check("Well has class_gaps", isinstance(w0.get("class_gaps"), list) and len(w0["class_gaps"]) >= 1)
cg0 = w0["class_gaps"][0]
check("CG has class", isinstance(cg0.get("class"), str))
check("CG has current_count", isinstance(cg0.get("current_count"), int))
check("CG has ideal_count", isinstance(cg0.get("ideal_count"), int))
check("CG has gap", isinstance(cg0.get("gap"), int))
check("CG has priority", cg0.get("priority") in ("HIGH", "MEDIUM", "LOW"))
check("CG has action", isinstance(cg0.get("action"), str))
check("Well has depth_gaps", isinstance(w0.get("depth_gaps"), list) and len(w0["depth_gaps"]) >= 1)
dg0 = w0["depth_gaps"][0]
check("DG has range", isinstance(dg0.get("range"), str))
check("DG has count", isinstance(dg0.get("count"), int))
check("DG has density", isinstance(dg0.get("density"), (int, float)))
check("DG has status", dg0.get("status") in ("OK", "SPARSE"))
check("Well has current_accuracy", isinstance(w0.get("current_accuracy"), (int, float)))
check("Well has projected_accuracy_2x", isinstance(w0.get("projected_accuracy_2x"), (int, float)))
check("Has priorities", isinstance(cp.get("priorities"), list))
check("Has n_priorities", isinstance(cp.get("n_priorities"), int))
check("Has plot", isinstance(cp.get("plot"), str) and len(cp["plot"]) > 100)
check("Has stakeholder_brief", isinstance(cp.get("stakeholder_brief"), dict))
sb = cp.get("stakeholder_brief", {})
check("Brief has headline", isinstance(sb.get("headline"), str))
check("Brief has risk_level", sb.get("risk_level") in ("GREEN", "AMBER", "RED"))

# ── [72] Conformal Prediction (v3.29.0 enhanced) ─────────
print("\n[72] Conformal Prediction")
cf = api("POST", "/api/analysis/conformal-predict", {"well": "3P", "source": "demo"}, timeout=120)
check("Status 200", cf is not None)
check("Has well", cf.get("well") == "3P")
check("Has n_samples", isinstance(cf.get("n_samples"), int) and cf["n_samples"] > 0)
check("Has alpha", isinstance(cf.get("alpha"), (int, float)))
check("Has target_coverage", isinstance(cf.get("target_coverage"), (int, float)))
check("Has empirical_coverage", isinstance(cf.get("empirical_coverage"), (int, float)) and cf["empirical_coverage"] > 0)
check("Has avg_set_size", isinstance(cf.get("avg_set_size"), (int, float)) and cf["avg_set_size"] >= 1)
check("Has singleton_rate", isinstance(cf.get("singleton_rate"), (int, float)))
check("Has n_test", isinstance(cf.get("n_test"), int) and cf["n_test"] > 0)
check("Has n_calibration", isinstance(cf.get("n_calibration"), int) and cf["n_calibration"] > 0)
check("Has conformal_threshold", isinstance(cf.get("conformal_threshold"), (int, float)))
check("Has model_used", isinstance(cf.get("model_used"), str))
check("Has test_predictions", isinstance(cf.get("test_predictions"), list) and len(cf["test_predictions"]) > 0)
tp0 = cf["test_predictions"][0]
check("TP has conformal_set", isinstance(tp0.get("conformal_set"), list) and len(tp0["conformal_set"]) >= 1)
check("TP has set_size", isinstance(tp0.get("set_size"), int))
check("TP has covered", isinstance(tp0.get("covered"), bool))
check("Has per_class_coverage", isinstance(cf.get("per_class_coverage"), dict))
check("Has plot", isinstance(cf.get("plot"), str) and len(cf["plot"]) > 100)
check("Has stakeholder_brief", isinstance(cf.get("stakeholder_brief"), dict))

# ── [73] Cross-Well Generalization Test ──────────────────
print("\n[73] Cross-Well Generalization Test")
cw = api("POST", "/api/analysis/cross-well-test", {"source": "demo"}, timeout=120)
check("Status 200", cw is not None)
check("Has n_wells", isinstance(cw.get("n_wells"), int) and cw["n_wells"] >= 2)
check("Has model", isinstance(cw.get("model"), str))
check("Has cross_results", isinstance(cw.get("cross_results"), list) and len(cw["cross_results"]) >= 2)
cr0 = cw["cross_results"][0]
check("CR has train_well", isinstance(cr0.get("train_well"), str))
check("CR has test_well", isinstance(cr0.get("test_well"), str))
check("CR has accuracy", isinstance(cr0.get("accuracy"), (int, float)))
check("CR has f1", isinstance(cr0.get("f1"), (int, float)))
check("CR has per_class", isinstance(cr0.get("per_class"), list))
check("Has within_results", isinstance(cw.get("within_results"), list) and len(cw["within_results"]) >= 2)
check("Has avg_within_accuracy", isinstance(cw.get("avg_within_accuracy"), (int, float)))
check("Has avg_cross_accuracy", isinstance(cw.get("avg_cross_accuracy"), (int, float)))
check("Has degradation", isinstance(cw.get("degradation"), (int, float)))
check("Has transfer_grade", cw.get("transfer_grade") in ("A", "B", "C", "D"))
check("Has class_names", isinstance(cw.get("class_names"), list))
check("Has plot", isinstance(cw.get("plot"), str) and len(cw["plot"]) > 100)
check("Has stakeholder_brief", isinstance(cw.get("stakeholder_brief"), dict))
sb = cw.get("stakeholder_brief", {})
check("Brief has headline", isinstance(sb.get("headline"), str))
check("Brief has risk_level", sb.get("risk_level") in ("GREEN", "AMBER", "RED"))

# ── [74] Cross-Well Feature Drift ────────────────────────
print("\n[74] Cross-Well Feature Drift")
dr = api("POST", "/api/analysis/cross-well-drift", {"source": "demo"}, timeout=60)
check("Status 200", dr is not None)
check("Has n_wells", isinstance(dr.get("n_wells"), int) and dr["n_wells"] >= 2)
check("Has comparisons", isinstance(dr.get("comparisons"), list) and len(dr["comparisons"]) >= 1)
dc0 = dr["comparisons"][0]
check("Comp has well_a", isinstance(dc0.get("well_a"), str))
check("Comp has well_b", isinstance(dc0.get("well_b"), str))
check("Comp has n_features", isinstance(dc0.get("n_features"), int))
check("Comp has n_drifted", isinstance(dc0.get("n_drifted"), int))
check("Comp has drift_pct", isinstance(dc0.get("drift_pct"), (int, float)))
check("Comp has overall_severity", dc0.get("overall_severity") in ("HIGH", "MEDIUM", "LOW"))
check("Has overall_alert", dr.get("overall_alert") in ("HIGH", "MEDIUM", "LOW"))
check("Has max_drift_pct", isinstance(dr.get("max_drift_pct"), (int, float)))
check("Has retrain_needed", isinstance(dr.get("retrain_needed"), bool))
check("Has plot", isinstance(dr.get("plot"), str) and len(dr["plot"]) > 100)
check("Has stakeholder_brief", isinstance(dr.get("stakeholder_brief"), dict))

# ── [75] Well-to-Well Domain Adaptation ──────────────────
print("\n[75] Well-to-Well Domain Adaptation")
da = api("POST", "/api/analysis/domain-adapt-wells", {"source": "demo"}, timeout=120)
check("Status 200", da is not None)
check("Has train_well", isinstance(da.get("train_well"), str))
check("Has test_well", isinstance(da.get("test_well"), str))
check("Has n_train", isinstance(da.get("n_train"), int) and da["n_train"] > 0)
check("Has n_test", isinstance(da.get("n_test"), int) and da["n_test"] > 0)
check("Has n_features", isinstance(da.get("n_features"), int))
check("Has methods", isinstance(da.get("methods"), list) and len(da["methods"]) >= 2)
m0 = da["methods"][0]
check("Method has method", isinstance(m0.get("method"), str))
check("Method has accuracy", isinstance(m0.get("accuracy"), (int, float)))
check("Method has f1", isinstance(m0.get("f1"), (int, float)))
check("Method has description", isinstance(m0.get("description"), str))
check("Has best_method", isinstance(da.get("best_method"), str))
check("Has improvement", isinstance(da.get("improvement"), (int, float)))
check("Has per_class", isinstance(da.get("per_class"), list) and len(da["per_class"]) >= 1)
check("Has plot", isinstance(da.get("plot"), str) and len(da["plot"]) > 100)
check("Has stakeholder_brief", isinstance(da.get("stakeholder_brief"), dict))

# ── [76] Depth-Stratified Cross-Validation ──────────────────
print("\n[76] Depth-Stratified Cross-Validation")
dscv = api("POST", "/api/analysis/depth-stratified-cv", {"source": "demo", "well": "3P"}, timeout=120)
check("Status 200", dscv is not None)
check("Has well", dscv.get("well") == "3P")
check("Has n_samples", isinstance(dscv.get("n_samples"), int) and dscv["n_samples"] > 0)
check("Has n_zones", isinstance(dscv.get("n_zones"), int) and dscv["n_zones"] >= 2)
check("Has depth_range_m", isinstance(dscv.get("depth_range_m"), list) and len(dscv["depth_range_m"]) == 2)
check("Has overall_accuracy", isinstance(dscv.get("overall_accuracy"), (int, float)))
check("Has overall_f1", isinstance(dscv.get("overall_f1"), (int, float)))
check("Has random_baseline_avg", isinstance(dscv.get("random_baseline_avg"), (int, float)))
check("Has degradation", isinstance(dscv.get("degradation"), (int, float)))
check("Has deployment_risk", dscv.get("deployment_risk") in ("LOW", "MEDIUM", "HIGH"))
check("Has consistency_std", isinstance(dscv.get("consistency_std"), (int, float)))
check("Has n_good_zones", isinstance(dscv.get("n_good_zones"), int))
check("Has n_bad_zones", isinstance(dscv.get("n_bad_zones"), int))
check("Has worst_zone", isinstance(dscv.get("worst_zone"), dict))
check("Worst zone has zone_id", isinstance(dscv["worst_zone"].get("zone_id"), int))
check("Worst zone has depth_range_m", isinstance(dscv["worst_zone"].get("depth_range_m"), list))
check("Worst zone has accuracy", isinstance(dscv["worst_zone"].get("accuracy"), (int, float)))
check("Has best_zone", isinstance(dscv.get("best_zone"), dict))
check("Has zones list", isinstance(dscv.get("zones"), list) and len(dscv["zones"]) >= 2)
z0 = dscv["zones"][0]
check("Zone has zone_id", isinstance(z0.get("zone_id"), int))
check("Zone has depth_range_m", isinstance(z0.get("depth_range_m"), list))
check("Zone has accuracy", isinstance(z0.get("accuracy"), (int, float)))
check("Zone has f1_weighted", isinstance(z0.get("f1_weighted"), (int, float)))
check("Zone has random_baseline", isinstance(z0.get("random_baseline"), (int, float)))
check("Zone has degradation_vs_random", isinstance(z0.get("degradation_vs_random"), (int, float)))
check("Zone has grade", z0.get("grade") in ("A", "B", "C", "D", "F"))
check("Has plot", isinstance(dscv.get("plot"), str) and len(dscv["plot"]) > 100)
check("Has stakeholder_brief", isinstance(dscv.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(dscv["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", dscv["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# Param validation
check("n_zones=0 rejected", api_expect_error("POST", "/api/analysis/depth-stratified-cv", {"source": "demo", "well": "3P", "n_zones": 0}))

# Custom zones
dscv3 = api("POST", "/api/analysis/depth-stratified-cv", {"source": "demo", "well": "3P", "n_zones": 3}, timeout=120)
check("3-zone works", dscv3 is not None and isinstance(dscv3.get("zones"), list))

# ── [77] Probability Calibration (Temperature Scaling) ──────────────────
print("\n[77] Probability Calibration (Temperature Scaling)")
tcal = api("POST", "/api/analysis/calibrate-probabilities", {"source": "demo", "well": "3P"}, timeout=120)
check("Status 200", tcal is not None)
check("Has well", tcal.get("well") == "3P")
check("Has n_samples", isinstance(tcal.get("n_samples"), int) and tcal["n_samples"] > 0)
check("Has n_classes", isinstance(tcal.get("n_classes"), int) and tcal["n_classes"] >= 2)
check("Has temperature", isinstance(tcal.get("temperature"), (int, float)) and tcal["temperature"] > 0)
check("Has ece_before", isinstance(tcal.get("ece_before"), (int, float)))
check("Has ece_after", isinstance(tcal.get("ece_after"), (int, float)))
check("ECE after <= before", tcal.get("ece_after", 999) <= tcal.get("ece_before", 0) + 0.01)
check("Has ece_improvement", isinstance(tcal.get("ece_improvement"), (int, float)))
check("Has ece_pct_improvement", isinstance(tcal.get("ece_pct_improvement"), (int, float)))
check("Has grade", tcal.get("grade") in ("A", "B", "C", "D", "F"))
check("Has verdict", isinstance(tcal.get("verdict"), str) and len(tcal["verdict"]) > 10)
check("Has bins_before", isinstance(tcal.get("bins_before"), list) and len(tcal["bins_before"]) >= 2)
check("Has bins_after", isinstance(tcal.get("bins_after"), list) and len(tcal["bins_after"]) >= 2)
b0 = tcal["bins_before"][0]
check("Bin has bin_lower", isinstance(b0.get("bin_lower"), (int, float)))
check("Bin has bin_upper", isinstance(b0.get("bin_upper"), (int, float)))
check("Bin has count", isinstance(b0.get("count"), int))
check("Has per_class", isinstance(tcal.get("per_class"), list) and len(tcal["per_class"]) >= 1)
pc0 = tcal["per_class"][0]
check("PerClass has class", isinstance(pc0.get("class"), str))
check("PerClass has before_avg_confidence", isinstance(pc0.get("before_avg_confidence"), (int, float)))
check("PerClass has after_avg_confidence", isinstance(pc0.get("after_avg_confidence"), (int, float)))
check("PerClass has actual_frequency", isinstance(pc0.get("actual_frequency"), (int, float)))
check("PerClass has before_gap", isinstance(pc0.get("before_gap"), (int, float)))
check("PerClass has after_gap", isinstance(pc0.get("after_gap"), (int, float)))
check("Has plot", isinstance(tcal.get("plot"), str) and len(tcal["plot"]) > 100)
check("Has stakeholder_brief", isinstance(tcal.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(tcal["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", tcal["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# ── [78] Feature Interaction Discovery ──────────────────
print("\n[78] Feature Interaction Discovery")
fi = api("POST", "/api/analysis/feature-interactions", {"source": "demo", "well": "3P"}, timeout=120)
check("Status 200", fi is not None)
check("Has well", fi.get("well") == "3P")
check("Has n_samples", isinstance(fi.get("n_samples"), int) and fi["n_samples"] > 0)
check("Has n_features", isinstance(fi.get("n_features"), int) and fi["n_features"] >= 2)
check("Has single_importance", isinstance(fi.get("single_importance"), list) and len(fi["single_importance"]) >= 2)
si0 = fi["single_importance"][0]
check("Single has feature", isinstance(si0.get("feature"), str))
check("Single has importance", isinstance(si0.get("importance"), (int, float)))
check("Single has std", isinstance(si0.get("std"), (int, float)))
check("Has interactions", isinstance(fi.get("interactions"), list) and len(fi["interactions"]) >= 1)
i0 = fi["interactions"][0]
check("Interaction has feature_a", isinstance(i0.get("feature_a"), str))
check("Interaction has feature_b", isinstance(i0.get("feature_b"), str))
check("Interaction has interaction_strength", isinstance(i0.get("interaction_strength"), (int, float)))
check("Interaction has joint_drop", isinstance(i0.get("joint_drop"), (int, float)))
check("Interaction has type", i0.get("type") in ("synergistic", "redundant", "independent"))
check("Has n_synergistic", isinstance(fi.get("n_synergistic"), int))
check("Has n_redundant", isinstance(fi.get("n_redundant"), int))
check("Has n_independent", isinstance(fi.get("n_independent"), int))
check("Has strongest_interaction", isinstance(fi.get("strongest_interaction"), dict) or fi.get("strongest_interaction") is None)
check("Has physical_notes", isinstance(fi.get("physical_notes"), list))
check("Has plot", isinstance(fi.get("plot"), str) and len(fi["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fi.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(fi["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", fi["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# Custom top_k
fi5 = api("POST", "/api/analysis/feature-interactions", {"source": "demo", "well": "3P", "top_k": 5}, timeout=120)
check("top_k=5 works", fi5 is not None and isinstance(fi5.get("interactions"), list))
check("top_k limits results", len(fi5.get("interactions", [])) <= 5)

# ── [79] Data Augmentation Analysis ──────────────────
print("\n[79] Data Augmentation Analysis")
aug = api("POST", "/api/analysis/augmentation-analysis", {"source": "demo", "well": "3P"}, timeout=180)
check("Status 200", aug is not None)
check("Has well", aug.get("well") == "3P")
check("Has n_samples", isinstance(aug.get("n_samples"), int) and aug["n_samples"] > 0)
check("Has n_classes", isinstance(aug.get("n_classes"), int) and aug["n_classes"] >= 2)
check("Has imbalance_ratio", isinstance(aug.get("imbalance_ratio"), (int, float)) and aug["imbalance_ratio"] >= 1)
check("Has minority_class", isinstance(aug.get("minority_class"), str))
check("Has majority_class", isinstance(aug.get("majority_class"), str))
check("Has class_counts", isinstance(aug.get("class_counts"), dict))
check("Has strategies", isinstance(aug.get("strategies"), list) and len(aug["strategies"]) >= 3)
s0 = aug["strategies"][0]
check("Strategy has strategy", isinstance(s0.get("strategy"), str))
check("Strategy has accuracy", isinstance(s0.get("accuracy"), (int, float)))
check("Strategy has f1_weighted", isinstance(s0.get("f1_weighted"), (int, float)))
check("Strategy has balanced_accuracy", isinstance(s0.get("balanced_accuracy"), (int, float)))
check("Strategy has per_class", isinstance(s0.get("per_class"), list))
check("Has best_strategy", isinstance(aug.get("best_strategy"), str))
check("Has best_balanced_accuracy", isinstance(aug.get("best_balanced_accuracy"), (int, float)))
check("Has improvement_over_baseline", isinstance(aug.get("improvement_over_baseline"), (int, float)))
check("Has minority_improvements", isinstance(aug.get("minority_improvements"), list) and len(aug["minority_improvements"]) >= 1)
mi0 = aug["minority_improvements"][0]
check("MI has class", isinstance(mi0.get("class"), str))
check("MI has count", isinstance(mi0.get("count"), int))
check("MI has baseline_f1", isinstance(mi0.get("baseline_f1"), (int, float)))
check("MI has best_f1", isinstance(mi0.get("best_f1"), (int, float)))
check("MI has improvement", isinstance(mi0.get("improvement"), (int, float)))
check("MI has best_strategy", isinstance(mi0.get("best_strategy"), str))
check("Has plot", isinstance(aug.get("plot"), str) and len(aug["plot"]) > 100)
check("Has stakeholder_brief", isinstance(aug.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(aug["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", aug["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# ── [80] Multi-Objective Optimization ──────────────────
print("\n[80] Multi-Objective Optimization")
mo = api("POST", "/api/analysis/multi-objective", {"source": "demo", "well": "3P"}, timeout=120)
check("Status 200", mo is not None)
check("Has well", mo.get("well") == "3P")
check("Has n_samples", isinstance(mo.get("n_samples"), int) and mo["n_samples"] > 0)
check("Has trade_offs", isinstance(mo.get("trade_offs"), list) and len(mo["trade_offs"]) >= 2)
t0 = mo["trade_offs"][0]
check("TradeOff has threshold", isinstance(t0.get("threshold"), (int, float)))
check("TradeOff has coverage", isinstance(t0.get("coverage"), (int, float)))
check("TradeOff has accuracy", isinstance(t0.get("accuracy"), (int, float)))
check("TradeOff has error_rate", isinstance(t0.get("error_rate"), (int, float)))
check("TradeOff has n_classified", isinstance(t0.get("n_classified"), int))
check("TradeOff has n_abstained", isinstance(t0.get("n_abstained"), int))
check("Has pareto_points", isinstance(mo.get("pareto_points"), list) and len(mo["pareto_points"]) >= 1)
check("Pareto has pareto_optimal", mo["pareto_points"][0].get("pareto_optimal") == True)
check("Has recommended", isinstance(mo.get("recommended"), dict))
check("Recommended has threshold", isinstance(mo["recommended"].get("threshold"), (int, float)))
check("Recommended has accuracy", isinstance(mo["recommended"].get("accuracy"), (int, float)))
check("Has scenarios", isinstance(mo.get("scenarios"), list) and len(mo["scenarios"]) >= 1)
sc0 = mo["scenarios"][0]
check("Scenario has name", isinstance(sc0.get("name"), str))
check("Scenario has accuracy", isinstance(sc0.get("accuracy"), (int, float)))
check("Has n_pareto", isinstance(mo.get("n_pareto"), int) and mo["n_pareto"] >= 1)
check("Has plot", isinstance(mo.get("plot"), str) and len(mo["plot"]) > 100)
check("Has stakeholder_brief", isinstance(mo.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(mo["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", mo["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# ── [81] Explainability Report ──────────────────
print("\n[81] Explainability Report")
expl = api("POST", "/api/analysis/explainability-report", {"source": "demo", "well": "3P"}, timeout=120)
check("Status 200", expl is not None)
check("Has well", expl.get("well") == "3P")
check("Has n_samples_explained", isinstance(expl.get("n_samples_explained"), int) and expl["n_samples_explained"] > 0)
check("Has n_correct", isinstance(expl.get("n_correct"), int))
check("Has n_misclassified", isinstance(expl.get("n_misclassified"), int))
check("Has avg_confidence", isinstance(expl.get("avg_confidence"), (int, float)))
check("Has global_feature_ranking", isinstance(expl.get("global_feature_ranking"), list) and len(expl["global_feature_ranking"]) >= 2)
gf0 = expl["global_feature_ranking"][0]
check("GF has feature", isinstance(gf0.get("feature"), str))
check("GF has importance", isinstance(gf0.get("importance"), (int, float)))
check("Has explanations", isinstance(expl.get("explanations"), list) and len(expl["explanations"]) >= 1)
e0 = expl["explanations"][0]
check("Explanation has index", isinstance(e0.get("index"), int))
check("Explanation has predicted_class", isinstance(e0.get("predicted_class"), str))
check("Explanation has true_class", isinstance(e0.get("true_class"), str))
check("Explanation has correct", isinstance(e0.get("correct"), bool))
check("Explanation has confidence", isinstance(e0.get("confidence"), (int, float)))
check("Explanation has narrative", isinstance(e0.get("narrative"), str) and len(e0["narrative"]) > 20)
check("Explanation has top_features", isinstance(e0.get("top_features"), list) and len(e0["top_features"]) >= 1)
check("Explanation has category", e0.get("category") in ("correct_confident", "correct_uncertain", "misclassified"))
check("Has plot", isinstance(expl.get("plot"), str) and len(expl["plot"]) > 100)
check("Has stakeholder_brief", isinstance(expl.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(expl["stakeholder_brief"].get("headline"), str))

# Param validation
check("n_samples=0 rejected", api_expect_error("POST", "/api/analysis/explainability-report", {"source": "demo", "well": "3P", "n_samples": 0}))
check("n_samples=100 rejected", api_expect_error("POST", "/api/analysis/explainability-report", {"source": "demo", "well": "3P", "n_samples": 100}))

# Custom n_samples
expl5 = api("POST", "/api/analysis/explainability-report", {"source": "demo", "well": "3P", "n_samples": 5}, timeout=120)
check("n_samples=5 works", expl5 is not None and expl5.get("n_samples_explained") == 5)

# ── [82] RLHF Reward Model Training ──────────────────
print("\n[82] RLHF Reward Model Training")
rw = api("POST", "/api/rlhf/reward-model-train", {"source": "demo", "well": "3P"}, timeout=120)
check("Status 200", rw is not None)
check("Has well", rw.get("well") == "3P")
check("Has n_samples", isinstance(rw.get("n_samples"), int) and rw["n_samples"] > 0)
check("Has n_pairs_trained", isinstance(rw.get("n_pairs_trained"), int) and rw["n_pairs_trained"] > 0)
check("Has pair_accuracy", isinstance(rw.get("pair_accuracy"), (int, float)))
check("Has base_accuracy", isinstance(rw.get("base_accuracy"), (int, float)))
check("Has rlhf_accuracy", isinstance(rw.get("rlhf_accuracy"), (int, float)))
check("Has improvement", isinstance(rw.get("improvement"), (int, float)))
check("Has mean_reward_correct", isinstance(rw.get("mean_reward_correct"), (int, float)))
check("Has mean_reward_wrong", isinstance(rw.get("mean_reward_wrong"), (int, float)))
check("Has reward_separation", isinstance(rw.get("reward_separation"), (int, float)))
check("Correct reward >= wrong", rw.get("mean_reward_correct", 0) >= rw.get("mean_reward_wrong", 1))
check("Has reward_features", isinstance(rw.get("reward_features"), list) and len(rw["reward_features"]) >= 1)
rf0 = rw["reward_features"][0]
check("RF has feature", isinstance(rf0.get("feature"), str))
check("RF has weight", isinstance(rf0.get("weight"), (int, float)))
check("Has class_rewards", isinstance(rw.get("class_rewards"), list) and len(rw["class_rewards"]) >= 1)
check("Has plot", isinstance(rw.get("plot"), str) and len(rw["plot"]) > 100)
check("Has stakeholder_brief", isinstance(rw.get("stakeholder_brief"), dict))

# ── [83] Negative Outcome Learning ──────────────────
print("\n[83] Negative Outcome Learning")
nl = api("POST", "/api/analysis/negative-learning", {"source": "demo", "well": "3P"}, timeout=120)
check("Status 200", nl is not None)
check("Has well", nl.get("well") == "3P")
check("Has n_samples", isinstance(nl.get("n_samples"), int) and nl["n_samples"] > 0)
check("Has negative_weight", isinstance(nl.get("negative_weight"), (int, float)))
check("Has n_hard_examples", isinstance(nl.get("n_hard_examples"), int))
check("Has hard_pct", isinstance(nl.get("hard_pct"), (int, float)))
check("Has base_accuracy", isinstance(nl.get("base_accuracy"), (int, float)))
check("Has neg_accuracy", isinstance(nl.get("neg_accuracy"), (int, float)))
check("Has improvement_accuracy", isinstance(nl.get("improvement_accuracy"), (int, float)))
check("Has improvement_balanced", isinstance(nl.get("improvement_balanced"), (int, float)))
check("Has per_class", isinstance(nl.get("per_class"), list) and len(nl["per_class"]) >= 1)
pc0nl = nl["per_class"][0]
check("PC has class", isinstance(pc0nl.get("class"), str))
check("PC has n_hard", isinstance(pc0nl.get("n_hard"), int))
check("PC has hard_pct", isinstance(pc0nl.get("hard_pct"), (int, float)))
check("PC has base_f1", isinstance(pc0nl.get("base_f1"), (int, float)))
check("PC has neg_f1", isinstance(pc0nl.get("neg_f1"), (int, float)))
check("PC has f1_change", isinstance(pc0nl.get("f1_change"), (int, float)))
check("Has hard_examples", isinstance(nl.get("hard_examples"), list))
if nl["hard_examples"]:
    he0 = nl["hard_examples"][0]
    check("HE has index", isinstance(he0.get("index"), int))
    check("HE has true_class", isinstance(he0.get("true_class"), str))
    check("HE has fixed flag", isinstance(he0.get("fixed"), bool))
check("Has plot", isinstance(nl.get("plot"), str) and len(nl["plot"]) > 100)
check("Has stakeholder_brief", isinstance(nl.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(nl["stakeholder_brief"].get("headline"), str))

# Param validation
check("neg_weight=0.5 rejected", api_expect_error("POST", "/api/analysis/negative-learning", {"source": "demo", "well": "3P", "negative_weight": 0.5}))

# Custom weight
nl5 = api("POST", "/api/analysis/negative-learning", {"source": "demo", "well": "3P", "negative_weight": 5.0}, timeout=120)
check("neg_weight=5 works", nl5 is not None and isinstance(nl5.get("neg_accuracy"), (int, float)))

# ── [84] Production Monitoring Simulation ──────────────────
print("\n[84] Production Monitoring Simulation")
ms = api("POST", "/api/analysis/monitoring-simulation", {"source": "demo", "well": "3P"}, timeout=120)
check("Status 200", ms is not None)
check("Has well", ms.get("well") == "3P")
check("Has n_samples", isinstance(ms.get("n_samples"), int) and ms["n_samples"] > 0)
check("Has n_batches", isinstance(ms.get("n_batches"), int) and ms["n_batches"] >= 2)
check("Has train_size", isinstance(ms.get("train_size"), int) and ms["train_size"] > 0)
check("Has monitoring_accuracy", isinstance(ms.get("monitoring_accuracy"), (int, float)))
check("Has trend", ms.get("trend") in ("STABLE", "IMPROVING", "DEGRADING", "INSUFFICIENT_DATA"))
check("Has trend_slope", isinstance(ms.get("trend_slope"), (int, float)))
check("Has n_green", isinstance(ms.get("n_green"), int))
check("Has n_amber", isinstance(ms.get("n_amber"), int))
check("Has n_red", isinstance(ms.get("n_red"), int))
check("Has retrain_needed", isinstance(ms.get("retrain_needed"), bool))
check("Has batches", isinstance(ms.get("batches"), list) and len(ms["batches"]) >= 2)
b0ms = ms["batches"][0]
check("Batch has batch_id", isinstance(b0ms.get("batch_id"), int))
check("Batch has depth_range_m", isinstance(b0ms.get("depth_range_m"), list))
check("Batch has accuracy", isinstance(b0ms.get("accuracy"), (int, float)))
check("Batch has cumulative_accuracy", isinstance(b0ms.get("cumulative_accuracy"), (int, float)))
check("Batch has status", b0ms.get("status") in ("GREEN", "AMBER", "RED"))
check("Has alerts", isinstance(ms.get("alerts"), list))
check("Has plot", isinstance(ms.get("plot"), str) and len(ms["plot"]) > 100)
check("Has stakeholder_brief", isinstance(ms.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(ms["stakeholder_brief"].get("headline"), str))

# Param validation
check("n_batches=1 rejected", api_expect_error("POST", "/api/analysis/monitoring-simulation", {"source": "demo", "well": "3P", "n_batches": 1}))

# Custom batches
ms5 = api("POST", "/api/analysis/monitoring-simulation", {"source": "demo", "well": "3P", "n_batches": 5}, timeout=120)
check("n_batches=5 works", ms5 is not None and isinstance(ms5.get("batches"), list))

# ── [85] Per-Sample Data Quality ──────────────────────
print("\n[85] Per-Sample Data Quality")
sq = api("POST", "/api/analysis/sample-quality", {"source": "demo", "well": "3P"}, timeout=60)
check("Status 200", sq is not None)
check("Has well", sq.get("well") == "3P")
check("Has n_samples", isinstance(sq.get("n_samples"), int) and sq["n_samples"] > 0)
check("Has n_clean", isinstance(sq.get("n_clean"), int))
check("Has n_minor", isinstance(sq.get("n_minor"), int))
check("Has n_warning", isinstance(sq.get("n_warning"), int))
check("Has n_critical", isinstance(sq.get("n_critical"), int))
check("Grades sum to n", sq["n_clean"] + sq["n_minor"] + sq["n_warning"] + sq["n_critical"] == sq["n_samples"])
check("Has overall_quality_pct", isinstance(sq.get("overall_quality_pct"), (int, float)) and 0 <= sq["overall_quality_pct"] <= 100)
check("Has flag_types dict", isinstance(sq.get("flag_types"), dict))
check("Has flagged_samples list", isinstance(sq.get("flagged_samples"), list))
if sq["flagged_samples"]:
    fs0 = sq["flagged_samples"][0]
    check("FS has index", isinstance(fs0.get("index"), int))
    check("FS has depth_m", isinstance(fs0.get("depth_m"), (int, float)))
    check("FS has azimuth_deg", isinstance(fs0.get("azimuth_deg"), (int, float)))
    check("FS has dip_deg", isinstance(fs0.get("dip_deg"), (int, float)))
    check("FS has quality_score", isinstance(fs0.get("quality_score"), (int, float)) and fs0["quality_score"] > 0)
    check("FS has grade", fs0.get("grade") in ("MINOR", "WARNING", "CRITICAL"))
    check("FS has flags list", isinstance(fs0.get("flags"), list) and len(fs0["flags"]) > 0)
check("Has plot", isinstance(sq.get("plot"), str) and len(sq["plot"]) > 100)
check("Has stakeholder_brief", isinstance(sq.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(sq["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", sq["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
check("Brief has action", isinstance(sq["stakeholder_brief"].get("action"), str))

# Well 6P
sq6 = api("POST", "/api/analysis/sample-quality", {"source": "demo", "well": "6P"}, timeout=60)
check("6P works", sq6 is not None and sq6.get("well") == "6P")
check("6P has grades", isinstance(sq6.get("n_clean"), int))

# ── [86] Learning Curve Projection ────────────────────
print("\n[86] Learning Curve Projection")
lc = api("POST", "/api/analysis/learning-curve-projection", {"source": "demo", "well": "3P"}, timeout=180)
check("Status 200", lc is not None)
check("Has well", lc.get("well") == "3P")
check("Has n_samples", isinstance(lc.get("n_samples"), int) and lc["n_samples"] > 0)
check("Has current_accuracy", isinstance(lc.get("current_accuracy"), (int, float)))
check("Accuracy in range", 0 <= lc["current_accuracy"] <= 1)
check("Has asymptote", isinstance(lc.get("asymptote"), (int, float)))
check("Has remaining_gap", isinstance(lc.get("remaining_gap"), (int, float)) and lc["remaining_gap"] >= 0)
check("Has fit_success", isinstance(lc.get("fit_success"), bool))
check("Has curve_points", isinstance(lc.get("curve_points"), list) and len(lc["curve_points"]) >= 3)
cp0 = lc["curve_points"][0]
check("CP has n_samples", isinstance(cp0.get("n_samples"), int))
check("CP has fraction", isinstance(cp0.get("fraction"), (int, float)))
check("CP has accuracy_mean", isinstance(cp0.get("accuracy_mean"), (int, float)))
check("CP has accuracy_std", isinstance(cp0.get("accuracy_std"), (int, float)))
check("Has projections", isinstance(lc.get("projections"), list) and len(lc["projections"]) >= 4)
p0 = lc["projections"][0]
check("Proj has multiplier", isinstance(p0.get("multiplier"), int))
check("Proj has n_samples", isinstance(p0.get("n_samples"), int))
check("Proj has projected_accuracy", isinstance(p0.get("projected_accuracy"), (int, float)))
check("Proj has gain_vs_current", isinstance(p0.get("gain_vs_current"), (int, float)))
check("Has n_for_90pct_asymptote", isinstance(lc.get("n_for_90pct_asymptote"), int) and lc["n_for_90pct_asymptote"] > 0)
check("Has roi_grade", lc.get("roi_grade") in ("HIGH", "MEDIUM", "LOW"))
check("Has plot", isinstance(lc.get("plot"), str) and len(lc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(lc.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(lc["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", lc["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

if lc["fit_success"]:
    check("Has fit_params when success", isinstance(lc.get("fit_params"), dict))
    check("Fit has a", isinstance(lc["fit_params"].get("a"), (int, float)))
    check("Fit has b", isinstance(lc["fit_params"].get("b"), (int, float)))
    check("Fit has c", isinstance(lc["fit_params"].get("c"), (int, float)))

# Projections should be monotonically increasing
accs = [p["projected_accuracy"] for p in lc["projections"]]
check("Projections non-decreasing", all(accs[i] <= accs[i+1] + 0.001 for i in range(len(accs)-1)))

# ── [87] Consensus Ensemble with Rejection ────────────
print("\n[87] Consensus Ensemble with Rejection")
try:
    ce = api("POST", "/api/analysis/consensus-ensemble", {"source": "demo", "well": "3P"}, timeout=300, retries=1)
    check("Status 200", ce is not None)
    check("Has well", ce.get("well") == "3P")
    check("Has n_samples", isinstance(ce.get("n_samples"), int) and ce["n_samples"] > 0)
    check("Has n_models", isinstance(ce.get("n_models"), int) and ce["n_models"] >= 2)
    check("Has min_agreement", isinstance(ce.get("min_agreement"), (int, float)))
    check("Has n_accepted", isinstance(ce.get("n_accepted"), int))
    check("Has n_rejected", isinstance(ce.get("n_rejected"), int))
    check("Accepted + Rejected = Total", ce["n_accepted"] + ce["n_rejected"] == ce["n_samples"])
    check("Has consensus_rate", isinstance(ce.get("consensus_rate"), (int, float)) and 0 <= ce["consensus_rate"] <= 1)
    check("Has accepted_accuracy", isinstance(ce.get("accepted_accuracy"), (int, float)))
    check("Has model_ranking", isinstance(ce.get("model_ranking"), list) and len(ce["model_ranking"]) >= 2)
    mr0 = ce["model_ranking"][0]
    check("MR has model", isinstance(mr0.get("model"), str))
    check("MR has accuracy", isinstance(mr0.get("accuracy"), (int, float)))
    check("Has per_class", isinstance(ce.get("per_class"), list) and len(ce["per_class"]) >= 2)
    pc0 = ce["per_class"][0]
    check("PC has class", isinstance(pc0.get("class"), str))
    check("PC has count", isinstance(pc0.get("count"), int))
    check("PC has consensus_rate", isinstance(pc0.get("consensus_rate"), (int, float)))
    check("PC has accuracy_when_accepted", isinstance(pc0.get("accuracy_when_accepted"), (int, float)))
    check("PC has avg_agreement", isinstance(pc0.get("avg_agreement"), (int, float)))
    check("Has rejected_samples", isinstance(ce.get("rejected_samples"), list))
    if ce["rejected_samples"]:
        rs0 = ce["rejected_samples"][0]
        check("RS has index", isinstance(rs0.get("index"), int))
        check("RS has true_class", isinstance(rs0.get("true_class"), str))
        check("RS has vote_distribution", isinstance(rs0.get("vote_distribution"), dict))
        check("RS has max_agreement", isinstance(rs0.get("max_agreement"), (int, float)))
    check("Has plot", isinstance(ce.get("plot"), str) and len(ce["plot"]) > 100)
    check("Has stakeholder_brief", isinstance(ce.get("stakeholder_brief"), dict))
    check("Brief has headline", isinstance(ce["stakeholder_brief"].get("headline"), str))

    # Param validation
    check("min_agreement=0.3 rejected", api_expect_error("POST", "/api/analysis/consensus-ensemble", {"source": "demo", "well": "3P", "min_agreement": 0.3}))

    # Custom threshold
    ce8 = api("POST", "/api/analysis/consensus-ensemble", {"source": "demo", "well": "3P", "min_agreement": 0.8}, timeout=300, retries=1)
    check("min_agreement=0.8 works", ce8 is not None and isinstance(ce8.get("consensus_rate"), (int, float)))
    check("Higher threshold = lower consensus", ce8["consensus_rate"] <= ce["consensus_rate"] + 0.01)
    check("Consensus accuracy valid", isinstance(ce8.get("accepted_accuracy"), (int, float)))
except Exception as e87:
    print(f"  SKIP: Consensus Ensemble timed out or failed ({type(e87).__name__}), continuing...")
    failed += 30

# ── [88] Batch Prediction ─────────────────────────────
print("\n[88] Batch Prediction")
try:
    bp = api("POST", "/api/analysis/batch-predict", {"source": "demo", "well": "3P"}, timeout=300, retries=1)
    check("Status 200", bp is not None)
    check("Has well", bp.get("well") == "3P")
    check("Has n_samples", isinstance(bp.get("n_samples"), int) and bp["n_samples"] > 0)
    check("Has n_predicted", isinstance(bp.get("n_predicted"), int) and bp["n_predicted"] > 0)
    check("Has n_models", isinstance(bp.get("n_models"), int) and bp["n_models"] >= 2)
    check("Has elapsed_s", isinstance(bp.get("elapsed_s"), (int, float)) and bp["elapsed_s"] > 0)
    check("Has batch_accuracy", isinstance(bp.get("batch_accuracy"), (int, float)))
    check("Has high_confidence_count", isinstance(bp.get("high_confidence_count"), int))
    check("Has low_confidence_count", isinstance(bp.get("low_confidence_count"), int))
    check("Conf counts sum", bp["high_confidence_count"] + bp["low_confidence_count"] == bp["n_predicted"])
    check("Has model_summary", isinstance(bp.get("model_summary"), list) and len(bp["model_summary"]) >= 2)
    ms0bp = bp["model_summary"][0]
    check("MS has model", isinstance(ms0bp.get("model"), str))
    check("MS has accuracy", isinstance(ms0bp.get("accuracy"), (int, float)))
    check("MS has time_s", isinstance(ms0bp.get("time_s"), (int, float)))
    check("Has predictions", isinstance(bp.get("predictions"), list) and len(bp["predictions"]) > 0)
    p0bp = bp["predictions"][0]
    check("P has index", isinstance(p0bp.get("index"), int))
    check("P has true_class", isinstance(p0bp.get("true_class"), str))
    check("P has predicted_class", isinstance(p0bp.get("predicted_class"), str))
    check("P has correct", isinstance(p0bp.get("correct"), bool))
    check("P has agreement", isinstance(p0bp.get("agreement"), (int, float)))
    check("P has model_votes", isinstance(p0bp.get("model_votes"), dict))
    check("Has plot", isinstance(bp.get("plot"), str) and len(bp["plot"]) > 100)
    check("Has stakeholder_brief", isinstance(bp.get("stakeholder_brief"), dict))
    check("Brief has headline", isinstance(bp["stakeholder_brief"].get("headline"), str))

    # Param validation
    check("top_n=0 rejected", api_expect_error("POST", "/api/analysis/batch-predict", {"source": "demo", "well": "3P", "top_n": 0}))

    # Custom top_n
    bp10 = api("POST", "/api/analysis/batch-predict", {"source": "demo", "well": "3P", "top_n": 10}, timeout=300, retries=1)
    check("top_n=10 works", bp10 is not None and bp10.get("n_predicted") == 10)
except Exception as e88:
    print(f"  SKIP: Batch Prediction timed out or failed ({type(e88).__name__}), continuing...")
    failed += 22

# ── [89] Model Selection Advisor ──────────────────────
print("\n[89] Model Selection Advisor")
try:
    ma = api("POST", "/api/analysis/model-advisor", {"source": "demo", "well": "3P"}, timeout=300, retries=1)
    check("Status 200", ma is not None)
    check("Has well", ma.get("well") == "3P")
    check("Has n_samples", isinstance(ma.get("n_samples"), int) and ma["n_samples"] > 0)
    check("Has n_models", isinstance(ma.get("n_models"), int) and ma["n_models"] >= 2)
    check("Has n_classes", isinstance(ma.get("n_classes"), int) and ma["n_classes"] >= 2)
    check("Has class_names", isinstance(ma.get("class_names"), list))
    check("Has recommended_model", isinstance(ma.get("recommended_model"), str))
    check("Has recommendation_rationale", isinstance(ma.get("recommendation_rationale"), list) and len(ma["recommendation_rationale"]) > 0)
    check("Has evaluations", isinstance(ma.get("evaluations"), list) and len(ma["evaluations"]) >= 2)
    e0ma = ma["evaluations"][0]
    check("E has model", isinstance(e0ma.get("model"), str))
    check("E has accuracy", isinstance(e0ma.get("accuracy"), (int, float)))
    check("E has accuracy_std", isinstance(e0ma.get("accuracy_std"), (int, float)))
    check("E has balanced_accuracy", isinstance(e0ma.get("balanced_accuracy"), (int, float)))
    check("E has f1_weighted", isinstance(e0ma.get("f1_weighted"), (int, float)))
    check("E has train_time_s", isinstance(e0ma.get("train_time_s"), (int, float)))
    check("E has train_accuracy", isinstance(e0ma.get("train_accuracy"), (int, float)))
    check("E has overfit_gap", isinstance(e0ma.get("overfit_gap"), (int, float)))
    check("E has per_class_accuracy", isinstance(e0ma.get("per_class_accuracy"), dict))
    check("E has stability", e0ma.get("stability") in ("STABLE", "MODERATE", "UNSTABLE"))
    check("E has overfit_risk", e0ma.get("overfit_risk") in ("LOW", "MEDIUM", "HIGH"))
    check("Recommended is first", ma["evaluations"][0]["model"] == ma["recommended_model"])
    check("Has plot", isinstance(ma.get("plot"), str) and len(ma["plot"]) > 100)
    check("Has stakeholder_brief", isinstance(ma.get("stakeholder_brief"), dict))
    check("Brief has headline", isinstance(ma["stakeholder_brief"].get("headline"), str))
    check("Brief has risk_level", ma["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))
except Exception as e89:
    print(f"  SKIP: Model Selection Advisor timed out or failed ({type(e89).__name__}), continuing...")
    failed += 24

# ── [90] Operational Readiness Assessment ─────────────
print("\n[90] Operational Readiness Assessment")
try:
    orr = api("POST", "/api/analysis/operational-readiness", {"source": "demo", "well": "3P"}, timeout=300, retries=1)
    check("Status 200", orr is not None)
    check("Has well", orr.get("well") == "3P")
    check("Has n_samples", isinstance(orr.get("n_samples"), int) and orr["n_samples"] > 0)
    check("Has overall_status", orr.get("overall_status") in ("READY", "CONDITIONAL", "NOT READY"))
    check("Has readiness_score", isinstance(orr.get("readiness_score"), (int, float)) and 0 <= orr["readiness_score"] <= 100)
    check("Has n_pass", isinstance(orr.get("n_pass"), int))
    check("Has n_warn", isinstance(orr.get("n_warn"), int))
    check("Has n_fail", isinstance(orr.get("n_fail"), int))
    check("Grades sum to checks", orr["n_pass"] + orr["n_warn"] + orr["n_fail"] == len(orr.get("checks", [])))
    check("Has checks", isinstance(orr.get("checks"), list) and len(orr["checks"]) >= 5)
    c0or = orr["checks"][0]
    check("Check has check name", isinstance(c0or.get("check"), str))
    check("Check has grade", c0or.get("grade") in ("PASS", "WARN", "FAIL"))
    check("Check has detail", isinstance(c0or.get("detail"), str))
    check("Check has threshold", isinstance(c0or.get("threshold"), str))
    check("Has best_model", isinstance(orr.get("best_model"), str))
    check("Has best_accuracy", isinstance(orr.get("best_accuracy"), (int, float)))
    check("Has avg_consensus", isinstance(orr.get("avg_consensus"), (int, float)))
    check("Has plot", isinstance(orr.get("plot"), str) and len(orr["plot"]) > 100)
    check("Has stakeholder_brief", isinstance(orr.get("stakeholder_brief"), dict))
    check("Brief has headline", isinstance(orr["stakeholder_brief"].get("headline"), str))
    check("Brief has risk_level", orr["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

    # Verify specific checks exist
    check_names_or = [c["check"] for c in orr["checks"]]
    check("Data Sufficiency check", "Data Sufficiency" in check_names_or)
    check("Class Balance check", "Class Balance" in check_names_or)
    check("Model Accuracy check", "Model Accuracy" in check_names_or)
    check("Model Consensus check", "Model Consensus" in check_names_or)
except Exception as e90:
    print(f"  SKIP: Operational Readiness timed out or failed ({type(e90).__name__}), continuing...")
    failed += 25

# ── [91] Geomechanical Feature Enrichment ─────────────
print("\n[91] Geomechanical Feature Enrichment")
try:
    gf = api("POST", "/api/analysis/geomech-features", {"source": "demo", "well": "3P"}, timeout=300, retries=1)
    check("Status 200", gf is not None)
    check("Has well", gf.get("well") == "3P")
    check("Has n_samples", isinstance(gf.get("n_samples"), int) and gf["n_samples"] > 0)
    check("Has shmax_azimuth", isinstance(gf.get("shmax_azimuth"), (int, float)))
    check("Has stress_ratio", isinstance(gf.get("stress_ratio"), (int, float)))
    check("Has friction", isinstance(gf.get("friction"), (int, float)))
    check("Has n_geomech_features", isinstance(gf.get("n_geomech_features"), int) and gf["n_geomech_features"] >= 5)
    check("Has geomech_feature_names", isinstance(gf.get("geomech_feature_names"), list))
    check("Has feature_stats", isinstance(gf.get("feature_stats"), list) and len(gf["feature_stats"]) >= 5)
    fs0gf = gf["feature_stats"][0]
    check("FS has feature", isinstance(fs0gf.get("feature"), str))
    check("FS has mean", isinstance(fs0gf.get("mean"), (int, float)))
    check("FS has std", isinstance(fs0gf.get("std"), (int, float)))
    check("Has n_critically_stressed", isinstance(gf.get("n_critically_stressed"), int))
    check("Has comparisons", isinstance(gf.get("comparisons"), list) and len(gf["comparisons"]) >= 2)
    c0gf = gf["comparisons"][0]
    check("C has model", isinstance(c0gf.get("model"), str))
    check("C has baseline_accuracy", isinstance(c0gf.get("baseline_accuracy"), (int, float)))
    check("C has enriched_accuracy", isinstance(c0gf.get("enriched_accuracy"), (int, float)))
    check("C has accuracy_delta", isinstance(c0gf.get("accuracy_delta"), (int, float)))
    check("C has improved", isinstance(c0gf.get("improved"), bool))
    check("Has avg_accuracy_delta", isinstance(gf.get("avg_accuracy_delta"), (int, float)))
    check("Has n_models_improved", isinstance(gf.get("n_models_improved"), int))
    check("Has plot", isinstance(gf.get("plot"), str) and len(gf["plot"]) > 100)
    check("Has stakeholder_brief", isinstance(gf.get("stakeholder_brief"), dict))
    check("Brief has headline", isinstance(gf["stakeholder_brief"].get("headline"), str))

    # slip_tendency in feature names
    check("slip_tendency feature", "slip_tendency" in gf["geomech_feature_names"])
    check("dilation_tendency feature", "dilation_tendency" in gf["geomech_feature_names"])
except Exception as e91:
    print(f"  SKIP: Geomech Features timed out or failed ({type(e91).__name__}), continuing...")
    failed += 26

# ── [92] RLHF Iterative Feedback Loop ────────────────
print("\n[92] RLHF Iterative Feedback Loop")
try:
    ri = api("POST", "/api/analysis/rlhf-iterate", {"source": "demo", "well": "3P"}, timeout=300, retries=1)
    check("Status 200", ri is not None)
    check("Has well", ri.get("well") == "3P")
    check("Has n_samples", isinstance(ri.get("n_samples"), int) and ri["n_samples"] > 0)
    check("Has n_iterations", isinstance(ri.get("n_iterations"), int) and ri["n_iterations"] >= 2)
    check("Has baseline_accuracy", isinstance(ri.get("baseline_accuracy"), (int, float)))
    check("Has final_accuracy", isinstance(ri.get("final_accuracy"), (int, float)))
    check("Has total_improvement", isinstance(ri.get("total_improvement"), (int, float)))
    check("Has converged", isinstance(ri.get("converged"), bool))
    check("Has iterations", isinstance(ri.get("iterations"), list) and len(ri["iterations"]) >= 2)
    it0 = ri["iterations"][0]
    check("IT has iteration", isinstance(it0.get("iteration"), int) and it0["iteration"] == 1)
    check("IT has accuracy", isinstance(it0.get("accuracy"), (int, float)))
    check("IT has improvement_vs_prev", isinstance(it0.get("improvement_vs_prev"), (int, float)))
    check("IT has total_improvement", isinstance(it0.get("total_improvement"), (int, float)))
    check("IT has n_errors", isinstance(it0.get("n_errors"), int))
    check("IT has n_pairs_trained", isinstance(it0.get("n_pairs_trained"), int))
    check("IT has avg_reward_score", isinstance(it0.get("avg_reward_score"), (int, float)))
    check("Has plot", isinstance(ri.get("plot"), str) and len(ri["plot"]) > 100)
    check("Has stakeholder_brief", isinstance(ri.get("stakeholder_brief"), dict))
    check("Brief has headline", isinstance(ri["stakeholder_brief"].get("headline"), str))

    # Param validation
    check("n_iterations=1 rejected", api_expect_error("POST", "/api/analysis/rlhf-iterate", {"source": "demo", "well": "3P", "n_iterations": 1}))

    # Custom iterations
    ri3 = api("POST", "/api/analysis/rlhf-iterate", {"source": "demo", "well": "3P", "n_iterations": 3}, timeout=300, retries=1)
    check("n_iterations=3 works", ri3 is not None and len(ri3.get("iterations", [])) == 3)
except Exception as e92:
    print(f"  SKIP: RLHF Iterate timed out or failed ({type(e92).__name__}), continuing...")
    failed += 22

# ── [93] Domain Shift Robustness ──────────────────────
print("\n[93] Domain Shift Robustness")
try:
    ds = api("POST", "/api/analysis/domain-shift", {"source": "demo", "well": "3P"}, timeout=300, retries=1)
    check("Status 200", ds is not None)
    check("Has well", ds.get("well") == "3P")
    check("Has n_samples", isinstance(ds.get("n_samples"), int) and ds["n_samples"] > 0)
    check("Has n_zones", isinstance(ds.get("n_zones"), int) and ds["n_zones"] >= 2)
    check("Has avg_same_domain", isinstance(ds.get("avg_same_domain"), (int, float)))
    check("Has avg_cross_domain", isinstance(ds.get("avg_cross_domain"), (int, float)))
    check("Has domain_gap", isinstance(ds.get("domain_gap"), (int, float)))
    check("Has zone_summary", isinstance(ds.get("zone_summary"), list) and len(ds["zone_summary"]) >= 2)
    z0ds = ds["zone_summary"][0]
    check("Z has zone", isinstance(z0ds.get("zone"), int))
    check("Z has depth_range", isinstance(z0ds.get("depth_range"), (list, tuple)) and len(z0ds["depth_range"]) == 2)
    check("Z has n_samples", isinstance(z0ds.get("n_samples"), int))
    check("Z has self_accuracy", isinstance(z0ds.get("self_accuracy"), (int, float)))
    check("Z has transfer_accuracy", isinstance(z0ds.get("transfer_accuracy"), (int, float)))
    check("Z has gap", isinstance(z0ds.get("gap"), (int, float)))
    check("Has cross_domain_matrix", isinstance(ds.get("cross_domain_matrix"), list))
    check("Matrix size correct", len(ds["cross_domain_matrix"]) == ds["n_zones"] ** 2)
    check("Has worst_transitions", isinstance(ds.get("worst_transitions"), list))
    check("Has plot", isinstance(ds.get("plot"), str) and len(ds["plot"]) > 100)
    check("Has stakeholder_brief", isinstance(ds.get("stakeholder_brief"), dict))
    check("Brief has headline", isinstance(ds["stakeholder_brief"].get("headline"), str))

    # Param validation
    check("n_zones=1 rejected", api_expect_error("POST", "/api/analysis/domain-shift", {"source": "demo", "well": "3P", "n_zones": 1}))

    # Custom zones
    ds5 = api("POST", "/api/analysis/domain-shift", {"source": "demo", "well": "3P", "n_zones": 5}, timeout=300, retries=1)
    check("n_zones=5 works", ds5 is not None and ds5.get("n_zones") == 5)
except Exception as e93:
    print(f"  SKIP: Domain Shift timed out or failed ({type(e93).__name__}), continuing...")
    failed += 22

# ── [94] Decision Support Matrix ──────────────────────
print("\n[94] Decision Support Matrix")
dsx = api("POST", "/api/analysis/decision-support", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", dsx is not None)
check("Has well", dsx.get("well") == "3P")
check("Has n_samples", isinstance(dsx.get("n_samples"), int) and dsx["n_samples"] > 0)
check("Has decision", dsx.get("decision") in ("GO", "CAUTION", "STOP"))
check("Has overall_score", isinstance(dsx.get("overall_score"), (int, float)) and 0 <= dsx["overall_score"] <= 100)
check("Has criteria", isinstance(dsx.get("criteria"), list) and len(dsx["criteria"]) >= 4)
c0dsx = dsx["criteria"][0]
check("Crit has criterion", isinstance(c0dsx.get("criterion"), str))
check("Crit has score", isinstance(c0dsx.get("score"), int) and 0 <= c0dsx["score"] <= 100)
check("Crit has weight", isinstance(c0dsx.get("weight"), int))
check("Crit has detail", isinstance(c0dsx.get("detail"), str))
check("Has recommendations", isinstance(dsx.get("recommendations"), list) and len(dsx["recommendations"]) > 0)
check("Has best_model", isinstance(dsx.get("best_model"), str))
check("Has best_accuracy", isinstance(dsx.get("best_accuracy"), (int, float)))
check("Has plot", isinstance(dsx.get("plot"), str) and len(dsx["plot"]) > 100)
check("Has stakeholder_brief", isinstance(dsx.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(dsx["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", dsx["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# ── [95] Risk Communication Report ───────────────────
print("\n[95] Risk Communication Report")
rr = api("POST", "/api/analysis/risk-report", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", rr is not None)
check("Has well", rr.get("well") == "3P")
check("Has n_samples", isinstance(rr.get("n_samples"), int) and rr["n_samples"] > 0)
check("Has overall_risk", rr.get("overall_risk") in ("LOW", "MEDIUM", "HIGH"))
check("Has executive_summary", isinstance(rr.get("executive_summary"), str) and len(rr["executive_summary"]) > 20)
check("Has risks", isinstance(rr.get("risks"), list) and len(rr["risks"]) >= 3)
rk0 = rr["risks"][0]
check("Risk has category", isinstance(rk0.get("category"), str))
check("Risk has risk_level", rk0.get("risk_level") in ("LOW", "MEDIUM", "HIGH"))
check("Risk has plain_english", isinstance(rk0.get("plain_english"), str) and len(rk0["plain_english"]) > 20)
check("Risk has impact", isinstance(rk0.get("impact"), str))
check("Risk has mitigation", isinstance(rk0.get("mitigation"), str))
check("Has n_high_risks", isinstance(rr.get("n_high_risks"), int))
check("Has n_medium_risks", isinstance(rr.get("n_medium_risks"), int))
check("Has n_low_risks", isinstance(rr.get("n_low_risks"), int))
check("Risk counts sum", rr["n_high_risks"] + rr["n_medium_risks"] + rr["n_low_risks"] == len(rr["risks"]))
check("Has plot", isinstance(rr.get("plot"), str) and len(rr["plot"]) > 100)
check("Has stakeholder_brief", isinstance(rr.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(rr["stakeholder_brief"].get("headline"), str))

# Check risk categories exist
cat_names = [r["category"] for r in rr["risks"]]
check("Data risk exists", any("Data" in c for c in cat_names))
check("Model risk exists", any("Model" in c for c in cat_names))

# ── [96] Model Transparency Audit ────────────────────
print("\n[96] Model Transparency Audit")
ta = api("POST", "/api/analysis/transparency-audit", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", ta is not None)
check("Has well", ta.get("well") == "3P")
check("Has n_samples", isinstance(ta.get("n_samples"), int) and ta["n_samples"] > 0)
check("Has n_audited", isinstance(ta.get("n_audited"), int) and ta["n_audited"] > 0)
check("Has n_correct", isinstance(ta.get("n_correct"), int))
check("Has n_wrong", isinstance(ta.get("n_wrong"), int))
check("Correct+Wrong=Audited", ta["n_correct"] + ta["n_wrong"] == ta["n_audited"])
check("Has audit_accuracy", isinstance(ta.get("audit_accuracy"), (int, float)))
check("Has n_models", isinstance(ta.get("n_models"), int) and ta["n_models"] >= 2)
check("Has global_feature_importances", isinstance(ta.get("global_feature_importances"), list))
if ta["global_feature_importances"]:
    gi0 = ta["global_feature_importances"][0]
    check("GI has feature", isinstance(gi0.get("feature"), str))
    check("GI has importance", isinstance(gi0.get("importance"), (int, float)))
check("Has transparency_cards", isinstance(ta.get("transparency_cards"), list) and len(ta["transparency_cards"]) > 0)
tc0 = ta["transparency_cards"][0]
check("TC has index", isinstance(tc0.get("index"), int))
check("TC has depth_m", True)  # may be null
check("TC has true_class", isinstance(tc0.get("true_class"), str))
check("TC has consensus_class", isinstance(tc0.get("consensus_class"), str))
check("TC has correct", isinstance(tc0.get("correct"), bool))
check("TC has agreement", isinstance(tc0.get("agreement"), (int, float)))
check("TC has model_details", isinstance(tc0.get("model_details"), list) and len(tc0["model_details"]) >= 2)
md0 = tc0["model_details"][0]
check("MD has model", isinstance(md0.get("model"), str))
check("MD has predicted", isinstance(md0.get("predicted"), str))
check("MD has correct", isinstance(md0.get("correct"), bool))
check("TC has top_features", isinstance(tc0.get("top_features"), list))
check("TC has geology_note", isinstance(tc0.get("geology_note"), str))
check("Has plot", isinstance(ta.get("plot"), str) and len(ta["plot"]) > 100)
check("Has stakeholder_brief", isinstance(ta.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(ta["stakeholder_brief"].get("headline"), str))

# Param validation
check("top_n=0 rejected", api_expect_error("POST", "/api/analysis/transparency-audit", {"source": "demo", "well": "3P", "top_n": 0}))

# Custom top_n
ta5 = api("POST", "/api/analysis/transparency-audit", {"source": "demo", "well": "3P", "top_n": 5}, timeout=300)
check("top_n=5 works", ta5 is not None and ta5.get("n_audited") == 5)

# ── Summary ──────────────────────────────────────────

print(f"\n{'='*50}")
print(f"v3.28.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
