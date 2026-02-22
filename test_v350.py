"""Test suite for GeoStress AI v3.5.0.

Tests: input validation, field calibration, error boundaries.
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
check("Version is 3.4+",
      d.get("app_version", "").startswith("3.4") or
      d.get("app_version", "").startswith("3.5"))


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


# ── Summary ──────────────────────────────────────────

print(f"\n{'='*50}")
print(f"v3.5.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
