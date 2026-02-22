"""Playwright tests for GeoStress AI v3.3.0 — Production MLOps."""
from playwright.sync_api import sync_playwright
import json, time

BASE = "http://localhost:8099"

def test_api(method, path, body=None, expect_keys=None, label=""):
    """Test an API endpoint and return the response."""
    import urllib.request
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            d = json.loads(resp.read())
            if expect_keys:
                for k in expect_keys:
                    assert k in d, f"{label}: missing key '{k}' in response"
            print(f"  PASS: {label}")
            return d
    except Exception as e:
        print(f"  FAIL: {label} — {e}")
        return None


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    passed = 0
    failed = 0

    # ── API Tests ──
    print("\n=== API Tests ===")

    # 1. System health
    d = test_api("GET", "/api/system/health", expect_keys=["status", "health_score", "app_version"], label="System health")
    if d and d.get("app_version") == "3.3.0":
        passed += 1
    else:
        failed += 1

    # 2. Drift detection (creates baseline)
    d = test_api("POST", "/api/analysis/drift-detection", {"well": "6P"}, expect_keys=["status"], label="Drift baseline 6P")
    if d:
        passed += 1
    else:
        failed += 1

    # 3. Drift detection (compare)
    d = test_api("POST", "/api/analysis/drift-detection", {"well": "6P"}, expect_keys=["status", "avg_psi"], label="Drift compare 6P")
    if d and d.get("status") == "OK":
        passed += 1
    else:
        failed += 1

    # 4. Drift reset
    d = test_api("POST", "/api/analysis/drift-reset", {"well": "6P"}, expect_keys=["status"], label="Drift reset 6P")
    if d:
        passed += 1
    else:
        failed += 1

    # 5. Model registry (empty)
    d = test_api("GET", "/api/models/registry", expect_keys=["versions", "count"], label="Model registry")
    if d:
        passed += 1
    else:
        failed += 1

    # 6. Register model
    d = test_api("POST", "/api/models/register", {"well": "3P"}, expect_keys=["version"], label="Register model")
    if d and d.get("version"):
        passed += 1
    else:
        failed += 1

    # 7. Compare versions (need 2 versions first)
    test_api("POST", "/api/models/register", {"well": "3P", "notes": "Second registration"}, label="Register model v2")
    d = test_api("POST", "/api/models/compare-versions", {"well": "3P"}, expect_keys=["verdict"], label="Compare versions")
    if d and d.get("verdict"):
        passed += 1
    else:
        failed += 1

    # 8. Field stress model
    d = test_api("POST", "/api/analysis/field-stress-model", {"depth_m": 3000, "pp_mpa": 30},
                 expect_keys=["field_shmax_deg", "consistency", "well_results"], label="Field stress model")
    if d and d.get("field_shmax_deg") is not None:
        passed += 1
    else:
        failed += 1

    # 9. Record failure case
    d = test_api("POST", "/api/feedback/failure-case",
                 {"failure_type": "wrong_prediction", "well": "3P", "predicted": "Vuggy", "actual": "Continuous", "depth_m": 2950},
                 expect_keys=["case_id"], label="Record failure case")
    if d and d.get("case_id"):
        passed += 1
    else:
        failed += 1

    # 10. Failure analysis
    d = test_api("GET", "/api/feedback/failure-analysis?well=3P", expect_keys=["n_cases"], label="Failure analysis")
    if d and d.get("n_cases", 0) > 0:
        passed += 1
    else:
        failed += 1

    # ── UI Tests ──
    print("\n=== UI Tests ===")

    # 11. Page loads with v3.3 version
    page.goto(BASE)
    page.wait_for_load_state("networkidle")
    version_text = page.text_content("body")
    if "v3.3" in version_text:
        print("  PASS: Version 3.3 displayed")
        passed += 1
    else:
        print("  FAIL: Version 3.3 not found in page")
        failed += 1

    # 12. MLOps tab exists
    mlops_link = page.query_selector('[data-tab="mlops"]')
    if mlops_link:
        print("  PASS: MLOps tab present")
        passed += 1
    else:
        print("  FAIL: MLOps tab missing")
        failed += 1

    # 13. Click MLOps tab
    if mlops_link:
        mlops_link.click()
        page.wait_for_timeout(500)
        mlops_content = page.query_selector("#tab-mlops")
        if mlops_content and mlops_content.is_visible():
            print("  PASS: MLOps tab content visible")
            passed += 1
        else:
            print("  FAIL: MLOps tab content not visible")
            failed += 1
    else:
        failed += 1

    # 14. System health button exists
    health_btn = page.query_selector('button:has-text("Refresh")')
    if health_btn:
        print("  PASS: System health refresh button present")
        passed += 1
    else:
        print("  FAIL: System health refresh button missing")
        failed += 1

    # 15. Drift detection button exists
    drift_btn = page.query_selector('button:has-text("Check Drift")')
    if drift_btn:
        print("  PASS: Drift detection button present")
        passed += 1
    else:
        print("  FAIL: Drift detection button missing")
        failed += 1

    # 16. Failure report form exists
    failure_type = page.query_selector('#failure-type')
    if failure_type:
        print("  PASS: Failure report form present")
        passed += 1
    else:
        print("  FAIL: Failure report form missing")
        failed += 1

    browser.close()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'='*50}")

    if failed > 0:
        exit(1)
