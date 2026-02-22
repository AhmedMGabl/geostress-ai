"""Playwright tests for GeoStress AI v3.3.0-3.3.1 — Production MLOps + RLHF."""
from playwright.sync_api import sync_playwright
import json

BASE = "http://localhost:8099"

def test_api(method, path, body=None, expect_keys=None, label=""):
    """Test an API endpoint and return the response."""
    import urllib.request
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            d = json.loads(resp.read())
            if expect_keys:
                for k in expect_keys:
                    assert k in d, f"{label}: missing key '{k}'"
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

    print("\n=== v3.3.0 API Tests ===")

    # 1. System health
    d = test_api("GET", "/api/system/health", expect_keys=["status", "health_score"], label="System health")
    if d and d.get("app_version") == "3.3.0": passed += 1
    else: failed += 1

    # 2. Drift baseline
    d = test_api("POST", "/api/analysis/drift-detection", {"well": "6P"}, expect_keys=["status"], label="Drift baseline 6P")
    if d: passed += 1
    else: failed += 1

    # 3. Drift compare (same data = OK)
    d = test_api("POST", "/api/analysis/drift-detection", {"well": "6P"}, label="Drift compare 6P")
    if d and d.get("status") == "OK": passed += 1
    else: failed += 1

    # 4. Model register
    d = test_api("POST", "/api/models/register", {"well": "3P"}, expect_keys=["version"], label="Register model")
    if d and d.get("version"): passed += 1
    else: failed += 1

    # 5. Model registry
    d = test_api("GET", "/api/models/registry", expect_keys=["versions"], label="Model registry")
    if d and d.get("count", 0) > 0: passed += 1
    else: failed += 1

    # 6. Field stress model
    d = test_api("POST", "/api/analysis/field-stress-model", {"depth_m": 3000, "pp_mpa": 30},
                 expect_keys=["field_shmax_deg", "consistency"], label="Field stress model")
    if d and d.get("field_shmax_deg"): passed += 1
    else: failed += 1

    # 7. Record failure
    d = test_api("POST", "/api/feedback/failure-case",
                 {"failure_type": "wrong_prediction", "well": "3P", "predicted": "Vuggy", "actual": "Continuous"},
                 expect_keys=["case_id"], label="Record failure")
    if d: passed += 1
    else: failed += 1

    # 8. Failure analysis
    d = test_api("GET", "/api/feedback/failure-analysis?well=3P", expect_keys=["n_cases"], label="Failure analysis")
    if d and d.get("n_cases", 0) > 0: passed += 1
    else: failed += 1

    print("\n=== v3.3.1 API Tests ===")

    # 9. RLHF review queue
    d = test_api("POST", "/api/rlhf/review-queue", {"well": "3P", "n_samples": 5},
                 expect_keys=["queue", "n_total_samples"], label="RLHF review queue")
    if d and len(d.get("queue", [])) > 0: passed += 1
    else: failed += 1

    # 10. RLHF accept
    d = test_api("POST", "/api/rlhf/accept-reject",
                 {"well": "3P", "sample_index": 0, "verdict": "accept", "predicted_type": "Vuggy"},
                 expect_keys=["review_id"], label="RLHF accept")
    if d: passed += 1
    else: failed += 1

    # 11. RLHF reject
    d = test_api("POST", "/api/rlhf/accept-reject",
                 {"well": "3P", "sample_index": 1, "verdict": "reject", "predicted_type": "Continuous", "true_type": "Boundary"},
                 expect_keys=["review_id"], label="RLHF reject")
    if d: passed += 1
    else: failed += 1

    # 12. RLHF impact
    d = test_api("GET", "/api/rlhf/impact?well=3P", expect_keys=["total_reviews", "acceptance_rate"], label="RLHF impact")
    if d and d.get("total_reviews", 0) > 0: passed += 1
    else: failed += 1

    # 13. Batch analysis
    d = test_api("POST", "/api/batch/analyze-all", {"depth_m": 3000, "pp_mpa": 30},
                 expect_keys=["wells", "field_summary"], label="Batch analysis")
    if d and d.get("n_wells", 0) >= 2: passed += 1
    else: failed += 1

    # 14. Batch accuracy check
    if d:
        wells = d.get("wells", [])
        w3p = next((w for w in wells if w.get("well") == "3P"), {})
        if w3p.get("accuracy", 0) > 0.5:
            print(f"  PASS: Batch accuracy 3P = {w3p['accuracy']}")
            passed += 1
        else:
            print(f"  FAIL: Batch accuracy 3P = {w3p.get('accuracy')}")
            failed += 1
    else:
        failed += 1

    print("\n=== UI Tests ===")

    # 15. Page loads
    page.goto(BASE)
    page.wait_for_load_state("networkidle")
    if "v3.3" in page.text_content("body"):
        print("  PASS: v3.3 displayed")
        passed += 1
    else:
        print("  FAIL: v3.3 not displayed")
        failed += 1

    # 16. MLOps tab works
    mlops_link = page.query_selector('[data-tab="mlops"]')
    if mlops_link:
        mlops_link.click()
        page.wait_for_timeout(500)
        if page.query_selector("#tab-mlops"):
            print("  PASS: MLOps tab exists")
            passed += 1
        else:
            print("  FAIL: MLOps tab missing")
            failed += 1
    else:
        print("  FAIL: MLOps nav link missing")
        failed += 1

    # 17. RLHF section present
    if page.query_selector("#rlhf-queue-result"):
        print("  PASS: RLHF section present")
        passed += 1
    else:
        print("  FAIL: RLHF section missing")
        failed += 1

    # 18. Batch section present
    if page.query_selector("#batch-result"):
        print("  PASS: Batch section present")
        passed += 1
    else:
        print("  FAIL: Batch section missing")
        failed += 1

    browser.close()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'='*50}")
    if failed > 0:
        exit(1)
