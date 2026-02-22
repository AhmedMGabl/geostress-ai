"""Comprehensive test suite for GeoStress AI v3.4.0.

Tests performance, accuracy, MLOps, RLHF, batch, stakeholder features.
"""
from playwright.sync_api import sync_playwright
import json
import time
import urllib.request

BASE = "http://localhost:8099"


def api(method, path, body=None, timeout=180):
    """Call API and return parsed JSON."""
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def test(label, condition, detail=""):
    """Record a test result."""
    if condition:
        print(f"  PASS: {label}" + (f" ({detail})" if detail else ""))
        return True
    else:
        print(f"  FAIL: {label}" + (f" ({detail})" if detail else ""))
        return False


passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if test(label, condition, detail):
        passed += 1
    else:
        failed += 1


# ── Core API Tests ──────────────────────────────────

print("\n=== Core API Tests ===")

d = api("GET", "/api/system/health")
check("Health endpoint", d.get("status") == "HEALTHY")
check("Version is 3.x", d.get("app_version", "").startswith("3."))
check("Health score > 0", d.get("health_score", 0) > 0)

d = api("GET", "/api/data/summary")
check("Data summary loads", d.get("total_fractures", 0) > 0)
check("Multiple wells", d.get("wells_count", len(d.get("wells", []))) >= 2)

# ── Inversion Tests ──────────────────────────────────

print("\n=== Inversion Tests ===")

t0 = time.time()
d = api("POST", "/api/analysis/inversion", {"regime": "auto"})
elapsed = time.time() - t0
check("Inversion returns SHmax", d.get("shmax_azimuth_deg") is not None)
check("Inversion returns regime", d.get("auto_regime") is not None)
check("Inversion < 30s", elapsed < 30, f"{elapsed:.1f}s")

# ── Classification Tests ──────────────────────────────

print("\n=== Classification Tests ===")

d = api("POST", "/api/analysis/classify", {"classifier": "xgboost", "enhanced": True})
check("Classify returns accuracy", d.get("cv_mean_accuracy", 0) > 0)
check("Accuracy >= 0.5", d.get("cv_mean_accuracy", 0) >= 0.5, f"{d.get('cv_mean_accuracy', 0):.3f}")
check("Feature importances present", len(d.get("feature_importances", {})) > 0)
check("Confusion matrix present", len(d.get("confusion_matrix", [])) > 0)

# ── MLOps Tests ──────────────────────────────────────

print("\n=== MLOps Tests ===")

d = api("POST", "/api/analysis/drift-detection", {"well": "3P"})
check("Drift detection works", d.get("status") is not None)

d = api("POST", "/api/models/register", {"well": "3P"})
check("Model register works", d.get("version") is not None)

d = api("GET", "/api/models/registry")
check("Model registry has entries", d.get("count", 0) > 0)

d = api("POST", "/api/analysis/field-stress-model", {"depth_m": 3000, "pp_mpa": 30})
check("Field stress model returns SHmax", d.get("field_shmax_deg") is not None)
check("Field consistency computed", d.get("consistency") is not None)

# ── Failure Case Learning ────────────────────────────

print("\n=== Failure Case Learning ===")

d = api("POST", "/api/feedback/failure-case",
        {"failure_type": "wrong_prediction", "well": "3P",
         "predicted": "Vuggy", "actual": "Continuous"})
check("Record failure case", d.get("case_id") is not None)

d = api("GET", "/api/feedback/failure-analysis?well=3P")
check("Failure analysis works", d.get("n_cases", 0) > 0)

# ── RLHF Pipeline ───────────────────────────────────

print("\n=== RLHF Pipeline ===")

d = api("POST", "/api/rlhf/review-queue", {"well": "3P", "n_samples": 5})
check("RLHF queue returns samples", len(d.get("queue", [])) > 0)
check("Samples have priority scores", all("priority_score" in s for s in d.get("queue", [])[:3]))

d = api("POST", "/api/rlhf/accept-reject",
        {"well": "3P", "sample_index": 0, "verdict": "accept", "predicted_type": "Vuggy"})
check("RLHF accept works", d.get("review_id") is not None)

d = api("GET", "/api/rlhf/impact?well=3P")
check("RLHF impact computed", d.get("total_reviews", 0) > 0)

# ── Batch Processing ─────────────────────────────────

print("\n=== Batch Processing ===")

t0 = time.time()
d = api("POST", "/api/batch/analyze-all", {"depth_m": 3000, "pp_mpa": 30})
elapsed = time.time() - t0
check("Batch returns wells", d.get("n_wells", 0) >= 2)
check("Batch has field summary", d.get("field_summary") is not None)

wells = d.get("wells", [])
w3p = next((w for w in wells if w.get("well") == "3P"), {})
check("Well 3P accuracy >= 0.5", w3p.get("accuracy", 0) >= 0.5,
      f"acc={w3p.get('accuracy', 0):.3f}")
check("Well 3P has regime", w3p.get("regime") is not None)
check("Well 3P has SHmax", w3p.get("shmax_deg") is not None)

# Cached batch should be fast
t0 = time.time()
d2 = api("POST", "/api/batch/analyze-all", {"depth_m": 3000, "pp_mpa": 30})
elapsed2 = time.time() - t0
check("Cached batch < 5s", elapsed2 < 5, f"{elapsed2:.1f}s")

# ── Stakeholder Features ─────────────────────────────

print("\n=== Stakeholder Features ===")

d = api("POST", "/api/analysis/executive-summary", {"well": "3P"})
check("Executive summary returns", d.get("overall_risk") is not None)
check("Summary has sections", len(d.get("sections", [])) > 0)
check("Risk is GREEN/AMBER/RED", d.get("overall_risk") in ["GREEN", "AMBER", "RED"])

d = api("GET", "/api/help/glossary")
check("Glossary has terms", d.get("n_terms", len(d.get("terms", {}))) >= 5)

# ── Data Quality ──────────────────────────────────────

print("\n=== Data Quality ===")

d = api("GET", "/api/data/quality")
check("Data quality returns score", d.get("score", 0) > 0)
check("Data quality has grade", d.get("grade") is not None)

# ── UI Tests ──────────────────────────────────────────

print("\n=== UI Tests ===")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    page.goto(BASE)
    page.wait_for_load_state("networkidle")

    check("Page loads with v3", "v3." in page.text_content("body"))

    # Tab navigation
    for tab_name in ["mlops", "viz", "classify"]:
        link = page.query_selector(f'[data-tab="{tab_name}"]')
        if link:
            link.click()
            page.wait_for_timeout(300)
            tab_el = page.query_selector(f"#tab-{tab_name}")
            check(f"Tab '{tab_name}' activates", tab_el is not None)
        else:
            check(f"Tab '{tab_name}' nav link exists", False)

    # MLOps specific elements
    page.query_selector('[data-tab="mlops"]').click()
    page.wait_for_timeout(300)
    check("RLHF section present", page.query_selector("#rlhf-queue-result") is not None)
    check("Batch section present", page.query_selector("#batch-result") is not None)
    check("Drift section present", page.query_selector("#drift-result") is not None)

    # Loading overlay exists
    check("Loading overlay exists", page.query_selector("#loading-overlay") is not None)
    check("Progress bar exists", page.query_selector("#loading-progress") is not None)

    browser.close()

# ── Summary ───────────────────────────────────────────

print(f"\n{'='*60}")
print(f"GeoStress AI v3.4.0 Test Results: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'='*60}")
if failed > 0:
    exit(1)
