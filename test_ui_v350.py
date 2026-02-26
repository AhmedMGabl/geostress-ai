"""Quick UI smoke test for v3.5.0 calibration features."""
from playwright.sync_api import sync_playwright
import sys

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("http://localhost:8099")
    page.wait_for_load_state("networkidle")

    # Check page loads
    title = page.title()
    assert "GeoStress" in title, f"Title: {title}"
    print(f"PASS: Page loads ({title})")

    # Check version
    version_text = page.locator("#sidebar small").first.inner_text()
    assert "v3.5" in version_text, f"Version: {version_text}"
    print(f"PASS: Version v3.5 shown")

    # Navigate to calibration tab
    page.click('[data-tab="calibration"]')
    page.wait_for_timeout(500)

    # Check field calibration section exists
    field_section = page.locator("text=Field Stress Calibration")
    assert field_section.count() > 0, "Field Stress Calibration section not found"
    print("PASS: Field calibration section visible")

    # Check form elements
    assert page.locator("#field-test-type").count() > 0, "Test type selector missing"
    assert page.locator("#field-depth").count() > 0, "Depth input missing"
    assert page.locator("#field-stress").count() > 0, "Stress input missing"
    assert page.locator("#field-direction").count() > 0, "Direction selector missing"
    print("PASS: Field calibration form elements present")

    # Check buttons
    add_btn = page.locator("text=Add Measurement")
    validate_btn = page.locator("text=Validate Against Field Data")
    assert add_btn.count() > 0, "Add Measurement button missing"
    assert validate_btn.count() > 0, "Validate button missing"
    print("PASS: Calibration action buttons present")

    # Navigate to executive tab
    page.click('[data-tab="executive"]')
    page.wait_for_timeout(500)

    # Check executive calibration status card exists
    cal_card = page.locator("#exec-calibration")
    assert cal_card.count() > 0, "Executive calibration card missing"
    print("PASS: Executive calibration status card exists")

    # Take screenshot
    page.screenshot(path="test_v350_ui.png", full_page=True)
    print("PASS: Screenshot saved")

    browser.close()
    print("\nAll UI tests passed!")
