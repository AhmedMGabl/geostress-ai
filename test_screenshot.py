"""Take screenshots of key app tabs for review."""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto("http://localhost:8099")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2000)

    # Executive tab (default)
    page.screenshot(path="screenshots/executive.png")
    print("1. Executive tab captured")

    # Calibration tab
    page.click('[data-tab="calibration"]')
    page.wait_for_timeout(1000)
    page.screenshot(path="screenshots/calibration.png")
    print("2. Calibration tab captured")

    # MLOps tab
    page.click('[data-tab="mlops"]')
    page.wait_for_timeout(2000)
    page.screenshot(path="screenshots/mlops.png")
    print("3. MLOps tab captured")

    browser.close()
    print("Done!")
