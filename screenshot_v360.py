"""Take screenshots of v3.6.0 features: uncertainty, decision matrix, executive."""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto("http://localhost:8099")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(3000)

    # Executive tab with decision matrix (needs to click the summary button)
    page.click('[data-tab="executive"]')
    page.wait_for_timeout(1000)
    # Trigger executive summary
    exec_btn = page.locator("text=Generate Executive Summary")
    if exec_btn.count() > 0:
        exec_btn.first.click()
        page.wait_for_timeout(10000)  # Wait for executive summary to generate
    page.screenshot(path="screenshots/executive_v360.png", full_page=True)
    print("1. Executive tab with decision matrix captured")

    # Inversion tab with uncertainty
    page.click('[data-tab="inversion"]')
    page.wait_for_timeout(500)
    inv_btn = page.locator("text=Run Inversion")
    if inv_btn.count() > 0:
        inv_btn.first.click()
        page.wait_for_timeout(8000)
    page.screenshot(path="screenshots/inversion_v360.png", full_page=True)
    print("2. Inversion tab with uncertainty captured")

    browser.close()
    print("Done!")
