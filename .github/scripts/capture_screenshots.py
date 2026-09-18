from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Browser, Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

BASE_URL = "http://127.0.0.1:8501"
OUT = Path("docs/images")
VIEWPORT = {"width": 1920, "height": 1000}


def open_app(browser: Browser) -> tuple[object, Page]:
    context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
    page = context.new_page()
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60_000)
    page.get_by_text("CertiControl", exact=True).first.wait_for(timeout=60_000)
    return context, page


def choose_example(page: Page, name: str) -> None:
    selectbox = page.locator('[data-testid="stSelectbox"]').first
    selectbox.click()
    try:
        option = page.locator('[role="option"]').filter(has_text=name)
        option.first.wait_for(timeout=10_000)
        option.first.click()
    except PlaywrightTimeoutError:
        page.get_by_text(name, exact=True).last.click()

    page.get_by_role("button", name="Load selected example").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Analyze system").click()
    page.wait_for_timeout(2500)


def click_tab(page: Page, name: str) -> None:
    try:
        page.get_by_role("tab", name=name, exact=True).click()
    except PlaywrightTimeoutError:
        page.get_by_text(name, exact=True).first.click()
    page.wait_for_timeout(700)


def capture_overview(browser: Browser) -> None:
    context, page = open_app(browser)
    try:
        choose_example(page, "Four-part Kalman structure")
        click_tab(page, "Overview")
        page.get_by_text("Certificate status", exact=True).wait_for(timeout=30_000)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "overview.png"), full_page=False)
    finally:
        context.close()


def capture_lqr(browser: Browser) -> None:
    context, page = open_app(browser)
    try:
        choose_example(page, "Unstable but controllable")
        click_tab(page, "LQR")
        title = page.get_by_text("LQR certificate", exact=True)
        title.wait_for(timeout=30_000)
        title.scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "lqr.png"), full_page=False)
    finally:
        context.close()


def capture_kalman(browser: Browser) -> None:
    context, page = open_app(browser)
    try:
        choose_example(page, "Four-part Kalman structure")
        click_tab(page, "Kalman Structure")
        page.get_by_text("Four-part structure", exact=True).wait_for(timeout=30_000)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "kalman.png"), full_page=False)
    finally:
        context.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            capture_overview(browser)
            capture_lqr(browser)
            capture_kalman(browser)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
