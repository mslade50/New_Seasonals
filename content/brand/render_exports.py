"""Render the Sign Test SVG masters to exact-size PNG exports.

Uses an existing local Chrome/Edge install through Playwright. The SVG files
remain the editable source of truth.
"""

from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
EXPORTS = ROOT / "exports"

ASSETS = {
    "avatar.svg": (400, 400),
    "banner.svg": (1500, 500),
    "chart-template-dark.svg": (1600, 900),
    "chart-template-light.svg": (1600, 900),
    "weekly-scoreboard.svg": (1600, 900),
}

BROWSERS = (
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)


def browser_path() -> Path:
    for candidate in BROWSERS:
        if candidate.exists():
            return candidate
    raise RuntimeError("Chrome or Edge is required to render the PNG exports.")


def render() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(browser_path()),
            headless=True,
            args=["--allow-file-access-from-files"],
        )
        for source_name, (width, height) in ASSETS.items():
            page = browser.new_page(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
            )
            page.goto((SOURCE / source_name).as_uri(), wait_until="load")
            page.screenshot(
                path=str(EXPORTS / source_name.replace(".svg", ".png")),
                omit_background=False,
            )
            page.close()
        browser.close()

    avatar = Image.open(EXPORTS / "avatar.png").convert("RGB")
    avatar.resize((48, 48), Image.Resampling.LANCZOS).save(
        EXPORTS / "avatar-48.png", optimize=True
    )


if __name__ == "__main__":
    render()

