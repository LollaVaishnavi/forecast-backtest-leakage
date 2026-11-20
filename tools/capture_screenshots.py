"""Screenshot the real HTML report the CLI produces, and refuse an illegible verdict.

The screenshot is the one step in the pipeline that can see pixels, so it is where the
check that the verdict badge is actually visible belongs: in two earlier projects a badge
rendered the same colour as its background while every test still passed. This tool builds
the report if it is missing rather than photographing a stale file, reads back the badge's
computed style, and fails the run if the badge is invisible or its text and background
match. Dev-time only; CI never runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from backtest_leakage.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
REPORT_HTML = SHOTS / "report.html"
REPORT_PNG = SHOTS / "report.png"
MANIFEST = SHOTS / "manifest.json"


def _build_report() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    # Seed 2 with the leaky pipeline is the report the README discusses: a clear LEAK.
    cli_main(["--seed", "2", "--leaky", "--html", str(REPORT_HTML)])


def _badge_reading(page) -> dict:
    return page.evaluate(
        """() => {
          const badge = document.querySelector('.badge');
          if (!badge) return null;
          const s = getComputedStyle(badge);
          return {
            color: s.color,
            background: s.backgroundColor,
            visible: badge.offsetWidth > 0 && badge.offsetHeight > 0,
            text: badge.textContent.trim(),
          };
        }"""
    )


def main() -> int:
    _build_report()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 920, "height": 700})
        page.goto(REPORT_HTML.as_uri())

        reading = _badge_reading(page)
        if reading is None:
            raise SystemExit("no .badge element in the report; the verdict is missing")
        if not reading["visible"]:
            raise SystemExit("the verdict badge has zero size; it would be invisible")
        if reading["color"] == reading["background"]:
            raise SystemExit(
                f"verdict badge text and background are the same colour ({reading['color']})"
            )

        # Clip to the content so the image is not padded with an empty viewport. Frame to
        # the bottom of the last element rather than a fixed offset, so the shot stays
        # tight if the report grows a row.
        content_bottom = page.evaluate(
            "() => { const n = document.querySelector('.note');"
            " return Math.ceil(n.getBoundingClientRect().bottom); }"
        )
        page.screenshot(
            path=str(REPORT_PNG),
            full_page=True,
            clip={"x": 0, "y": 0, "width": 920, "height": content_bottom + 24},
        )
        browser.close()

    MANIFEST.write_text(
        json.dumps({"report": "report.html", "image": "report.png", "badge": reading}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {REPORT_PNG.relative_to(ROOT)} with badge {reading['text']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
