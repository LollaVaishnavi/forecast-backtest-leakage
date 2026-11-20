"""The committed evidence must exist, match the README, and dodge the rendering traps.

These assertions are cheap and need no browser: they guard against shipping a README that
points at a missing image, or an SVG that will not load as an <img src> because it kept
HTML labels or a percentage width.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_committed_evidence_files_are_present():
    for rel in [
        "docs/diagrams/architecture.svg",
        "docs/screenshots/report.png",
        "docs/video/demo.gif",
        "docs/video/manifest.json",
    ]:
        assert (ROOT / rel).exists(), f"missing evidence file {rel}"


def test_every_local_readme_image_points_at_a_real_file():
    readme = (ROOT / "README.md").read_text()
    srcs = re.findall(r'<img[^>]*src="([^"]+)"', readme)
    srcs += re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    local = [s for s in srcs if not s.startswith("http")]
    assert local, "expected at least one local image in the README"
    for src in local:
        assert (ROOT / src).exists(), f"README references missing image {src}"


def test_the_architecture_svg_avoids_the_rendering_traps():
    svg = (ROOT / "docs/diagrams/architecture.svg").read_text()
    assert "foreignObject" not in svg, "HTML labels leave a foreignObject that will not load"
    assert re.search(r'<svg[^>]*\bwidth="\d', svg), "SVG needs an intrinsic pixel width"
    assert re.search(r'<svg[^>]*\bheight="\d', svg), "SVG needs an intrinsic pixel height"
    assert 'fill="#ffffff"' in svg, "SVG needs a painted light background for dark themes"
