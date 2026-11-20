"""Render the README's diagram source to a committed SVG that renders anywhere.

GitHub cannot be relied on to render a mermaid fence, and a mermaid SVG with HTML labels
fails to load as an <img src>. So the diagram source lives in a fence labelled
mermaid-source, this tool renders it to an SVG with HTML labels off, rewrites the root
element with explicit dimensions from the viewBox, and paints a light background so edge
labels stay legible in a dark theme. The result renders identically on GitHub, in a
preview, in a PDF, and offline. Dev-time only; CI never runs it.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUT_SVG = ROOT / "docs" / "diagrams" / "architecture.svg"
MANIFEST = ROOT / "docs" / "diagrams" / "manifest.json"

MERMAID_CONFIG = {
    "htmlLabels": False,
    "flowchart": {"htmlLabels": False},
    "themeVariables": {"fontSize": "16px"},
}
PUPPETEER_CONFIG = {"args": ["--no-sandbox", "--disable-gpu"],
                    "executablePath": "/usr/bin/google-chrome"}


def extract_source() -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(r"```mermaid-source\n(.*?)```", text, flags=re.DOTALL)
    if not match:
        raise SystemExit("no ```mermaid-source fence found in README.md")
    return match.group(1).strip()


def _postprocess(svg_text: str) -> str:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.fromstring(svg_text)

    viewbox = root.get("viewBox")
    if not viewbox:
        raise SystemExit("rendered SVG has no viewBox; cannot set intrinsic dimensions")
    _, _, vb_w, vb_h = (float(v) for v in viewbox.split())

    # Explicit pixel dimensions so an <img src> has an aspect ratio to honour.
    root.set("width", str(int(round(vb_w))))
    root.set("height", str(int(round(vb_h))))
    style = root.get("style", "")
    root.set("style", re.sub(r"max-width:[^;]+;?", "", style))

    # A light background rectangle so subgraph titles and edge labels, which have no fill
    # behind them, do not come out dark grey on near black in a dark theme.
    rect = ET.Element("{http://www.w3.org/2000/svg}rect")
    rect.set("x", "0")
    rect.set("y", "0")
    rect.set("width", str(int(round(vb_w))))
    rect.set("height", str(int(round(vb_h))))
    rect.set("fill", "#ffffff")
    root.insert(0, rect)

    if root.findall(".//{http://www.w3.org/2000/svg}foreignObject"):
        raise SystemExit("SVG contains a foreignObject; HTML labels were not disabled")

    return ET.tostring(root, encoding="unicode")


def main() -> int:
    source = extract_source()
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mmd = tmp_path / "diagram.mmd"
        raw_svg = tmp_path / "raw.svg"
        cfg = tmp_path / "mermaid.json"
        pcfg = tmp_path / "puppeteer.json"
        mmd.write_text(source, encoding="utf-8")
        cfg.write_text(json.dumps(MERMAID_CONFIG), encoding="utf-8")
        pcfg.write_text(json.dumps(PUPPETEER_CONFIG), encoding="utf-8")

        subprocess.run(
            ["mmdc", "-i", str(mmd), "-o", str(raw_svg),
             "-c", str(cfg), "-p", str(pcfg), "-b", "transparent"],
            check=True,
        )
        processed = _postprocess(raw_svg.read_text(encoding="utf-8"))

    OUT_SVG.write_text(processed, encoding="utf-8")
    MANIFEST.write_text(
        json.dumps({"source": "README.md ```mermaid-source", "output": "architecture.svg",
                    "html_labels": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_SVG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
