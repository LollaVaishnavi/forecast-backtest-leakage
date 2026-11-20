"""Record a replay of the CLI as a GIF, from real captured output.

Every line in the GIF is the real stdout of a command that actually ran, and each segment
is paced by that command's measured wall time, so a slow command looks slow. This is a
replay built from a manifest, not a screen recording, and the README says so. The segment
order is the argument, not the feature list: the honest baseline first (the tool refusing
to certify a clean pipeline), then the leak it detects on the same series, then the
receipts proving the numbers. Dev-time only; CI never runs it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "docs" / "video"
GIF = VIDEO_DIR / "demo.gif"
MP4 = VIDEO_DIR / "demo.mp4"
MANIFEST = VIDEO_DIR / "manifest.json"

SEGMENTS = [
    ["btleak", "--seed", "2", "--clean"],
    ["btleak", "--seed", "2", "--leaky"],
    [sys.executable, "tools/check_numbers.py"],
]


def _label(argv: list[str]) -> str:
    out = []
    for token in argv:
        out.append("python" if token == sys.executable else token)
    return " ".join(out)


def _capture() -> list[dict]:
    # Warm the import once so the first timed segment does not pay the startup cost and
    # the pacing of the first command is not an outlier.
    subprocess.run(["btleak", "--seed", "0", "--clean"], cwd=ROOT,
                    capture_output=True, text=True, check=False)

    segments = []
    for argv in SEGMENTS:
        start = time.perf_counter()
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - start
        segments.append({
            "command": _label(argv),
            "output": (proc.stdout + proc.stderr).rstrip("\n"),
            "exit_code": proc.returncode,
            "seconds": round(elapsed, 2),
        })
    return segments


def _player_html(segments: list[dict]) -> tuple[str, int]:
    data = json.dumps(segments)
    # Typing pace and a reveal pause scaled by, but capped against, the real duration, so
    # the GIF stays short while a slower command still reads as slower.
    total_ms = 0
    for seg in segments:
        total_ms += len(seg["command"]) * 45 + 400 + min(int(seg["seconds"] * 1000), 2200) + 700
    total_ms += 1200

    page = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    body{margin:0;background:#0f1117;}
    #t{font-family:'DejaVu Sans Mono',monospace;font-size:16px;line-height:1.5;
       color:#e6e6e6;padding:20px;white-space:pre-wrap;}
    .cmd{color:#6cc6ff;} .ok{color:#6ee787;} .bad{color:#ff7b72;} .dim{color:#8b949e;}
    </style></head><body><div id="t"></div><script>
    const SEG = __DATA__;
    const term = document.getElementById('t');
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    function line(txt, cls){const s=document.createElement('span');
      if(cls)s.className=cls; s.textContent=txt+"\\n"; term.appendChild(s);}
    async function type(txt){const s=document.createElement('span'); s.className='cmd';
      term.appendChild(s);
      for(const ch of txt){s.textContent+=ch; await sleep(45);}
      s.textContent+="\\n";}
    (async () => {
      for(const seg of SEG){
        const p=document.createElement('span'); p.className='dim';
        p.textContent='$ '; term.appendChild(p);
        await type(seg.command);
        await sleep(400);
        const cls = seg.exit_code===0 ? 'ok' : (seg.exit_code===3 ? 'bad' : '');
        for(const l of seg.output.split("\\n")) line(l, cls);
        line('[exit '+seg.exit_code+']', 'dim');
        await sleep(Math.min(seg.seconds*1000, 2200));
        line('', null);
      }
    })();
    </script></body></html>"""
    return page.replace("__DATA__", data), total_ms


def _transcode(webm: Path) -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(webm), "-movflags", "+faststart",
         "-pix_fmt", "yuv420p", "-vf", "scale=820:-2", str(MP4)],
        check=True, capture_output=True,
    )
    palette = webm.parent / "palette.png"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(webm), "-vf", "fps=12,scale=820:-1:flags=lanczos,palettegen",
         str(palette)], check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(webm), "-i", str(palette),
         "-lavfi", "fps=12,scale=820:-1:flags=lanczos[x];[x][1:v]paletteuse", str(GIF)],
        check=True, capture_output=True,
    )


def main() -> int:
    segments = _capture()
    page_html, total_ms = _player_html(segments)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        html_file = tmp_path / "player.html"
        html_file.write_text(page_html, encoding="utf-8")

        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox"])
            context = browser.new_context(
                viewport={"width": 900, "height": 560},
                record_video_dir=str(tmp_path),
                record_video_size={"width": 900, "height": 560},
            )
            page = context.new_page()
            page.goto(html_file.as_uri())
            page.wait_for_timeout(total_ms + 500)
            context.close()
            browser.close()
            webm = next(tmp_path.glob("*.webm"))
            _transcode(webm)

    MANIFEST.write_text(json.dumps({"segments": segments, "note":
        "every line is real captured stdout; segments paced by measured wall time"},
        indent=2) + "\n", encoding="utf-8")
    size_mb = GIF.stat().st_size / 1e6
    print(f"wrote {GIF.relative_to(ROOT)} ({size_mb:.1f} MB) and {MP4.relative_to(ROOT)}")
    if size_mb > 5.0:
        print("warning: GIF exceeds 5 MB; consider fewer segments", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
