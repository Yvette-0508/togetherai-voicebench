"""report: summary.json + content.json -> report.html (design.md 11.4).

1. Deep-merge summary.json into content.json; the harness values win on
   overlapping numeric fields. Checks merge by id (content keeps the notes).
2. Replace the <script id="report-data"> block in report/template.html.
   JSON written with ensure_ascii=False and '</' escaped as '<\\/'.
3. Write site/index.html and copy results/clips/ into site/clips/ (design
   follows 11.4; output goes into site/ so it can be deployed statically,
   e.g. on Render).
4. Fail the build if a required key is missing or sample is still true.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .clock import RESULTS

ROOT = RESULTS.parent
TEMPLATE = ROOT / "report" / "template.html"
SITE_DIR = ROOT / "site"
OUT_HTML = SITE_DIR / "index.html"
DATA_RE = re.compile(
    r'(<script id="report-data" type="application/json">).*?(</script>)',
    re.DOTALL)

REQUIRED_KEYS = ["sample", "meta", "summary", "target_ms", "stages", "turns",
                 "vad_sweep", "segment", "checks", "fixes", "compare", "works",
                 "footnote", "network"]


def deep_merge(content: dict, harness: dict) -> dict:
    """Harness values win; content supplies text/narrative keys."""
    out = dict(content)
    for k, v in harness.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def merge_checks(content_checks: list, harness_checks: list) -> list:
    """Harness supplies pass/n/value; content supplies note text. Order by content."""
    by_id = {c["id"]: c for c in harness_checks if "id" in c}
    merged = []
    used = set()
    for c in content_checks:
        cid = c.get("id") or c.get("name")
        h = by_id.get(cid) or {}
        m = {**h, **{k: v for k, v in c.items() if k != "id" or h},
             **{k: h[k] for k in ("pass", "n", "kind", "value") if k in h}}
        if h:
            used.add(cid)
        merged.append(m)
    for cid, h in by_id.items():  # harness-only checks (shouldn't happen)
        if cid not in used:
            merged.append(h)
    return merged


def substitute_placeholders(obj, data: dict):
    """Replace {network.rtt_p50_ms}-style placeholders in content strings."""
    if isinstance(obj, str):
        def repl(m):
            cur = data
            try:
                for part in m.group(1).split("."):
                    cur = cur[int(part)] if isinstance(cur, list) else cur[part]
            except (KeyError, IndexError, TypeError, ValueError):
                return m.group(0)
            return str(cur)
        return re.sub(r"\{(\w[\w.\d]*)\}", repl, obj)
    if isinstance(obj, list):
        return [substitute_placeholders(v, data) for v in obj]
    if isinstance(obj, dict):
        return {k: substitute_placeholders(v, data) for k, v in obj.items()}
    return obj


def run(cfg: dict):
    summary = json.loads((RESULTS / "summary.json").read_text())
    content = json.loads((ROOT / "content.json").read_text())

    data = deep_merge(content, summary)
    data["checks"] = merge_checks(content.get("checks", []),
                                  summary.get("checks", []))
    # Footnote: hand-written sentence + network baseline + client location.
    net = data.get("network", {})
    data["footnote"] = (
        f"{content.get('footnote', '').rstrip('.')}. "
        f"Network baseline: round trip {net.get('rtt_p50_ms', '?')} ms, "
        f"STT connect {net.get('stt_connect_p50_ms', '?')} ms, "
        f"TTS connect {net.get('tts_connect_p50_ms', '?')} ms. "
        f"Client: {data.get('run', {}).get('client_location')}"
    )
    data = substitute_placeholders(data, data)

    if data.get("sample") is True:
        raise SystemExit("summary.json still has sample=true; refusing to build")
    if not data.get("turns"):
        raise SystemExit("summary.json has no T1 turn data; refusing to build")
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise SystemExit(f"report data is missing required keys: {missing}")

    html = TEMPLATE.read_text()
    payload = json.dumps(data, indent=2, ensure_ascii=False).replace("</", "<\\/")
    html, n = DATA_RE.subn(lambda m: m.group(1) + "\n" + payload + "\n" + m.group(2),
                           html)
    if n != 1:
        raise SystemExit("report-data script block not found in template")
    SITE_DIR.mkdir(exist_ok=True)
    OUT_HTML.write_text(html)

    clips_dst = SITE_DIR / "clips"
    if (RESULTS / "clips").exists():
        if clips_dst.exists():
            shutil.rmtree(clips_dst)
        shutil.copytree(RESULTS / "clips", clips_dst)
    print(f"wrote {OUT_HTML} (deploy the site/ dir as a static site)")
