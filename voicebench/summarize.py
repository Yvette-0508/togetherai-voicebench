"""summarize: raw logs -> results/summary.json + summary.provenance.json.

Every number carries the run ids it was computed from (design.md 11.2).
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from .clock import RAW_DIR, RESULTS, git_sha, read_events
from .corpus import load_manifest

PUNCT_RE = re.compile(r"[^\w\s]")


def p50(xs):
    return round(float(np.percentile(xs, 50)), 1) if xs else None


def p90(xs):
    return round(float(np.percentile(xs, 90)), 1) if xs else None


def by_run(events):
    groups = defaultdict(list)
    for e in events:
        groups[e["run_id"]].append(e)
    return groups


def turn_results(events, configs=None):
    """Collect per-run stage timestamps from turn.complete events."""
    out = defaultdict(dict)
    for e in events:
        if e["type"] == "turn.complete" and "t_eos" in e.get("meta", {}):
            out[e["config"]][e["run_id"]] = e["meta"]
    return out


def ms(a, b):
    """(a - b) in milliseconds, or None if either is missing."""
    if a is None or b is None:
        return None
    return (a - b) / 1e6


def _norm(s: str) -> str:
    return PUNCT_RE.sub("", s.lower()).strip()


def summarize(cfg: dict) -> tuple[dict, dict]:
    target_ms = cfg["target_ms"]
    summary = {
        "sample": False,
        "target_ms": target_ms,
        "target_label": "1 s" if target_ms == 1000 else f"{target_ms} ms",
        "turns": [],
        "vad_sweep": {"n": cfg["runs"]["t2_turns"], "rows": []},
        "segment": {"n": cfg["runs"]["t4_replies"], "rows": []},
        "checks": [],
        "compare": {"today": {"lines": 0}},
        "network": {},
        "clips": [],
        "run": {
            "date": None,
            "client_location": cfg.get("client_location", ""),
            "models": cfg["models"],
            "git_sha": git_sha(),
        },
    }
    prov = {}  # provenance: key path -> run ids

    # ---------- T0 network ----------
    t0 = read_events("t0")
    rtts = [e["meta"]["ms"] for e in t0 if e["type"] == "http.rtt"
            and e["meta"].get("status") == 200]
    connects = defaultdict(list)
    for e in t0:
        if e["type"] == "session.created":
            connects[e["config"]].append(e["meta"]["ms"])
    summary["network"] = {
        "rtt_p50_ms": p50(rtts) or 0,
        "stt_connect_p50_ms": p50(connects.get("stt_connect", [])) or 0,
        "tts_connect_p50_ms": p50(connects.get("tts_connect", [])) or 0,
    }
    prov["network"] = {
        "rtt": [e["run_id"] for e in t0 if e["type"] == "http.rtt"],
        "stt_connect": [e["run_id"] for e in t0
                        if e["config"] == "stt_connect"],
        "tts_connect": [e["run_id"] for e in t0
                        if e["config"] == "tts_connect"],
    }

    # ---------- T1 turn latency ----------
    t1 = read_events("t1")
    turns = turn_results(t1)
    control = turns.get("control", {})
    stt_stage_ms = p50([ms(v["t_final"], v["t_eos"]) for v in control.values()
                        if ms(v["t_final"], v["t_eos"]) is not None]) or 0
    prov.setdefault("turns", {})["control (stt stage source)"] = sorted(control)

    labels = {
        "default": ("Default settings", f"silence {cfg['vad']['default_silence_ms']} ms"),
        "tuned": ("Tuned VAD", f"silence {cfg['vad']['tuned_silence_ms']} ms"),
        "tool": ("With a tool call", "1 function call"),
    }
    for name in ("default", "tuned", "tool"):
        vals = [v for v in turns.get(name, {}).values()
                if v.get("t_first_audio") and v.get("t_eos")]
        if not vals:
            continue
        fin_minus_eos = [ms(v["t_final"], v["t_eos"]) for v in vals
                         if ms(v["t_final"], v["t_eos"]) is not None]
        vad_ms = max(p50(fin_minus_eos) - stt_stage_ms, 0) if fin_minus_eos else 0
        llm_ms = p50([ms(v["t_first_sentence"], v["t_final"]) for v in vals
                      if ms(v["t_first_sentence"], v["t_final"]) is not None]) or 0
        tts_ms = p50([ms(v["t_first_audio"], v["t_first_sentence_sent"]) for v in vals
                      if ms(v["t_first_audio"], v["t_first_sentence_sent"]) is not None]) or 0
        totals = [ms(v["t_first_audio"], v["t_eos"]) for v in vals]
        label, tag = labels[name]
        summary["turns"].append({
            "label": label, "tag": tag,
            "p50": {"vad": round(vad_ms), "stt": round(stt_stage_ms),
                    "llm": round(llm_ms), "tts": round(tts_ms)},
            "p50_total": round(p50(totals) or 0), "p90": round(p90(totals) or 0),
        })
        prov["turns"][name] = sorted(turns[name])

    # ---------- T2 VAD sweep ----------
    t2 = read_events("t2")
    sweeps = turn_results(t2)
    for ms_ in cfg["vad"]["sweep_ms"]:
        cfg_name = f"silence-{ms_}"
        vals = sweeps.get(cfg_name, {})
        cut = sum(1 for v in vals.values() if v.get("cut_off"))
        ok = [ms(v["t_first_audio"], v["t_eos"]) for v in vals.values()
              if not v.get("cut_off")
              and ms(v["t_first_audio"], v["t_eos"]) is not None]
        row = {"ms": ms_, "cut": cut, "p50": round(p50(ok) or 0)}
        if ms_ == cfg["vad"]["default_silence_ms"]:
            row["default"] = True
        summary["vad_sweep"]["rows"].append(row)
        prov.setdefault("vad_sweep", {})[str(ms_)] = sorted(vals)

    # ---------- T4 segment modes ----------
    t4 = read_events("t4")
    seg_events = defaultdict(list)
    for e in t4:
        if e["type"] == "turn.complete":
            seg_events[e["config"]].append(e)
    for mode in ("immediate", "sentence", "never"):
        evs = seg_events.get(mode, [])
        ttfas = [e["meta"]["ttfa_ms"] for e in evs
                 if e["meta"].get("ttfa_ms") is not None]
        stall_replies = sum(1 for e in evs if e["meta"].get("stalls", 0) > 0)
        row = {"mode": mode, "ttfa": round(p50(ttfas) or 0),
               "stalls": stall_replies}
        if mode == "sentence":
            row["default"] = True
        summary["segment"]["rows"].append(row)
        prov.setdefault("segment", {})[mode] = sorted(
            e["run_id"] for e in evs)

    # ---------- T3 checks ----------
    checks = []
    t3 = read_events("t3")

    def done_evs(config):
        evs = [e for e in t3 if e["config"] == config
               and e["type"] == "turn.complete"]
        return evs

    long_evs = done_evs("long_question")
    checks.append({
        "id": "long_question", "name": "Long question stays one turn",
        "tag": "8 s of speech",
        "pass": sum(1 for e in long_evs if e["meta"].get("pass")),
        "n": len(long_evs)})
    prov.setdefault("checks", {})["long_question"] = sorted(
        e["run_id"] for e in long_evs)

    # tool_valid from T1 tool-turn tool_call meta events
    tool_meta = [e for e in t1 if e["type"] == "llm.tool_call"
                 and e["config"] == "tool"][:10]
    checks.append({
        "id": "tool_valid", "name": "Tool call valid inside the turn",
        "pass": sum(1 for e in tool_meta if e["meta"].get("valid")),
        "n": len(tool_meta)})
    prov["checks"]["tool_valid"] = sorted(e["run_id"] for e in tool_meta)

    # WER from T1 default turns (first 10)
    from jiwer import wer
    manifest = load_manifest()
    by_set_id = {s: {e["id"]: e for e in manifest["sets"][s]}
                 for s in manifest["sets"]}
    starts = {e["run_id"]: e["meta"] for e in t1
              if e["type"] == "turn.start" and e["config"] == "default"}
    finals = {}
    for e in t1:
        if (e["config"] == "default" and e["type"].endswith("completed")
                and "text" in e.get("meta", {}) and e["run_id"] not in finals):
            finals[e["run_id"]] = e["meta"]["text"]
    wer_runs = [r for r in starts if r in finals][:10]
    wer_pass = 0
    wer_vals = []
    for r in wer_runs:
        corpus_id = starts[r].get("corpus", "")
        set_name, _, _ = corpus_id.rpartition("-")
        ref = by_set_id.get(set_name, {}).get(corpus_id, {}).get("text", "")
        if not ref:
            continue
        v = wer(_norm(ref), _norm(finals[r]))
        wer_vals.append(v)
        wer_pass += 1 if v <= 0.05 else 0
    checks.append({
        "id": "wer", "name": "Transcript matches the script", "tag": "WER ≤ 5%",
        "pass": wer_pass, "n": len(wer_vals)})
    prov["checks"]["wer"] = wer_runs

    voice_evs = [e for e in t3 if e["config"] == "openai_voice"
                 and e["type"] == "check.complete"]
    checks.append({
        "id": "openai_voice", "name": "OpenAI voice names", "tag": "voice=alloy",
        "pass": sum(1 for e in voice_evs if e["meta"].get("pass")),
        "n": len(voice_evs)})
    prov["checks"]["openai_voice"] = sorted(e["run_id"] for e in voice_evs)

    clear_evs = done_evs("interrupt_clear")
    clear_pass = sum(1 for e in clear_evs
                     if (e["meta"].get("last_delta_after_cmd_ms") is None
                         or e["meta"]["last_delta_after_cmd_ms"] <= 250))
    checks.append({
        "id": "clear_stops", "name": "input_text_buffer.clear stops audio",
        "pass": clear_pass, "n": len(clear_evs)})
    prov["checks"]["clear_stops"] = sorted(e["run_id"] for e in clear_evs)

    cancel_evs = done_evs("interrupt_cancel")
    quiet = [e["meta"]["quiesced_ms"] for e in cancel_evs
             if e["meta"].get("quiesced_ms") is not None]
    cancel_p50 = p50(quiet)
    checks.append({
        "id": "cancel_stop", "name": "context.cancel to silence",
        "kind": "value",
        "value": f"{round(cancel_p50)} ms" if cancel_p50 is not None else "n/a"})
    prov["checks"]["cancel_stop"] = sorted(e["run_id"] for e in cancel_evs)

    # Interruption detail numbers for content placeholders (design.md FE follow-up)
    both = clear_evs + cancel_evs
    buffered = [e["meta"]["buffered_at_cmd_ms"] for e in both
                if e["meta"].get("buffered_at_cmd_ms") is not None]
    after = [e["meta"]["audio_after_cmd_ms"] for e in clear_evs
             if e["meta"].get("audio_after_cmd_ms") is not None]
    summary["interrupt"] = {
        "buffered_at_cmd_p50_ms": p50(buffered) or 0,
        "clear_audio_after_cmd_p50_ms": p50(after) or 0,
        "cancel_to_silence_p50_ms": cancel_p50 or 0,
    }

    summary["checks"] = checks

    # ---------- T6 glue lines ----------
    pipe = RESULTS.parent / "pipeline.py"
    if pipe.exists():
        lines = sum(1 for ln in pipe.read_text().splitlines()
                    if ln.strip() and not ln.strip().startswith("#"))
        summary["compare"]["today"]["lines"] = lines
        prov["pipeline_lines"] = ["pipeline.py"]

    # ---------- clips ----------
    clips_dir = RESULTS / "clips"
    if clips_dir.exists():
        summary["clips"] = [
            json.loads(p.read_text()) for p in sorted(clips_dir.glob("*.json"))]

    return summary, prov


def run(cfg: dict):
    summary, prov = summarize(cfg)
    if len(summary["turns"]) < 3:
        print("warning: T1 turn blocks incomplete; run `voicebench run t1` first")
    out = RESULTS / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    (RESULTS / "summary.provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"wrote {out} and summary.provenance.json")
