"""voicebench CLI (design.md section 12).

  voicebench corpus                     generate and cache test audio
  voicebench run t0 t1 t2 t3 t4         run selected experiments
  voicebench run all --resume           skip runs already in raw logs
  voicebench clips
  voicebench summarize
  voicebench report
  voicebench all                        everything in order

Failures are logged and counted; they never crash the whole run.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import traceback

EXPERIMENTS = {
    "t0": "voicebench.experiments.t0_network",
    "t1": "voicebench.experiments.t1_turn_latency",
    "t2": "voicebench.experiments.t2_vad_sweep",
    "t3": "voicebench.experiments.t3_behavior",
    "t4": "voicebench.experiments.t4_tts_segment",
}


def _run_experiments(names, cfg, resume):
    import importlib
    failures = 0
    for name in names:
        if name == "all":
            names = list(EXPERIMENTS)
            continue
        mod_name = EXPERIMENTS[name]
        print(f"== {name} ({mod_name}) ==")
        try:
            mod = importlib.import_module(mod_name)
            asyncio.run(mod.run(cfg, resume=resume))
        except Exception as e:
            failures += 1
            print(f"!! experiment {name} failed: {e}")
            traceback.print_exc()
    if failures:
        print(f"{failures} experiment(s) failed; see results/raw/*.jsonl for turn.failed events")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(prog="voicebench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("corpus").add_argument("--force", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("names", nargs="+", choices=list(EXPERIMENTS) + ["all"])
    p.add_argument("--resume", action="store_true")
    sub.add_parser("clips")
    sub.add_parser("summarize")
    sub.add_parser("report")
    sub.add_parser("all")
    args = ap.parse_args(argv)

    from .config import load_config
    cfg = load_config()

    if args.cmd == "corpus":
        from . import corpus
        asyncio.run(corpus.generate(cfg, force=args.force))
    elif args.cmd == "run":
        names = list(EXPERIMENTS) if "all" in args.names else args.names
        sys.exit(1 if _run_experiments(names, cfg, args.resume) else 0)
    elif args.cmd == "clips":
        from . import clips
        clips.run(cfg)
    elif args.cmd == "summarize":
        from . import summarize
        summarize.run(cfg)
    elif args.cmd == "report":
        from . import report
        report.run(cfg)
    elif args.cmd == "all":
        from . import clips, corpus, report, summarize
        asyncio.run(corpus.generate(cfg))
        if _run_experiments(list(EXPERIMENTS), cfg, resume=True):
            print("continuing despite experiment failures (design.md 12)")
        clips.run(cfg)
        summarize.run(cfg)
        report.run(cfg)


if __name__ == "__main__":
    main()
