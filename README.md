# VoiceBench

A local test harness that measures a voice agent built on Together AI
(realtime STT → streamed chat → realtime TTS) and fills a static report page
with real, traceable results. Spec: `design.md`.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # paste your TOGETHER_API_KEY
brew install ffmpeg    # for clip export (opus)
```

## Run

```bash
voicebench all          # corpus -> T0..T4 -> clips -> summarize -> report
# or step by step:
voicebench corpus
voicebench run t0 t1 t2 t3 t4
voicebench run all --resume   # skip turn ids already in raw logs
voicebench clips
voicebench summarize
voicebench report             # writes site/index.html + site/clips/
```

Deploy: publish the `site/` directory as a static site (Render: Static Site →
Publish directory `site`). The page talks to no backend and contains no
credentials — raw logs and the API key never leave your machine.

## Method

- **Corpus** — `results/corpus/`, generated once with Together TTS (Kokoro)
  over HTTP, resampled 24 kHz → 16 kHz with `resample_poly(x, 2, 3)`.
  Sets: `short` (24 one-sentence questions), `paused` (24 with 300–600 ms of
  inserted silence mid-sentence), `long` (12 questions of 8–9 s), `tool`
  (24 weather questions). Pauses are inserted as digital silence after
  synthesis because TTS won't reliably produce them.
- **Runs** — 2 warm-up turns per config (discarded, never logged), then 20
  measured turns (`t1_turns`/`t2_turns`/`t3_runs` in `config.yaml`). Turn
  audio is streamed in real time: 20 ms frames on a drift-corrected schedule
  (`frame k` at `start + k·20 ms`), 300 ms silence pre-roll, then silence
  until the final transcript or a 3 s tail.
- **Client location** — see `client_location` in `config.yaml`; the page
  footnote also carries the T0 network baseline (HTTP round trip and cold
  WebSocket connect times) so server latency can be separated from network.
- **Stages** — `stt` is the median `t_final − t_eos` over the `control`
  config (VAD off, commit at end of speech); `vad` is the config's
  `t_final − t_eos` median minus that, floored at 0; `llm` is
  `t_first_sentence − t_final`; `tts` is `t_first_audio − t_first_sentence_sent`.
  Stage medians don't sum to the median total; the page shows `p50_total`.
- **Traceability** — every number in `results/summary.json` traces to
  `run_id`s listed in `results/summary.provenance.json`, which point into
  `results/raw/*.jsonl`.

## Known limits

- Synthetic speech is cleaner than real users, so transcript accuracy
  (WER) and VAD behavior are optimistic.
- One region, one residential network. T0 separates network from server
  time, but absolute latency varies with where you run it.
- One default model per stage; this is not a model comparison.
- `VoiceBench` measures client-side; it can't see queuing inside Together's
  infra.

## Layout

```
voicebench/          the harness package (cli, clock/pacer/event log, clients,
                     corpus, experiments t0–t4, clips, summarize, report)
pipeline.py          the minimal working voice loop; its line count is the
                     "lines of glue" metric on the page
report/template.html the report template (derived from together_voice_report.html)
site/                static build output, deployable as-is
results/             raw event logs, corpus, recorded audio, clips, summary.json
```
