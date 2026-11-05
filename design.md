# VoiceBench: build spec

A local test harness that measures a voice agent built on Together AI and fills the report page `together_voice_report.html` with real results.

- Status: draft v0.1, Nov 4, 2026
- Owner: Yvette Pan
- Deadline: results and page ready by Nov 5, noon

---

## 1. Goal

Every number on the report page must come from this harness and be traceable to a raw event log. The harness answers four questions:

1. In one voice turn, where do the milliseconds go between the user finishing speaking and the first audio of the reply?
2. What do the two main latency knobs trade off: STT end-of-turn silence (speed vs. cutting users off) and TTS `segment` mode (first audio vs. smooth playback)?
3. Does the agent behave correctly on things real voice agents hit: long questions, tool calls, interruptions, OpenAI-style voice names?
4. How much glue code does a developer write today to build the loop?

### Non-goals

- No server, database, or hosted backend. The page is a static HTML file.
- No live, public voice demo. The API key never leaves the local machine.
- No browser audio capture or WebRTC.
- No ranking of partner TTS or STT models. One default model per stage.

---

## 2. Architecture

```
local machine (Python 3.11+, TOGETHER_API_KEY in .env)
│
├── corpus      generate test questions as audio with Together TTS
├── run         stream corpus audio in real time through
│               STT WebSocket → streamed chat completion → TTS WebSocket
│               and log every event with a monotonic timestamp
├── clips       stitch recorded audio into short "listen" clips
├── summarize   raw logs → results/summary.json (medians, p90, counts)
└── report      summary.json + content.json → report.html (static)
                                             │
                     deploy anywhere static ─┘  (GitHub Pages, Vercel, yanpan.dev)
```

---

## 3. Stack

| Need | Choice |
| --- | --- |
| Runtime | Python 3.11+, `asyncio` |
| WebSockets | `websockets` (or `aiohttp`); raw protocol, so the code mirrors what a developer writes today |
| Chat and OpenAI-compat checks | `openai` SDK pointed at `https://api.together.ai/v1`; `together` SDK optional |
| Audio | `numpy`, `scipy.signal.resample_poly`, `soundfile`; `ffmpeg` for mp3/opus export |
| Transcript accuracy | `jiwer` |
| Config | `pyyaml`, `python-dotenv` |

---

## 4. Repo layout

```
together-voice-test/
  README.md               method, how to reproduce, known limits
  pyproject.toml
  .env.example            TOGETHER_API_KEY=
  config.yaml             models, run counts, knob values
  content.json            hand-written page content (see 11.3)
  pipeline.py             the voice loop itself; its line count is the "glue code" metric
  voicebench/
    cli.py
    clock.py              monotonic clock, real-time frame pacer
    clients/
      stt.py              realtime STT session wrapper
      tts.py              TTS WebSocket wrapper, HTTP TTS helpers
      llm.py              streamed chat with sentence splitting and tool loop
    playback.py           playback simulator (stall detection)
    corpus.py
    experiments/
      t0_network.py
      t1_turn_latency.py
      t2_vad_sweep.py
      t3_behavior.py
      t4_tts_segment.py
      t5_concurrency.py   (P2)
    clips.py
    summarize.py
    report.py
  report/template.html    copy of together_voice_report.html
  results/
    raw/*.jsonl
    corpus/
    clips/
    summary.json
  report.html             build output
```

---

## 5. Configuration

```yaml
client_location: "San Francisco, home Wi-Fi"   # printed in the page footnote
models:
  stt: openai/whisper-large-v3
  llm: "<fast instruct model from the serverless list>"
  tts: hexgrad/Kokoro-82M
  voice: af_alloy
audio:
  stt_sample_rate: 16000      # STT input: 16 kHz mono 16-bit PCM
  tts_sample_rate: 24000      # TTS output PCM
  frame_ms: 20
runs:
  warmup: 2                   # discarded
  t1_turns: 20
  t2_turns: 20
  t3_runs: 10
  t4_replies: 20
vad:
  default_silence_ms: 500
  tuned_silence_ms: 250
  sweep_ms: [150, 250, 500, 800]
llm:
  system: "You are a voice assistant. Answer in one or two short sentences."
  max_tokens: 120
  temperature: 0.3
target_ms: 1000
```

---

## 6. API surface used

As documented on docs.together.ai on Nov 4, 2026. Treat every name below as something to confirm on the first run (section 14).

### Realtime STT

- URL: `wss://api.together.ai/v1/realtime?model={stt}&input_audio_format=pcm_s16le_16000`
- Headers: `Authorization: Bearer <key>`, `OpenAI-Beta: realtime=v1`
- VAD on connect: `&turn_detection=server_vad&threshold=0.3&min_silence_duration_ms=500`. Disable with `turn_detection=none`.
- VAD can also be changed after `session.created` with `transcription_session.updated`.

| Direction | Event | Notes |
| --- | --- | --- |
| client → server | `input_audio_buffer.append` | `audio`: base64 PCM chunk |
| client → server | `input_audio_buffer.commit` | forces transcription; required when VAD is off |
| client → server | `transcription_session.updated` | VAD settings |
| server → client | `session.created` | |
| server → client | `conversation.item.input_audio_transcription.delta` | interim text |
| server → client | `conversation.item.input_audio_transcription.completed` | final text for one utterance |

Documented VAD defaults: `threshold` 0.3, `min_silence_duration_ms` 500, `min_speech_duration_ms` 250, `max_speech_duration_s` 5.0, `speech_pad_ms` 250.

### TTS WebSocket

- URL: `wss://api.together.ai/v1/audio/speech/websocket?model={tts}&voice={voice}&response_format=pcm&sample_rate=24000`
- Auth: `Authorization` header (preferred) or `api_key` query parameter.
- Other params: `segment` = `sentence` (default) | `immediate` | `never`; `max_partial_length`; `alignment=word`.
- No SDK; raw WebSocket only.

| Direction | Event | Notes |
| --- | --- | --- |
| client → server | `input_text_buffer.append` | `text`, optional `context_id` |
| client → server | `input_text_buffer.commit` | flush at end of input |
| client → server | `input_text_buffer.clear` | drops queued text; text already being processed continues |
| client → server | `context.cancel` | `context_id`; cancels that context |
| client → server | `tts_session.updated` | change voice and other settings |
| server → client | `session.created` | |
| server → client | `conversation.item.input_text.received` | |
| server → client | `conversation.item.audio_output.delta` | `delta`: base64 PCM |
| server → client | `conversation.item.audio_output.done` | |
| server → client | `context.cancelled` | confirms cancel |
| server → client | `conversation.item.tts.failed` | error |

### Other endpoints

- Chat: `POST /v1/chat/completions` with `stream=true`, `tools`.
- TTS over HTTP: `POST /v1/audio/speech` (full file, and a streaming variant for raw PCM bytes).
- Voices: the "List voices" endpoint.
- OpenAI SDK compatibility: `client.audio.speech.create(...)` with `base_url="https://api.together.ai/v1"`.

---

## 7. Shared building blocks

### 7.1 Clock and event log

- All timestamps come from `time.perf_counter_ns()`.
- Every event (sent or received) is appended to `results/raw/{experiment}.jsonl`:

```json
{"run_id": "t1-default-007", "experiment": "t1", "config": "default",
 "turn": 7, "t_ns": 123456789, "dir": "recv",
 "type": "conversation.item.input_audio_transcription.completed",
 "meta": {"chars": 41}}
```

- Do not log audio payloads in JSONL. Save audio separately when a clip needs it.

### 7.2 Real-time frame pacer

- Sends 20 ms frames (320 samples, 640 bytes at 16 kHz) on a drift-corrected schedule: the k-th frame is sent at `start + k × 20 ms`. Do not use `sleep(0.02)` in a loop, because the error accumulates.
- Sequence per turn: 300 ms of silence pre-roll, the speech, then continuous silence frames until the final transcript arrives or 3 s pass.
- `t_eos` (end of speech) is the send time of the last non-silence frame.

### 7.3 LLM streamer

- Streams the chat completion and records the time of the first token and of the first complete sentence. A sentence ends at the regex `(?<=[.!?])\s` or at the end of the stream.
- Sends each completed sentence to TTS as soon as it is available. This is the pattern in Together's phone-agent guide.
- Tool loop: if the response contains `tool_calls`, run a local stub (`get_weather` returns a fixed JSON immediately), append the result, and stream a second call. Validate the arguments against the tool schema.

### 7.4 Playback simulator

Computes stalls from chunk arrival times, as if playback started at the first chunk.

```
dur(chunk) = bytes / (sample_rate × 2)        # 16-bit mono PCM
play_end = t(first) + dur(first)
for each later chunk k:
    if t_k > play_end + 20 ms: stalls += 1; play_end = t_k + dur_k
    else:                      play_end += dur_k
```

The simulator also reports how much audio is buffered ahead of the playhead at any moment. T3 uses this for interruptions.

---

## 8. Corpus

Generate once with Together TTS over HTTP and cache it in `results/corpus/` with a `manifest.json` that records text, duration, and pause positions. Resample the 24 kHz output to 16 kHz with `resample_poly(x, 2, 3)`, unless TTS accepts `sample_rate=16000` directly (check in section 14).

| Set | Count | Content | Used by |
| --- | --- | --- | --- |
| `short` | 24 | one sentence, 2–3 s, general questions | T1 |
| `paused` | 24 | two clauses joined by 300–600 ms of inserted silence, for example "I need to book a table... for four people tonight." | T2 |
| `long` | 12 | one continuous question, 8–9 s | T3 |
| `tool` | 24 | questions that require `get_weather`, 2–3 s | T1, T3 |

Known limit: synthetic speech is cleaner than real users, so the accuracy numbers are optimistic. Say so in the README.

---

## 9. Experiments

### T0. Network baseline (P0)

- 20 small authenticated requests (for example, list models): report median round trip.
- 10 cold connects each for STT and TTS WebSockets: time from connect to `session.created`.
- Purpose: separate network from server time; footnote on the page.
- Output: `network.rtt_p50_ms`, `network.stt_connect_p50_ms`, `network.tts_connect_p50_ms`.

### T1. Turn latency (P0)

Configs, with warm connections, 2 warm-up turns discarded and 20 measured:

| Config | STT | LLM | Corpus |
| --- | --- | --- | --- |
| `default` | VAD on, silence 500 ms | plain answer | `short` |
| `tuned` | VAD on, silence 250 ms | plain answer | `short` |
| `tool` | VAD on, silence 250 ms | one tool call, then answer | `tool` |
| `control` | VAD off; `commit` sent at `t_eos` | none | `short` |

Timestamps per turn: `t_eos`, `t_final` (first `completed` after `t_eos`), `t_first_sentence`, `t_first_sentence_sent`, `t_first_audio`.

Stage values, which are the four colored segments on the page:

- `stt` = median over `control` turns of `t_final − t_eos` (pure transcription time)
- `vad` = median of `t_final − t_eos` for the config, minus `stt`, floored at 0
- `llm` = median of `t_first_sentence − t_final`
- `tts` = median of `t_first_audio − t_first_sentence_sent`
- `p50_total` and `p90_total` = percentiles of `t_first_audio − t_eos` over the config's turns

Stage medians do not sum exactly to the median total. Report both, and let the page show `p50_total` (section 16).

### T2. VAD silence sweep (P0)

- For each `min_silence_duration_ms` in `[150, 250, 500, 800]`, run 20 turns from `paused`.
- **Cut off**: a `completed` event arrives before `t_eos`. That means the agent would have answered half a question.
- **Latency**: `t_first_audio − t_eos` for turns that were not cut off, median.
- Output rows: `{ms, cut, p50, default}`, where `default` is true for 500.

### T3. Agent behavior checks (P0)

| id | Page label | Procedure | Pass rule | N |
| --- | --- | --- | --- | --- |
| `long_question` | Long question stays one turn | stream a `long` item with default VAD | exactly one `completed` event | 10 |
| `tool_valid` | Tool call valid inside the turn | from T1 `tool` turns | arguments parse and match the schema | 10 |
| `wer` | Transcript matches the script | final transcript vs. manifest text, lowercased, punctuation stripped | WER ≤ 5% | 10 |
| `openai_voice` | OpenAI voice names | OpenAI SDK `audio.speech.create` with `voice="alloy"` | HTTP 200 with audio | 3 |
| `clear_stops` | `input_text_buffer.clear` stops audio | see interruption procedure | no audio delta arrives more than 250 ms after the command | 10 |
| `cancel_stop` | `context.cancel` to silence | see interruption procedure | value check: report p50 time | 10 |

**Interruption procedure.** Use the same steps for `clear` and `cancel`.

1. Open TTS. For `cancel`, send every append with `context_id: "reply"`.
2. Append a 4-sentence reply (about 15 s of audio) and commit.
3. Send the command 500 ms after the first audio delta.
4. Keep reading until there are 2 s with no deltas.
5. Record these values:
   - `t_cmd`
   - `t_cancelled` (cancel only)
   - `t_last_delta`
   - `audio_after_cmd_ms`: total duration of chunks received after `t_cmd`
   - `buffered_at_cmd_ms`: audio already delivered but not yet played at `t_cmd`, taken from the playback simulator

Why `buffered_at_cmd_ms` matters: if TTS runs faster than real time, the rest of the sentence may already be on the client when the user interrupts. Then the server cannot stop it; the client must flush its own buffer. Report both numbers so the page states which side has to act.

### T4. TTS segment mode (P1)

- Record 20 LLM replies once, saving each token's text and relative timestamp.
- For each `segment` mode (`immediate`, `sentence`, `never`), replay the identical token streams into TTS with `input_text_buffer.append` at the original timing, then commit.
- Metrics per mode:
  - `ttfa`: median time from the first token sent to the first audio delta.
  - `stalls`: the number of replies with at least one playback stall (section 7.4).
- Replaying identical text and timing keeps the comparison fair across modes.

### T5. Concurrency (P2)

- Run the T1 `tuned` config with 1, 5, and 10 concurrent sessions, 10 turns each. Report `p90_total` per level.
- Stop on HTTP 429 and record the level where it happened.

### T6. Glue code size (P1)

- `pipeline.py` holds the working voice loop: STT session, turn handling, LLM stream, sentence splitting, tool loop, TTS session, interruption.
- Count non-blank, non-comment lines. The result goes to `compare.today.lines`.

---

## 10. Listen clips (P1)

Stitch real recorded audio at measured timings. Do not use simulated gaps. Export mono opus or mp3 under 200 KB each, plus a markers JSON for the page playhead.

| Clip | Composition |
| --- | --- |
| `vad_default` / `vad_tuned` | question audio, then silence equal to the measured `t_first_audio − t_eos` of the turn closest to p50, then the first 4 s of the reply |
| `cut_off` | a `paused` question that was cut off; the agent reply starts at its measured time, mixed over the user's second clause |
| `clear_vs_cancel` | reply audio; a short "wait, stop" overlay at `t_cmd`; the reply continues for the measured `audio_after_cmd_ms` (clear) or stops at the measured point (cancel) |

Markers format: `{"id": "vad_default", "src": "clips/vad_default.opus", "markers": [{"t_ms": 2310, "label": "you stop talking"}, {"t_ms": 3470, "label": "first audio"}]}`

---

## 11. Summarize and build the report

### 11.1 `summary.json` (written by the harness)

Keys must match the page's data block:

```json
{
  "sample": false,
  "target_ms": 1000,
  "target_label": "1 s",
  "turns": [
    {"label": "Default settings", "tag": "silence 500 ms",
     "p50": {"vad": 0, "stt": 0, "llm": 0, "tts": 0},
     "p50_total": 0, "p90": 0}
  ],
  "vad_sweep": {"n": 20, "rows": [{"ms": 500, "cut": 0, "p50": 0, "default": true}]},
  "segment": {"n": 20, "rows": [{"mode": "sentence", "ttfa": 0, "stalls": 0, "default": true}]},
  "checks": [
    {"id": "long_question", "name": "Long question stays one turn", "tag": "8 s of speech", "pass": 0, "n": 10},
    {"id": "cancel_stop", "name": "context.cancel to silence", "kind": "value", "value": "0 ms"}
  ],
  "compare": {"today": {"lines": 0}},
  "network": {"rtt_p50_ms": 0, "stt_connect_p50_ms": 0, "tts_connect_p50_ms": 0},
  "clips": [],
  "run": {"date": "Nov 5, 2026", "client_location": "", "models": {}, "git_sha": ""}
}
```

### 11.2 Traceability

Every number in `summary.json` carries the `run_id`s it was computed from, in a parallel `summary.provenance.json`. Any number on the page can then be traced back to raw events in an interview.

### 11.3 `content.json` (written by hand after reading the results)

This is the product judgment, so the harness does not generate it:

- `meta`: author, role, date, repo URL
- `summary`: one sentence
- `fixes`: three cards with title, why, do, target, and viz
- `works`: tags
- `footnote`: written by hand, plus `client_location` and the network numbers from the run
- `compare.today.code`, `compare.proposed`: code snippets and proposed line count
- check `note` text: written after seeing failures

### 11.4 `report.py`

1. Deep-merge `summary.json` into `content.json`; the harness values win on overlapping numeric fields.
2. Replace the contents of `<script id="report-data" type="application/json">…</script>` in `report/template.html`. Write the JSON with `ensure_ascii=False` and escape `</` as `<\/`.
3. Write `report.html`, and copy `results/clips/` next to it.
4. Fail the build if any required key is missing or if `sample` is still `true`.

---

## 12. CLI

```
voicebench corpus                     # generate and cache test audio
voicebench run t0 t1 t2 t3            # run selected experiments
voicebench run all --resume           # skip runs already in raw logs
voicebench clips
voicebench summarize
voicebench report
voicebench all                        # everything in order
voicebench run t1 --dry-run           # P2: mock servers with canned timings
```

Failures are logged and counted. They never crash the whole run.

---

## 13. Priorities and build order

| Priority | Item | Rough effort |
| --- | --- | --- |
| P0 | clock, pacer, event log; STT and TTS wrappers | 1.5 h |
| P0 | corpus | 0.5 h |
| P0 | T0, T1 | 1 h |
| P0 | T2 | 0.5 h (reuses T1) |
| P0 | T3 | 1 h |
| P0 | summarize, report | 0.5 h |
| P1 | T4, T6, clips | 1.5 h |
| P2 | T5, dry-run mock, local live demo | optional |

Build in this order: wrappers, corpus, T1, T2, T3, report. Then add P1 items if time allows. Total runtime for P0 is under 45 minutes.

Volume estimate for a full P0 run: about 200 STT turns (about 15 minutes of streamed audio including silence), about 300 short LLM completions, and about 150 TTS replies. Check current pricing before running.

---

## 14. Verify first (first 30 minutes)

- [ ] Run each event name in section 6 once, printing raw messages.
- [ ] `completed` arrives without `commit` when VAD is on, and arrives after `commit` when VAD is off.
- [ ] `transcription_session.updated` changes VAD mid-session. If not, reconnect per config.
- [ ] TTS accepts `sample_rate=16000`, which would skip resampling for the corpus.
- [ ] `context.cancel` works on the `"default"` context without passing `context_id` on appends. The result belongs on the page either way.
- [ ] TTS delivery speed vs. real time; this decides how to read `buffered_at_cmd_ms`.
- [ ] The streaming variant of HTTP TTS: exact parameter and response framing.
- [ ] Realtime session limits and rate limits; note any 429s.
- [ ] Optionally run once from a US cloud VM and compare with home Wi-Fi, to show how much is network.

---

## 15. Acceptance criteria

- `voicebench all` runs end to end on a fresh clone with only `TOGETHER_API_KEY` set.
- `report.html` builds with `sample: false`, renders, and contains no placeholder numbers.
- Every number on the page appears in `summary.json` and traces to `run_id`s in the raw logs.
- The README states the method: corpus, N, warm-up, client location, how stages are computed, and known limits.
- No API key appears in the repo, logs, or report.

---

## 16. Frontend follow-ups (small)

- Show `p50_total` as the total when present, instead of the sum of stage medians.
- Add a Listen section: play buttons for the clips, with a playhead that moves along the matching latency bar using the markers.
- In the footnote, add the network baseline and client location.
- Show the `buffered_at_cmd_ms` result in the interruption check note.
