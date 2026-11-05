"""Corpus generation (design.md section 8).

Generates test questions as audio with Together TTS over HTTP, caches 16 kHz
16-bit mono WAVs in results/corpus/ with a manifest.json recording text,
duration, and pause positions.

Known limit (stated in README): synthetic speech is cleaner than real users,
so accuracy numbers from this corpus are optimistic.
"""
from __future__ import annotations

import asyncio
import io
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from .clients.tts import http_tts
from .clock import RESULTS

CORPUS_DIR = RESULTS / "corpus"

# [PAUSE=ms] inserts silence mid-sentence (built after synthesis, since TTS
# won't produce a reliable long pause on demand).
SHORT = [
    "What's the capital of Japan?",
    "How do I reset my password?",
    "What time does the store close on Saturdays?",
    "Can you explain what an API is in one sentence?",
    "What's the difference between a bagel and a donut?",
    "How many days are there in February this year?",
    "What should I cook for dinner with eggs and spinach?",
    "Is it going to rain this weekend?",
    "How do I get to the airport from downtown?",
    "What's a good book to read on a long flight?",
    "Why is the sky blue?",
    "How do I convert cups to grams for flour?",
    "What's the fastest way to learn a language?",
    "Can you recommend a good coffee shop near the office?",
    "What does the red light on my router mean?",
    "How often should I water a snake plant?",
    "What's the best time to visit Yosemite?",
    "How do I unsubscribe from a mailing list?",
    "What's the exchange rate between dollars and euros right now?",
    "Can you give me a simple pancake recipe?",
    "Why do cats purr?",
    "How long does it take to boil pasta?",
    "What's the battery life of a typical laptop?",
    "How do I take a screenshot on a Mac?",
]

PAUSED = [
    "I need to book a table [PAUSE=450] for four people tonight.",
    "I was wondering [PAUSE=350] if you could help me move on Saturday.",
    "The meeting got moved to [PAUSE=550] actually, let me check, three o'clock.",
    "My flight lands at noon, [PAUSE=400] so can I still make the two o'clock tour?",
    "I tried restarting the router [PAUSE=500] but the light is still red.",
    "We should probably leave by [PAUSE=380] seven if we want good seats.",
    "He's allergic to peanuts, [PAUSE=420] and I think eggs too.",
    "I bought the tickets yesterday [PAUSE=600] but I haven't got the email.",
    "Can you add the eggs [PAUSE=340] and also pick up some milk?",
    "The doctor said to take two pills [PAUSE=480] twice a day with food.",
    "I'd like to return this jacket, [PAUSE=370] I bought it last week.",
    "The kids loved the museum, [PAUSE=520] especially the dinosaur hall.",
    "My train was delayed an hour [PAUSE=410] so I missed the connection.",
    "Could you water the plants [PAUSE=460] and maybe feed the cat too?",
    "The recipe says to simmer [PAUSE=390] but doesn't say how long.",
    "I want to learn guitar, [PAUSE=430] do you have any beginner tips?",
    "The hotel has a pool, [PAUSE=570] it's open until ten at night.",
    "I locked my keys in the car [PAUSE=360] and my spare is at home.",
    "She likes mystery novels, [PAUSE=440] anything by Agatha Christie.",
    "My laptop won't charge, [PAUSE=510] I already tried a different cable.",
    "We need more chairs for dinner [PAUSE=400] since six people are coming.",
    "The package says it arrived [PAUSE=470] but nothing is at my door.",
    "I'd like a medium latte, [PAUSE=350] and one of those croissants.",
    "The wifi password changed, [PAUSE=540] do you know the new one?",
]

LONG = [
    "I'm planning a trip to Portugal next spring with my partner, probably about ten days, and we want to mix Lisbon and Porto with maybe some smaller towns, so what would you suggest for an itinerary that doesn't feel rushed?",
    "My seven-year-old has gotten really into space recently and wants to be an astronaut, so I'm looking for books, documentaries, maybe a telescope, and any local events or museums we could visit in the next few months, what do you recommend?",
    "I run a small bakery and I've been thinking about offering delivery, but I'm worried about the cost of drivers versus apps, the packaging to keep things fresh, and whether it's even worth it at our scale, can you walk me through the options?",
    "Our team is debating whether to rewrite our old internal dashboard or keep patching it, considering we have two developers, about three months, and the current version mostly works but is impossible to extend, what's a sensible way to decide?",
    "I've been getting knee pain when I run more than five kilometers, nothing at rest, and I've had the same shoes for about a year and a half, so before I see a doctor I wanted to know the common causes and what I could try at home?",
    "We're hosting Thanksgiving for twelve adults and three kids, two of whom are vegetarian and one of whom can't have gluten, and I'd love a menu that doesn't mean cooking three separate dinners, can you suggest one?",
    "My apartment gets really hot in the afternoon because of west-facing windows, AC isn't an option in the lease, and fans just move hot air around, so what are some actually effective ways of keeping the room cool?",
    "I inherited a houseplant collection and I don't know what any of them are, some are dropping leaves and others seem fine, the light in my place is mostly indirect, so how do I figure out what I have and keep it alive?",
    "Our startup is choosing between a job board, referrals, and a recruiter for our first five engineering hires next quarter, budget is tight but speed matters more, can you compare the three options?",
    "The guitar I bought secondhand buzzes on the third and fourth frets of the low E string no matter how I tune it, and I've already changed the strings, so what should I check before paying for a setup?",
    "I want to start composting in a small apartment without a balcony, and I've heard about worm bins and bokashi, but I'm worried about smell and fruit flies, which method would you pick and why?",
    "My dog barks at the doorbell and it's getting worse with package deliveries every day, I've tried ignoring it for two weeks without success, so what's the right training approach for a noise-shy rescue?",
]

TOOL = [
    "What's the weather like in Seattle right now?",
    "Is it raining in London at the moment?",
    "How warm is it in Miami today?",
    "Do I need a jacket in Chicago this evening?",
    "What's the temperature in Phoenix?",
    "Will I need an umbrella in Dublin today?",
    "How's the weather in Tokyo right now?",
    "Is it cold in Denver tonight?",
    "What's the forecast temperature in Austin today?",
    "Should I wear sunscreen in Sydney today?",
    "How cold does it get in Oslo right now?",
    "What's the weather doing in San Diego?",
    "Is it windy in Wellington today?",
    "What's the temperature in Toronto at the moment?",
    "Do I need a hat and gloves in Boston tonight?",
    "How hot is it in Cairo today?",
    "Is it humid in Singapore right now?",
    "What's the weather like in Reykjavik?",
    "Should I bring a raincoat to Portland today?",
    "How warm is Barcelona this afternoon?",
    "What's the current temperature in Nairobi?",
    "Is it snowing in Denver right now?",
    "What's the weather in Auckland today?",
    "Do I need an umbrella in Amsterdam tonight?",
]

SETS = {"short": SHORT, "paused": PAUSED, "long": LONG, "tool": TOOL}
PAUSE_RE = re.compile(r"\[PAUSE=(\d+)\]")


async def synthesize_one(api_key: str, model: str, voice: str, text: str,
                         out_path: Path, pauses: list[tuple[int, int]],
                         stt_rate: int, tts_rate: int) -> dict:
    """Synthesize text at tts_rate, insert pauses, resample to stt_rate, save."""
    from scipy.signal import resample_poly

    wav = await http_tts(api_key, model, voice, text, sample_rate=tts_rate)
    audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Normalize duration split for pauses proportionally to character position.
    plain = PAUSE_RE.sub("", text)
    full_text = text
    parts, positions_ms = [], []
    cursor = 0
    for m in PAUSE_RE.finditer(text):
        parts.append(text[cursor:m.start()])
        positions_ms.append(int(m.group(1)))
        cursor = m.end()
    parts.append(full_text[cursor:])
    if len(parts) > 1:
        # Split the waveform proportionally to spoken character counts.
        spoken_chars = sum(len(PAUSE_RE.sub("", p)) for p in parts)
        splits, offset = [], 0
        cuts = np.cumsum([len(PAUSE_RE.sub("", p)) for p in parts[:-1]])
        sample_cuts = [int(round(c / spoken_chars * len(audio))) for c in cuts]
        chunks = np.split(audio, sample_cuts)
        segs = [chunks[0]]
        for pause_ms, ch in zip(positions_ms, chunks[1:]):
            segs.append(np.zeros(int(sr * pause_ms / 1000), dtype="float32"))
            segs.append(ch)
        audio = np.concatenate(segs)
    if sr != stt_rate:
        # e.g. 24000 -> 16000: resample_poly(x, 2, 3)
        from math import gcd
        g = gcd(stt_rate, sr)
        audio = resample_poly(audio, stt_rate // g, sr // g).astype("float32")
    pcm16 = np.clip(audio, -1.0, 1.0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, pcm16, stt_rate, subtype="PCM_16")
    duration_ms = int(len(pcm16) / stt_rate * 1000)
    return {"duration_ms": duration_ms}


async def generate(cfg: dict, force: bool = False):
    """Generate and cache the corpus. Idempotent: skips existing files."""
    from .config import require_key
    require_key(cfg)
    model, voice = cfg["models"]["tts"], cfg["models"]["voice"]
    stt_rate = cfg["audio"]["stt_sample_rate"]
    tts_rate = cfg["audio"]["tts_sample_rate"]
    manifest_path = CORPUS_DIR / "manifest.json"
    manifest: dict = {"sets": {}}
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("sample_rate", stt_rate)

    for set_name, texts in SETS.items():
        entries: list[dict] = manifest["sets"].get(set_name, [])
        for i, text in enumerate(texts):
            out_path = CORPUS_DIR / set_name / f"{i:03d}.wav"
            pauses = [[m.start(), int(m.group(1))] for m in PAUSE_RE.finditer(text)]
            plain = PAUSE_RE.sub("", text)
            if i < len(entries) and out_path.exists() and not force:
                continue
            print(f"  corpus {set_name}/{i:03d}: {plain[:55]}...")
            try:
                meta = await synthesize_one(cfg["api_key"], model, voice, plain,
                                            out_path, pauses, stt_rate, tts_rate)
            except Exception as e:  # failures are logged, never crash the run
                print(f"    FAILED: {e}")
                continue
            entry = {
                "id": f"{set_name}-{i:03d}", "set": set_name, "file": str(out_path.relative_to(CORPUS_DIR.parent)),
                "text": plain, "duration_ms": meta["duration_ms"], "pauses": pauses,
            }
            if i < len(entries):
                entries[i] = entry
            else:
                entries.append(entry)
        manifest["sets"][set_name] = entries
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {manifest_path}")


def load_manifest() -> dict:
    return json.loads((CORPUS_DIR / "manifest.json").read_text())


def item_pcm(set_name: str, idx: int) -> tuple[bytes, dict]:
    """Load one corpus item as 16-bit PCM bytes + its manifest entry."""
    manifest = load_manifest()
    entry = manifest["sets"][set_name][idx]
    data, _sr = sf.read(CORPUS_DIR / set_name / f"{idx:03d}.wav", dtype="int16")
    return data.tobytes(), entry
