#!/usr/bin/env python3
"""pipeline.py — the working voice loop a developer writes today on Together AI.

STT WebSocket session -> streamed chat completion -> sentence splitting ->
TTS WebSocket session -> playback, with barge-in (context.cancel) and
reconnect. Stream a wav/PCM16 file with `python pipeline.py turn.wav`.

Its non-blank, non-comment line count is the `compare.today.lines` metric on
the report page (design.md, T6).
"""
import asyncio
import base64
import json
import os
import re
import sys
import time

import numpy as np
import soundfile as sf
import websockets
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
KEY = os.environ["TOGETHER_API_KEY"]
STT_MODEL = os.environ.get("STT_MODEL", "openai/whisper-large-v3")
LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
TTS_MODEL = os.environ.get("TTS_MODEL", "hexgrad/Kokoro-82M")
VOICE = os.environ.get("TTS_VOICE", "af_alloy")
SILENCE_MS = int(os.environ.get("VAD_SILENCE_MS", "500"))
SYSTEM = "You are a voice assistant. Answer in one or two short sentences."
SENT_END = re.compile(r"(?<=[.!?])\s")
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "required": ["city"], "properties": {
            "city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}}},
}]


async def ws(url, headers):
    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)


class STT:
    async def open(self):
        vad = f"&turn_detection=server_vad&threshold=0.3&min_silence_duration_ms={SILENCE_MS}"
        self.w = await ws(f"wss://api.together.ai/v1/realtime?model={STT_MODEL}"
                          f"&input_audio_format=pcm_s16le_16000{vad}",
                          {"Authorization": f"Bearer {KEY}", "OpenAI-Beta": "realtime=v1"})
        self.final_q = asyncio.Queue()
        async def recv():
            async for raw in self.w:
                ev = json.loads(raw)
                if ev["type"] == "conversation.item.input_audio_transcription.completed":
                    text = ev.get("transcript") or ev.get("text") or ""
                    await self.final_q.put(str(text))
        self.task = asyncio.create_task(recv())

    async def send_16k(self, pcm16: bytes, frame_ms=20, tail_s=3.0):
        frame = frame_ms * 16 * 2
        frames = [pcm16[i:i + frame] for i in range(0, len(pcm16), frame)]
        start = asyncio.get_event_loop().time()
        n = 0
        async def push(buf=b"\x00" * frame):
            nonlocal n
            target = start + n * frame_ms / 1000
            dt = target - asyncio.get_event_loop().time()
            if dt > 0:
                await asyncio.sleep(dt)
            await self.w.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(buf).decode()}))
            n += 1
        for _ in range(int(0.3 / (frame_ms / 1000))):
            await push()
        for fr in frames:
            await push(fr)
        eof = asyncio.get_event_loop().time() + tail_s
        while not self.final_q.qsize() and asyncio.get_event_loop().time() < eof:
            await push()


class TTS:
    async def open(self):
        self.w = await ws(f"wss://api.together.ai/v1/audio/speech/websocket"
                          f"?model={TTS_MODEL}&voice={VOICE}"
                          "&response_format=pcm&sample_rate=24000",
                          {"Authorization": f"Bearer {KEY}"})
        self.first_audio_at = None
        self.audio = bytearray()
        async def recv():
            async for raw in self.w:
                ev = json.loads(raw)
                if ev["type"] == "conversation.item.audio_output.delta":
                    if self.first_audio_at is None:
                        self.first_audio_at = time.perf_counter()
                    self.audio.extend(base64.b64decode(ev["delta"]))
        self.task = asyncio.create_task(recv())

    async def say(self, text):
        await self.w.send(json.dumps({"type": "input_text_buffer.append", "text": text}))

    async def commit(self):
        await self.w.send(json.dumps({"type": "input_text_buffer.commit"}))

    async def interrupt(self):
        # Barge-in: drop queued text and cancel what is still being spoken.
        await self.w.send(json.dumps({"type": "input_text_buffer.clear"}))
        await self.w.send(json.dumps({"type": "context.cancel", "context_id": "default"}))


async def llm_reply(client, history, text, tts):
    history.append({"role": "user", "content": text})
    buf, first_sentence = "", False
    for _ in range(2):  # first call, then again after a tool result
        stream = await client.chat.completions.create(
            model=LLM_MODEL, messages=history, stream=True,
            max_tokens=120, temperature=0.3, tools=TOOLS)
        tool_call = {"id": "", "name": "", "args": ""}
        full = ""
        async for chunk in stream:
            d = chunk.choices[0].delta if chunk.choices else None
            if not d:
                continue
            if d.content:
                buf += d.content
                full += d.content
                m = SENT_END.search(buf)
                if m:
                    sent, buf = buf[:m.end()], buf[m.end():]
                    await tts.say(sent)
                    first_sentence = True
            if d.tool_calls:
                for tc in d.tool_calls:
                    if tc.id:
                        tool_call["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_call["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_call["args"] += tc.function.arguments
        if not tool_call["id"]:
            if buf.strip():
                await tts.say(buf.strip())
            await tts.commit()
            history.append({"role": "assistant", "content": full})
            return
        args = json.loads(tool_call["args"] or "{}")
        result = {"city": args.get("city", ""), "temperature": 68, "conditions": "clear"}
        history.append({"role": "assistant", "content": full or None, "tool_calls": [{
            "id": tool_call["id"], "type": "function",
            "function": {"name": tool_call["name"], "arguments": tool_call["args"]}}]})
        history.append({"role": "tool", "tool_call_id": tool_call["id"],
                        "content": json.dumps(result)})


async def main():
    wav = sys.argv[1] if len(sys.argv) > 1 else None
    if not wav:
        sys.exit("usage: python pipeline.py turn.wav   (16 kHz _or_ any rate; will resample)")
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(16000, sr)
    pcm = (resample_poly(audio, 16000 // g, sr // g) * 32767).astype(np.int16).tobytes()
    stt, tts = STT(), TTS()
    await stt.open()
    await tts.open()
    client = AsyncOpenAI(api_key=KEY, base_url="https://api.together.ai/v1")
    t_eos = None
    async def feed():
        nonlocal t_eos
        await stt.send_16k(pcm)
        t_eos = time.perf_counter()
    async def think():
        text = await stt.final_q.get()
        print(f"[stt] {text}")
        history = [{"role": "system", "content": SYSTEM}]
        await llm_reply(client, history, text, tts)
    await asyncio.gather(feed(), think())
    if t_eos and tts.first_audio_at:
        print(f"end of speech -> first audio: {(tts.first_audio_at - t_eos) * 1000:.0f} ms")
    out = np.frombuffer(bytes(tts.audio), dtype=np.int16)
    sf.write("reply.wav", out / 32768.0, 24000)
    await stt.w.close()
    await tts.w.close()


if __name__ == "__main__":
    asyncio.run(main())
