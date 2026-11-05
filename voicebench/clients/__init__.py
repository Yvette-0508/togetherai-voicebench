"""Shared helpers for the raw-protocol clients."""
from __future__ import annotations

import websockets


async def ws_connect(url: str, headers: dict, **kwargs):
    """websockets.connect with a headers arg that works across library versions."""
    import inspect
    sig = inspect.signature(websockets.connect)
    if "additional_headers" in sig.parameters:
        return await websockets.connect(url, additional_headers=headers, **kwargs)
    return await websockets.connect(url, extra_headers=headers, **kwargs)
