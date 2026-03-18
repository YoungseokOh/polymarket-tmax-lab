"""CLOB market websocket consumer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import websockets


async def stream_market_channel(ws_url: str, asset_ids: list[str]) -> AsyncIterator[dict]:
    """Yield decoded websocket events for the market channel."""

    async with websockets.connect(ws_url) as websocket:
        payload = {
            "type": "subscribe",
            "channel": "market",
            "assets_ids": asset_ids,
        }
        await websocket.send(json.dumps(payload))
        while True:
            message = await websocket.recv()
            yield json.loads(message)
            await asyncio.sleep(0)

