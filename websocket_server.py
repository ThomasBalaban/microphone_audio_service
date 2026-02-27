"""
Lightweight WebSocket broadcast server for the Microphone Audio Service.
Clients connect here to receive real-time transcription events.
"""

import asyncio
import json
import queue
import threading
import time
import websockets

from config import WEBSOCKET_PORT


class WebSocketServer:
    def __init__(self):
        self.connected_clients: set = set()
        self.message_queue: queue.Queue = queue.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.running = True

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self):
        t = threading.Thread(target=self._run_in_thread, daemon=True, name="MicWS")
        t.start()
        print(f"🔌 [MicWS] WebSocket server starting on ws://localhost:{WEBSOCKET_PORT}")

    def stop(self):
        self.running = False
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

    def broadcast(self, data: dict):
        """Thread-safe broadcast to all connected clients."""
        try:
            self.message_queue.put_nowait(data)
        except queue.Full:
            print("⚠️  [MicWS] Message queue full, dropping message")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run_in_thread(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        finally:
            self.loop.close()

    async def _serve(self):
        asyncio.create_task(self._queue_processor())
        asyncio.create_task(self._heartbeat())
        async with websockets.serve(self._handler, "0.0.0.0", WEBSOCKET_PORT):
            await asyncio.Future()  # run forever

    async def _handler(self, ws, path=None):
        self.connected_clients.add(ws)
        print(f"🔌 [MicWS] Client connected (total: {len(self.connected_clients)})")
        try:
            await ws.send(json.dumps({
                "type": "connection_established",
                "service": "microphone_audio_service",
                "timestamp": time.time(),
            }))
            async for msg in ws:
                try:
                    data = json.loads(msg)
                    if data.get("type") == "ping":
                        await ws.send(json.dumps({"type": "pong", "timestamp": time.time()}))
                except Exception:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.discard(ws)
            print(f"🔌 [MicWS] Client disconnected (total: {len(self.connected_clients)})")

    async def _queue_processor(self):
        while self.running:
            try:
                try:
                    data = self.message_queue.get_nowait()
                    await self._do_broadcast(data)
                except queue.Empty:
                    pass
            except Exception as e:
                print(f"⚠️  [MicWS] Queue processor error: {e}")
            await asyncio.sleep(0.01)

    async def _heartbeat(self):
        while self.running:
            await asyncio.sleep(5)
            if self.connected_clients:
                await self._do_broadcast({"type": "heartbeat", "timestamp": time.time()})

    async def _do_broadcast(self, data: dict):
        if not self.connected_clients:
            return
        msg = json.dumps(data)
        dead = set()
        for client in self.connected_clients.copy():
            try:
                await client.send(msg)
            except Exception:
                dead.add(client)
        for c in dead:
            self.connected_clients.discard(c)