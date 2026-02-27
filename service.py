"""
MicrophoneService
─────────────────
• Runs MicrophoneTranscriber (Parakeet MLX) in a background thread
• Polls the result queue and broadcasts transcripts to:
    - The central Hub  (Socket.IO events: spoken_word_context, audio_context)
    - Local WebSocket clients  (port 8013)
• Exposes volume-meter data via the same WebSocket feed
"""

import asyncio
import threading
import time
import uuid
from queue import Empty

import socketio

from config import (
    HUB_URL,
    MICROPHONE_DEVICE_ID,
    SERVICE_NAME,
)
from websocket_server import WebSocketServer

from transcriber_core.microphone import MicrophoneTranscriber


class MicrophoneService:
    def __init__(self):
        print(f"🎤 [{SERVICE_NAME}] Initializing …")

        self._shutting_down = False
        self._shutdown_lock = threading.Lock()

        # ── WebSocket broadcast server ───────────────────────────────────────
        self.ws_server = WebSocketServer()

        # ── Socket.IO hub client ─────────────────────────────────────────────
        self.sio = socketio.AsyncClient(reconnection=True, reconnection_delay=5)
        self.hub_loop: asyncio.AbstractEventLoop | None = None

        # ── Microphone transcriber ───────────────────────────────────────────
        self.transcriber = MicrophoneTranscriber(
            keep_files=False,
            device_id=MICROPHONE_DEVICE_ID,
        )
        # Forward volume updates over WebSocket
        self.transcriber.set_volume_callback(self._on_volume)

        self._polling_active = True

    # ── Public ───────────────────────────────────────────────────────────────

    def run(self):
        """Start all components; blocks until stop() is called."""
        # Hub event-loop in background thread
        self.hub_loop = asyncio.new_event_loop()
        threading.Thread(
            target=self._hub_thread, args=(self.hub_loop,), daemon=True, name="MicHub"
        ).start()

        # WebSocket server
        self.ws_server.start()

        # Microphone transcriber
        threading.Thread(target=self.transcriber.run, daemon=True, name="MicTranscriber").start()

        # Result-queue poller
        threading.Thread(target=self._poll_loop, daemon=True, name="MicPoller").start()

        print(f"✅ [{SERVICE_NAME}] All components running. Press Ctrl-C to stop.")

        # Block main thread until shutdown
        try:
            while not self._shutting_down:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        with self._shutdown_lock:
            if self._shutting_down:
                return
            self._shutting_down = True

        print(f"🛑 [{SERVICE_NAME}] Shutting down …")
        self._polling_active = False

        try:
            self.transcriber.stop()
        except Exception:
            pass

        try:
            self.ws_server.stop()
        except Exception:
            pass

        if self.hub_loop:
            self.hub_loop.call_soon_threadsafe(self.hub_loop.stop)

        print(f"🛑 [{SERVICE_NAME}] Stopped.")

    # ── Hub helpers ──────────────────────────────────────────────────────────

    def _hub_thread(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.create_task(self._hub_connection_loop())
        loop.run_forever()

    async def _hub_connection_loop(self):
        while not self._shutting_down:
            if not self.sio.connected:
                try:
                    await self.sio.connect(HUB_URL)
                    print(f"✅ [{SERVICE_NAME}] Hub connected: {HUB_URL}")
                except Exception as e:
                    print(f"⚠️  [{SERVICE_NAME}] Hub connect failed: {e}. Retrying in 5s …")
                    await asyncio.sleep(5)
            await asyncio.sleep(2)

    def _emit_to_hub(self, event: str, data: dict):
        """Thread-safe hub emit."""
        if self.sio.connected and self.hub_loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.sio.emit(event, data), self.hub_loop
                )
            except Exception as e:
                print(f"❌ [{SERVICE_NAME}] Hub emit error: {e}")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_volume(self, level: float):
        """Forward volume level to WebSocket clients."""
        self.ws_server.broadcast({"type": "volume", "source": "mic", "level": level})

    # ── Polling loop ─────────────────────────────────────────────────────────

    def _poll_loop(self):
        """Pull transcripts from the transcriber queue and distribute them."""
        while self._polling_active:
            try:
                text, _filename, source, confidence = self.transcriber.result_queue.get(timeout=0.1)

                if not text or not text.strip():
                    continue

                transcript_id = str(uuid.uuid4())
                payload = {
                    "type":       "transcript",
                    "source":     "microphone",
                    "speaker":    "User",
                    "text":       text,
                    "confidence": confidence,
                    "id":         transcript_id,
                    "timestamp":  time.time(),
                }

                # ── Local WebSocket clients ──────────────────────────────────
                self.ws_server.broadcast(payload)

                # ── Hub events ───────────────────────────────────────────────
                # spoken_word_context → fills 'Spoken' column in Director UI
                self._emit_to_hub("spoken_word_context", {"context": text})

                # audio_context → routed to Director Engine / general logs
                self._emit_to_hub("audio_context", {
                    "context":    text,
                    "is_partial": False,
                    "metadata":   {
                        "source":     "microphone",
                        "confidence": confidence,
                        "id":         transcript_id,
                    },
                })

                print(f"🎤 [{SERVICE_NAME}] → {text}")

            except Empty:
                continue
            except Exception as e:
                print(f"⚠️  [{SERVICE_NAME}] Poll error: {e}")
                time.sleep(0.1)