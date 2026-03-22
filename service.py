"""
MicrophoneService — full verbose logging
"""
import asyncio
import threading
import time
import traceback
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


# ── Shared logger ─────────────────────────────────────────────────────────────
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [{SERVICE_NAME}] {msg}", flush=True)


class MicrophoneService:
    def __init__(self):
        log("🎤 Initializing …")
        self._shutting_down     = False
        self._shutdown_lock     = threading.Lock()
        self._hub_emit_count    = 0
        self._ws_broadcast_count = 0

        # ── WebSocket server ──────────────────────────────────────────────────
        log("Creating WebSocket broadcast server …")
        self.ws_server = WebSocketServer()

        # ── Socket.IO hub client ──────────────────────────────────────────────
        self.sio      = socketio.AsyncClient(reconnection=True, reconnection_delay=5)
        self.hub_loop: asyncio.AbstractEventLoop | None = None

        @self.sio.event
        async def connect():
            log(f"✅ Hub CONNECTED → {HUB_URL}")

        @self.sio.event
        async def disconnect():
            log("⚠️  Hub DISCONNECTED")

        @self.sio.event
        async def connect_error(data):
            log(f"❌ Hub CONNECTION ERROR: {data}")

        # ── Transcriber ───────────────────────────────────────────────────────
        log(f"Initializing MicrophoneTranscriber (device_id={MICROPHONE_DEVICE_ID}) …")
        try:
            self.transcriber = MicrophoneTranscriber(
                keep_files=False,
                device_id=MICROPHONE_DEVICE_ID,
            )
            self.transcriber.set_volume_callback(self._on_volume)
            log("✅ MicrophoneTranscriber ready")
        except Exception as e:
            log(f"❌ MicrophoneTranscriber init FAILED: {e}")
            log(traceback.format_exc())
            raise

        self._polling_active = True

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self):
        self.hub_loop = asyncio.new_event_loop()
        threading.Thread(target=self._hub_thread, args=(self.hub_loop,), daemon=True, name="MicHub").start()
        log("Hub thread started")

        self.ws_server.start()
        log("WebSocket server started")

        threading.Thread(target=self.transcriber.run, daemon=True, name="MicTranscriber").start()
        log(f"MicrophoneTranscriber started — listening on device {MICROPHONE_DEVICE_ID} …")

        threading.Thread(target=self._poll_loop, daemon=True, name="MicPoller").start()
        log("Poll loop started")

        log("✅ All components running — speak into the mic …")

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
        log(f"🛑 Shutting down (hub emits: {self._hub_emit_count}, WS broadcasts: {self._ws_broadcast_count})")
        self._polling_active = False
        try:
            self.transcriber.stop()
        except Exception as e:
            log(f"Error stopping transcriber: {e}")
        try:
            self.ws_server.stop()
        except Exception as e:
            log(f"Error stopping WS server: {e}")
        if self.hub_loop:
            self.hub_loop.call_soon_threadsafe(self.hub_loop.stop)
        log("🛑 Stopped.")

    # ── Hub ───────────────────────────────────────────────────────────────────

    def _hub_thread(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.create_task(self._hub_connection_loop())
        loop.run_forever()

    async def _hub_connection_loop(self):
        while not self._shutting_down:
            if not self.sio.connected:
                try:
                    log(f"Attempting hub connect → {HUB_URL} …")
                    await self.sio.connect(HUB_URL)
                except Exception as e:
                    log(f"⚠️  Hub connect failed: {e} — retry in 5s")
                    await asyncio.sleep(5)
            await asyncio.sleep(2)

    def _emit_to_hub(self, event: str, data: dict):
        if not self.sio.connected:
            log(f"⚠️  SKIPPED hub emit (not connected): {event}")
            return
        if not self.hub_loop:
            log(f"⚠️  SKIPPED hub emit (no loop): {event}")
            return
        try:
            asyncio.run_coroutine_threadsafe(self.sio.emit(event, data), self.hub_loop)
            self._hub_emit_count += 1
            log(f"→ HUB [{event}] {str(data)[:160]}")
        except Exception as e:
            log(f"❌ HUB EMIT ERROR [{event}]: {e}")
            log(traceback.format_exc())

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_volume(self, level: float):
        # self.ws_server.broadcast({"type": "volume", "source": "mic", "level": level})
        pass

    # ── Poll loop ─────────────────────────────────────────────────────────────

    def _poll_loop(self):
        log("Poll loop active — waiting for transcriber results …")
        while self._polling_active:
            try:
                result = self.transcriber.result_queue.get(timeout=0.1)
                log(f"📥 Raw queue result: {result}")

                text, _filename, source, confidence = result

                if not text or not text.strip():
                    log(f"  ↳ Empty transcript skipped (source={source})")
                    continue

                transcript_id = str(uuid.uuid4())
                log(f"🎤 TRANSCRIPT | source={source} confidence={confidence:.3f} | text={repr(text)}")

                payload = {
                    "type":       "transcript",
                    "source":     "microphone",
                    "speaker":    "User",
                    "text":       text,
                    "confidence": confidence,
                    "id":         transcript_id,
                    "timestamp":  time.time(),
                }

                # WebSocket broadcast
                try:
                    self.ws_server.broadcast(payload)
                    self._ws_broadcast_count += 1
                    log(f"→ WS broadcast #{self._ws_broadcast_count} sent")
                except Exception as e:
                    log(f"❌ WS BROADCAST ERROR: {e}")
                    log(traceback.format_exc())

                # Hub events
                self._emit_to_hub("spoken_word_context", {"context": text, "timestamp": payload["timestamp"]})

            except Empty:
                continue
            except Exception as e:
                log(f"❌ POLL LOOP EXCEPTION: {e}")
                log(traceback.format_exc())
                time.sleep(0.1)

    # ── Device hot-swap ───────────────────────────────────────────────────────

    def swap_device(self, device_id: int):
        """Stop the current transcriber and restart it on a new device."""
        import config
        log(f"🔄 Device swap requested → device_id={device_id}")

        # Stop existing transcriber
        try:
            self.transcriber.stop()
            log("  ↳ Old transcriber stopped")
        except Exception as e:
            log(f"  ↳ Error stopping old transcriber: {e}")

        # Update config in-memory
        config.MICROPHONE_DEVICE_ID = device_id

        # Create and start new transcriber
        try:
            from transcriber_core.microphone import MicrophoneTranscriber
            self.transcriber = MicrophoneTranscriber(
                keep_files=False,
                device_id=device_id,
            )
            self.transcriber.set_volume_callback(self._on_volume)
            threading.Thread(target=self.transcriber.run, daemon=True, name="MicTranscriber").start()
            log(f"✅ New transcriber started on device {device_id}")
        except Exception as e:
            log(f"❌ Failed to start transcriber on device {device_id}: {e}")
            log(traceback.format_exc())