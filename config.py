"""
Configuration for the Microphone Audio Service.
Handles real-time microphone transcription via Parakeet MLX.
"""
import os
import sys

# ── Path bootstrap: pull shared modules from desktop_mon_gemini ──────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR  = os.path.dirname(_THIS_DIR)
_DESKTOP_DIR = os.path.join(_PARENT_DIR, "desktop_mon_gemini")

for _p in (_DESKTOP_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

# Audio
MICROPHONE_DEVICE_ID = 5
AUDIO_SAMPLE_RATE    = 16000

# Network
WEBSOCKET_PORT    = 8013   # WebSocket broadcast port
HTTP_CONTROL_PORT = 8014   # HTTP health-check / shutdown port
HUB_URL           = "http://localhost:8002"

# Service identity (used in hub events / logs)
SERVICE_NAME = "microphone_audio_service"