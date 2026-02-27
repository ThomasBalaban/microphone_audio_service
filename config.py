"""
Configuration for the Microphone Audio Service.
"""
import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Adds THIS service directory first, then the parent projects folder.
# transcriber_core/ can live in EITHER location:
#   - /projects/microphone_audio_service/transcriber_core/   (own dir)
#   - /projects/transcriber_core/                            (parent dir)
# api_keys.py should be in this service's own folder.
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

print(f"[config] sys.path[0:3] = {sys.path[:3]}", flush=True)

try:
    from api_keys import GEMINI_API_KEY, OPENAI_API_KEY  # noqa: F401
    print("[config] ✅ api_keys loaded", flush=True)
except ImportError as e:
    print(f"[config] ❌ api_keys FAILED: {e}", flush=True)
    raise

try:
    import transcriber_core  # noqa: F401
    print(f"[config] ✅ transcriber_core found at: {transcriber_core.__file__}", flush=True)
except ImportError as e:
    print(f"[config] ❌ transcriber_core NOT FOUND: {e}", flush=True)
    print(f"[config]    Searched in: {_THIS_DIR}", flush=True)
    print(f"[config]              and {_PARENT_DIR}", flush=True)
    print(f"[config]    Copy transcriber_core/ into one of those folders.", flush=True)
    raise

# Audio
MICROPHONE_DEVICE_ID = 5
AUDIO_SAMPLE_RATE    = 16000

# Network
WEBSOCKET_PORT    = 8013
HTTP_CONTROL_PORT = 8014
HUB_URL           = "http://localhost:8002"

SERVICE_NAME = "microphone_audio_service"