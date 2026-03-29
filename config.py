"""
Configuration for the Microphone Audio Service.
"""
import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
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
AUDIO_SAMPLE_RATE      = 16000

_PREFERRED_DEVICE_NAME = "Scarlett Solo 4th Gen"
_FALLBACK_DEVICE_ID    = 5

def _find_device_by_name(name: str) -> int | None:
    try:
        import sounddevice as sd # type: ignore
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and name.lower() in dev["name"].lower():
                print(f"[config] ✅ Found '{name}' → device_id={i} ({dev['name']})", flush=True)
                return i
    except Exception as e:
        print(f"[config] ⚠️  Device lookup failed: {e}", flush=True)
    return None

_detected = _find_device_by_name(_PREFERRED_DEVICE_NAME)
if _detected is None:
    print(f"[config] ⚠️  '{_PREFERRED_DEVICE_NAME}' not found — falling back to device_id={_FALLBACK_DEVICE_ID}", flush=True)
MICROPHONE_DEVICE_ID = _detected if _detected is not None else _FALLBACK_DEVICE_ID

# Network
WEBSOCKET_PORT    = 8013
HTTP_CONTROL_PORT = 8014
HUB_URL           = "http://localhost:8002"

SERVICE_NAME = "microphone_audio_service"