"""
Configuration for the Microphone Audio Service.
"""
import os
import sys
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

print(f"[config] sys.path[0:3] = {sys.path[:3]}", flush=True)


def _load_sibling_secrets() -> None:
    secrets_dir = Path(__file__).resolve().parent.parent / "director_ui" / "secrets"
    if not secrets_dir.is_dir():
        return
    for path in sorted(secrets_dir.glob("*.env")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


if "GEMINI_API_KEY" not in os.environ or "OPENAI_API_KEY" not in os.environ:
    _load_sibling_secrets()

for _required in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
    if _required not in os.environ:
        raise RuntimeError(
            f"Missing required env var {_required}. "
            "Credentials live in director_ui/secrets/*.env. Start this service "
            "via the launcher, or export the variable manually."
        )

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