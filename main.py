#!/usr/bin/env python3
"""
Microphone Audio Service — Entry Point
=======================================
Standalone service that transcribes microphone input in real-time using
Parakeet MLX and broadcasts results to:
  • WebSocket clients on port 8013
  • The central Hub on port 8002 (Socket.IO)

Health check:  GET  http://localhost:8014/health
Shutdown:      POST http://localhost:8014/shutdown
"""

import os
import signal
import sys

# ── Ensure this directory is the working dir so relative imports resolve ─────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_THIS_DIR)
# ─────────────────────────────────────────────────────────────────────────────

import http_control
from service import MicrophoneService

_service: MicrophoneService | None = None


def _shutdown(*_):
    global _service
    if _service:
        _service.stop()
    sys.exit(0)


def main():
    global _service
    _service = MicrophoneService()

    # HTTP health-check + remote shutdown support
    http_control.start(shutdown_callback=_shutdown)

    # Handle OS signals
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    _service.run()


if __name__ == "__main__":
    main()