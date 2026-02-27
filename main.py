#!/usr/bin/env python3
"""
Microphone Audio Service — Entry Point

Health check:    GET  http://localhost:8014/health
List devices:    GET  http://localhost:8014/devices
Set device:      POST http://localhost:8014/set-device  {"device_id": N}
Shutdown:        POST http://localhost:8014/shutdown
"""

import os
import signal
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_THIS_DIR)

import http_control
from service import MicrophoneService

_service: MicrophoneService | None = None


def _shutdown(*_):
    global _service
    if _service:
        _service.stop()
    sys.exit(0)


def _swap_device(device_id: int):
    global _service
    if _service:
        _service.swap_device(device_id)


def main():
    global _service
    _service = MicrophoneService()

    http_control.start(
        shutdown_callback   = _shutdown,
        set_device_callback = _swap_device,
    )

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    _service.run()


if __name__ == "__main__":
    main()