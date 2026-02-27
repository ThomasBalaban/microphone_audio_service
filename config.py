"""
Configuration for the Microphone Audio Service.
Handles real-time microphone transcription via Parakeet MLX.
"""

# Audio
MICROPHONE_DEVICE_ID = 5
AUDIO_SAMPLE_RATE    = 16000

# Network
WEBSOCKET_PORT    = 8013   # WebSocket broadcast port
HTTP_CONTROL_PORT = 8014   # HTTP health-check / shutdown port
HUB_URL           = "http://localhost:8002"

# Service identity (used in hub events / logs)
SERVICE_NAME = "microphone_audio_service"