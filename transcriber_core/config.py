# transcriber_core/config.py
"""
Configuration for the transcription system.
Optimized for high-performance M4 Pro hardware.
"""

# Audio Settings
FS = 16000  # Sample rate in Hz
CHUNK_DURATION = 5  # Increased slightly for better lyrical context
OVERLAP = 1.5       # Overlap between chunks
MAX_THREADS = 4     
SAVE_DIR = "audio_captures"

# Device IDs (Verify with helper/sound_devices.py)
DESKTOP_DEVICE_ID = 4 
MICROPHONE_DEVICE_ID = 5 

# ==============================================================================
# FASTER-WHISPER SETTINGS (for Desktop Audio)
# ==============================================================================
# Upgraded to distil-large-v3 for maximum accuracy on M4 Pro
WHISPER_MODEL_SIZE = "distil-large-v3" 
WHISPER_DEVICE = "cpu"          # Use "cpu" with "int8" or "float32" on Mac
WHISPER_COMPUTE_TYPE = "int8"   # High speed, low memory overhead