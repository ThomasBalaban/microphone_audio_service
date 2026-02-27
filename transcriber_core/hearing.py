# nami/audio_utils/hearing.py
import threading
import time
import signal
import sys

# Import the main function from each of our self-contained Whisper transcribers
from .desktop_transcriber import run_desktop_transcriber, stop_event as desktop_stop_event
# --- MODIFIED: Import the improved microphone transcriber ---
from .microphone import transcribe_microphone, stop_event as mic_stop_event

def main():
    print("🎙️ Initializing Dual faster-whisper Transcription System...")

    # --- Create a thread for the Desktop Whisper Transcriber ---
    desktop_thread = threading.Thread(
        target=run_desktop_transcriber,
        daemon=True,
        name="WhisperDesktopThread"
    )

    # --- Create a thread for the Improved Microphone Whisper Transcriber ---
    mic_thread = threading.Thread(
        target=transcribe_microphone,
        daemon=True,
        name="WhisperMicThread"
    )

    # --- Start both threads ---
    desktop_thread.start()
    time.sleep(2) # Give it a moment to load the first model
    mic_thread.start()

    print("✅ Both Whisper transcription systems are running.")
    print("Press Ctrl+C to stop.")

    # --- Graceful Shutdown Logic ---
    def shutdown(sig, frame):
        print("\n🛑 Shutting down all systems...")
        mic_stop_event.set()
        desktop_stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep the main thread alive to wait for the shutdown signal
    while desktop_thread.is_alive() and mic_thread.is_alive():
        time.sleep(1)

if __name__ == "__main__":
    main()