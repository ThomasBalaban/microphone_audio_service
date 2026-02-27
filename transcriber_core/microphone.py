import os
import sys
import numpy as np # type: ignore
import sounddevice as sd # type: ignore
import time
import soundfile as sf # type: ignore
import re
import traceback
from queue import Queue
from threading import Thread, Event, Lock
import parakeet_mlx # type: ignore
import mlx.core as mx # type: ignore
from .config import MICROPHONE_DEVICE_ID, FS, SAVE_DIR, MAX_THREADS

# VAD Parameters
VAD_ENERGY_THRESHOLD = 0.008
VAD_SILENCE_DURATION = 0.5
VAD_MAX_SPEECH_DURATION = 15.0

class MicrophoneTranscriber:
    """Batch microphone transcriber with fast VAD and smart buffering"""

    def __init__(self, keep_files=False, transcript_manager=None, device_id=None):
        self.FS = FS
        self.SAVE_DIR = SAVE_DIR
        self.MAX_THREADS = MAX_THREADS
        
        # Use passed device_id or fallback to config
        self.device_id = device_id if device_id is not None else MICROPHONE_DEVICE_ID
        
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        try:
            self.model = parakeet_mlx.from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")
        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            raise

        self.result_queue = Queue()
        self.stop_event = Event()
        self.saved_files = []
        self.keep_files = keep_files
        self.active_threads = 0
        self.processing_lock = Event()
        self.processing_lock.set()

        # VAD and Buffering State
        self.speech_buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.silence_start_time = None
        self.speech_start_time = None
        self.buffer_lock = Lock()

        self.transcript_manager = transcript_manager
        
        # New: Volume monitoring
        self.volume_callback = None
        
        # Name Correction
        self.name_variations = {
            r'\bnaomi\b': 'Nami', r'\bnow may\b': 'Nami', r'\bnomi\b': 'Nami',
            r'\bnamy\b': 'Nami', r'\bnot me\b': 'Nami', r'\bnah me\b': 'Nami',
            r'\bnonny\b': 'Nami', r'\bnonni\b': 'Nami', r'\bmamie\b': 'Nami',
            r'\bgnomey\b': 'Nami', r'\barmy\b': 'Nami', 
            r'\bpeepingnaomi\b': 'PeepingNami', r'\bpeepingnomi\b': 'PeepingNami'
        }

    def set_volume_callback(self, callback):
        """Register callback for GUI volume updates"""
        self.volume_callback = callback

    def stop(self):
        """Signals the run loop to stop."""
        self.stop_event.set()

    def audio_callback(self, indata, frames, timestamp, status):
        """Analyzes audio for speech, calculates volume, and buffers it."""
        if self.stop_event.is_set():
            return

        new_audio = indata.flatten().astype(np.float32)
        rms_amplitude = np.sqrt(np.mean(new_audio**2))

        # Update GUI volume meter (scaled for visual impact)
        if self.volume_callback:
            self.volume_callback(min(1.0, rms_amplitude * 10))

        with self.buffer_lock:
            if rms_amplitude > VAD_ENERGY_THRESHOLD:
                # Speech Detected
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_time = time.time()
                self.speech_buffer = np.concatenate([self.speech_buffer, new_audio])
                self.silence_start_time = None

                # Smart Overflow Protection
                if time.time() - self.speech_start_time > VAD_MAX_SPEECH_DURATION:
                    self._process_speech_buffer()

            elif self.is_speaking:
                # Silence after speech
                if self.silence_start_time is None:
                    self.silence_start_time = time.time()

                if time.time() - self.silence_start_time > VAD_SILENCE_DURATION:
                    self._process_speech_buffer()

    def _process_speech_buffer(self):
        """Processes the buffered speech in a separate thread."""
        if len(self.speech_buffer) > self.FS * 0.3 and self.active_threads < self.MAX_THREADS:
            chunk_to_process = self.speech_buffer.copy()
            self.speech_buffer = np.array([], dtype=np.float32)
            self.is_speaking = False
            self.silence_start_time = None
            self.speech_start_time = None

            self.active_threads += 1
            Thread(target=self.process_chunk, args=(chunk_to_process,)).start()
        else:
            self.speech_buffer = np.array([], dtype=np.float32)
            self.is_speaking = False

    def process_chunk(self, chunk):
        """Transcribes a chunk of audio."""
        filename = None
        try:
            # Only save to disk if explicitly requested
            if self.keep_files:
                filename = self.save_audio(chunk)
            
            with self.model.transcribe_stream() as transcriber:
                transcriber.add_audio(mx.array(chunk))
                result = transcriber.result
                text = result.text.strip() if result and hasattr(result, 'text') else ""
            
            if text and len(text) >= 2:
                corrected_text = text
                for variation, name in self.name_variations.items():
                    corrected_text = re.sub(variation, name, corrected_text, flags=re.IGNORECASE)
                
                self.result_queue.put((corrected_text, filename, "microphone", 0.85))
            else:
                # Cleanup if file was created and text was empty
                if self.keep_files and not self.keep_files and filename and os.path.exists(filename):
                    os.remove(filename)
        except Exception as e:
            print(f"[MIC-ERROR] Transcription thread failed: {str(e)}", file=sys.stderr)
            if self.keep_files and filename and os.path.exists(filename):
                try: os.remove(filename)
                except: pass
        finally:
            self.active_threads -= 1

    def save_audio(self, chunk):
        timestamp = time.strftime("%Y%m%d-%H%M%S-%f")[:-3]
        filename = os.path.join(self.SAVE_DIR, f"microphone_{timestamp}.wav")
        sf.write(filename, chunk, self.FS, subtype='PCM_16')
        self.saved_files.append(filename)
        return filename

    def run(self):
        """Start the audio stream"""
        try:
            device_info = sd.query_devices(self.device_id)
            print(f"\n🎤 Microphone Active (Device {self.device_id}): {device_info['name']}")
        except Exception as e:
            print(f"⚠️ Could not get device info: {e}")

        try:
            blocksize = self.FS // 20 

            with sd.InputStream(
                device=self.device_id,
                samplerate=self.FS,
                channels=1,
                callback=self.audio_callback,
                blocksize=blocksize,
                dtype='float32'
            ):
                while not self.stop_event.is_set():
                    time.sleep(0.1)

        except Exception as e:
            print(f"\n[MIC-FATAL] Run loop error: {e}", file=sys.stderr)
        finally:
            self.stop_event.set()
            print(f"🎤 Microphone listener (Device {self.device_id}) stopped.")