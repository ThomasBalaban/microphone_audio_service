"""
transcriber_core/microphone.py
─────────────────────────────────────────────────────────────────────────────
Improvements over the original:

  1. OpenAI whisper-1 API  — far better accuracy on fast / loud speech than
     the local parakeet-mlx model.

  2. Adaptive noise gate   — on startup the mic is silently recorded for
     NOISE_CALIBRATION_SECS to measure the ambient noise floor (A/C, fans,
     room tone). The VAD energy threshold is set to noise_floor × multiplier
     so the gate tracks your actual environment automatically.

  3. High-pass filter      — a 4th-order Butterworth HPF at HPF_CUTOFF_HZ
     (80 Hz) is applied *before* the VAD energy check. This strips the low-
     frequency rumble that A/C units produce, which was causing long periods
     of false activity.

  4. Pre-processing before transcription — the same HPF plus RMS
     normalisation is applied to every chunk before it is sent to the API,
     which gives Whisper a cleaner, consistently-levelled signal.

  5. In-memory WAV buffer — audio is written to an io.BytesIO object instead
     of a temp file on disk, keeping latency low.
"""

import io
import os
import re
import sys
import time
import traceback
from queue import Queue
from threading import Event, Lock, Thread

import numpy as np # type: ignore
import sounddevice as sd # type: ignore
import soundfile as sf # type: ignore
from openai import OpenAI # type: ignore
from scipy import signal # type: ignore

from .config import FS, MAX_THREADS, MICROPHONE_DEVICE_ID, SAVE_DIR

# ── VAD / gate parameters ─────────────────────────────────────────────────────
VAD_SILENCE_DURATION    = 0.55  # seconds of silence that ends an utterance
VAD_MAX_SPEECH_DURATION = 15.0  # hard cap: flush even if speech never stops

NOISE_CALIBRATION_SECS  = 2.5   # how long to listen before opening the stream
NOISE_GATE_MULTIPLIER   = 2.8   # threshold = noise_floor_rms × this value
MIN_VAD_THRESHOLD       = 0.005 # never go below this (prevents hair-trigger)
MAX_VAD_THRESHOLD       = 0.045 # never go above this (prevents deaf mode)

# ── Audio pre-processing ──────────────────────────────────────────────────────
HPF_CUTOFF_HZ   = 80.0   # kill everything below this (A/C hum, desk rumble)
SPEECH_NORM_RMS = 0.08   # target RMS after normalisation


class MicrophoneTranscriber:
    """
    Batch microphone transcriber.

    Public interface is identical to the original so that service.py
    and any other callers need no changes.
    """

    def __init__(self, keep_files=False, transcript_manager=None, device_id=None):
        self.FS          = FS
        self.SAVE_DIR    = SAVE_DIR
        self.MAX_THREADS = MAX_THREADS
        self.device_id   = device_id if device_id is not None else MICROPHONE_DEVICE_ID

        os.makedirs(self.SAVE_DIR, exist_ok=True)

        # ── OpenAI client ─────────────────────────────────────────────────────
        try:
            from api_keys import OPENAI_API_KEY  # noqa: PLC0415
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            print("✅ [Mic] OpenAI Whisper-1 client ready", flush=True)
        except Exception as exc:
            print(f"❌ [Mic] OpenAI init failed: {exc}", flush=True)
            raise

        # ── High-pass filter coefficients (built once, reused everywhere) ────
        self._hpf_b, self._hpf_a = signal.butter(
            4, HPF_CUTOFF_HZ / (FS / 2), btype="high"
        )

        # ── Thread-safety / state ─────────────────────────────────────────────
        self.result_queue    = Queue()
        self.stop_event      = Event()
        self.saved_files     = []
        self.keep_files      = keep_files
        self.active_threads  = 0
        self.processing_lock = Event()
        self.processing_lock.set()

        # VAD state
        self.speech_buffer      = np.array([], dtype=np.float32)
        self.is_speaking        = False
        self.silence_start_time = None
        self.speech_start_time  = None
        self.buffer_lock        = Lock()

        # Will be set by _calibrate_noise_floor() inside run()
        self.vad_threshold = 0.012

        self.transcript_manager = transcript_manager
        self.volume_callback    = None

        # Name correction table
        self.name_variations = {
            r"\bnaomi\b":        "Nami",
            r"\bnow may\b":      "Nami",
            r"\bnomi\b":         "Nami",
            r"\bnamy\b":         "Nami",
            r"\bnot me\b":       "Nami",
            r"\bnah me\b":       "Nami",
            r"\bnonny\b":        "Nami",
            r"\bnonni\b":        "Nami",
            r"\bmamie\b":        "Nami",
            r"\bgnomey\b":       "Nami",
            r"\barmy\b":         "Nami",
            r"\bpeepingnaomi\b": "PeepingNami",
            r"\bpeepingnomi\b":  "PeepingNami",
        }

    # ── Public helpers ────────────────────────────────────────────────────────

    def set_volume_callback(self, callback):
        """Register callback for GUI volume updates."""
        self.volume_callback = callback

    def stop(self):
        """Signal the run loop to stop."""
        self.stop_event.set()

    # ── Audio pre-processing ──────────────────────────────────────────────────

    def _apply_hpf(self, audio: np.ndarray) -> np.ndarray:
        """Apply the high-pass filter to remove low-frequency noise."""
        return signal.lfilter(self._hpf_b, self._hpf_a, audio).astype(np.float32)

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline used before transcription:
          1. High-pass filter  — strips A/C hum / rumble
          2. RMS normalise     — gives Whisper a consistent level regardless of
                                 whether the speaker is quiet, loud, or shouting
          3. Hard clip         — prevents clipping artefacts after normalisation
        """
        filtered = self._apply_hpf(audio)

        rms = np.sqrt(np.mean(filtered ** 2))
        if rms > 1e-6:
            filtered = filtered * (SPEECH_NORM_RMS / rms)

        return np.clip(filtered, -1.0, 1.0).astype(np.float32)

    # ── Noise-floor calibration ───────────────────────────────────────────────

    def _calibrate_noise_floor(self, duration: float) -> float:
        """
        Record ambient audio (mic must be open to room noise) and set the VAD
        threshold relative to the measured noise-floor RMS.

        The HPF is applied first so that A/C rumble below 80 Hz doesn't inflate
        the threshold and inadvertently mute your voice.
        """
        print(
            f"🎙️  [Mic] Calibrating noise floor ({duration:.1f}s) — "
            "don't speak yet…",
            flush=True,
        )

        samples   = int(self.FS * duration)
        recording = sd.rec(
            samples,
            samplerate=self.FS,
            channels=1,
            device=self.device_id,
            dtype="float32",
        )
        sd.wait()

        flat      = self._apply_hpf(recording.flatten())
        noise_rms = float(np.sqrt(np.mean(flat ** 2)))
        threshold = float(
            np.clip(
                noise_rms * NOISE_GATE_MULTIPLIER,
                MIN_VAD_THRESHOLD,
                MAX_VAD_THRESHOLD,
            )
        )

        print(
            f"✅ [Mic] Noise RMS={noise_rms:.5f}  "
            f"VAD threshold={threshold:.5f}",
            flush=True,
        )
        return threshold

    # ── Audio callback (runs in sounddevice thread) ───────────────────────────

    def audio_callback(self, indata, frames, timestamp, status):
        if self.stop_event.is_set():
            return

        raw      = indata.flatten().astype(np.float32)
        filtered = self._apply_hpf(raw)
        rms      = float(np.sqrt(np.mean(filtered ** 2)))

        if self.volume_callback:
            # Scale so 0.08 RMS (typical speech) ≈ 1.0 on the meter
            self.volume_callback(min(1.0, rms / SPEECH_NORM_RMS))

        with self.buffer_lock:
            if rms > self.vad_threshold:
                # ── Speech detected ───────────────────────────────────────────
                if not self.is_speaking:
                    self.is_speaking       = True
                    self.speech_start_time = time.time()

                self.speech_buffer      = np.concatenate([self.speech_buffer, raw])
                self.silence_start_time = None

                # Hard cap: flush if utterance runs too long
                if time.time() - self.speech_start_time > VAD_MAX_SPEECH_DURATION:
                    self._flush_buffer()

            elif self.is_speaking:
                # ── Trailing silence ──────────────────────────────────────────
                # Keep buffering a little so the last word isn't clipped
                self.speech_buffer = np.concatenate([self.speech_buffer, raw])

                if self.silence_start_time is None:
                    self.silence_start_time = time.time()

                if time.time() - self.silence_start_time > VAD_SILENCE_DURATION:
                    self._flush_buffer()

    def _flush_buffer(self):
        """
        Called inside buffer_lock.
        Hands the accumulated speech buffer to a transcription thread.
        """
        min_samples = int(self.FS * 0.3)

        if len(self.speech_buffer) > min_samples and self.active_threads < self.MAX_THREADS:
            chunk              = self.speech_buffer.copy()
            self.speech_buffer = np.array([], dtype=np.float32)
            self.is_speaking        = False
            self.silence_start_time = None
            self.speech_start_time  = None

            self.active_threads += 1
            Thread(target=self.process_chunk, args=(chunk,), daemon=True).start()
        else:
            # Too short or too many threads — discard
            self.speech_buffer      = np.array([], dtype=np.float32)
            self.is_speaking        = False
            self.silence_start_time = None
            self.speech_start_time  = None

    # ── Transcription ─────────────────────────────────────────────────────────

    def process_chunk(self, chunk: np.ndarray):
        """
        Pre-process audio and send to OpenAI whisper-1.
        Writes to an in-memory BytesIO buffer — no temp files unless
        keep_files=True.
        """
        filename = None
        try:
            processed = self._preprocess(chunk)

            # Build an in-memory WAV file
            buf = io.BytesIO()
            sf.write(buf, processed, self.FS, format="WAV", subtype="PCM_16")
            buf.seek(0)

            if self.keep_files:
                filename = self.save_audio(chunk)

            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", buf, "audio/wav"),
                language="en",
                # Prompt nudges Whisper toward casual, fast-paced speech and
                # away from adding filler punctuation / hallucinating silence.
                prompt=(
                    "Transcribe exactly as spoken. "
                    "The speaker may talk quickly, loudly, or trail off. "
                    "Do not add punctuation that wasn't spoken."
                ),
            )

            text = (response.text or "").strip()

            if text and len(text) >= 2:
                # Name correction
                for pattern, name in self.name_variations.items():
                    text = re.sub(pattern, name, text, flags=re.IGNORECASE)

                self.result_queue.put((text, filename, "microphone", 0.92))
                print(f"✅ [Mic] {repr(text)}", flush=True)

        except Exception as exc:
            print(f"[Mic-ERROR] Transcription failed: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            self.active_threads -= 1

    # ── File helpers ──────────────────────────────────────────────────────────

    def save_audio(self, chunk: np.ndarray) -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S-%f")[:-3]
        filename  = os.path.join(self.SAVE_DIR, f"microphone_{timestamp}.wav")
        sf.write(filename, chunk, self.FS, subtype="PCM_16")
        self.saved_files.append(filename)
        return filename

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self):
        try:
            info = sd.query_devices(self.device_id)
            print(
                f"\n🎤 [Mic] Device {self.device_id}: {info['name']}",
                flush=True,
            )
        except Exception as exc:
            print(f"⚠️  [Mic] Could not query device info: {exc}", flush=True)

        # Calibrate noise gate *before* opening the live stream
        try:
            self.vad_threshold = self._calibrate_noise_floor(NOISE_CALIBRATION_SECS)
        except Exception as exc:
            print(
                f"⚠️  [Mic] Noise calibration failed ({exc}), "
                f"using default threshold {self.vad_threshold:.4f}",
                flush=True,
            )

        try:
            with sd.InputStream(
                device=self.device_id,
                samplerate=self.FS,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.FS // 20,   # 50 ms blocks
                dtype="float32",
            ):
                print("✅ [Mic] Listening…", flush=True)
                while not self.stop_event.is_set():
                    time.sleep(0.1)

        except Exception as exc:
            print(f"[Mic-FATAL] Run loop error: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            self.stop_event.set()
            print(
                f"🎤 [Mic] Listener stopped (device {self.device_id}).",
                flush=True,
            )