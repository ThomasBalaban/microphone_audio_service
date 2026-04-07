"""
transcriber_core/microphone.py
─────────────────────────────────────────────────────────────────────────────
Speed improvements over the previous version:

  1. VAD_SILENCE_DURATION 0.55s → 0.30s
       Saves ~250 ms at the tail of every utterance.

  2. 120 ms pre-roll ring buffer
       The callback continuously keeps the last PRE_ROLL_SECS of audio in a
       circular deque. Every flushed chunk is prepended with that pre-roll so
       the shorter silence timeout doesn't clip the first phoneme of speech.

  3. HTTP/2 on the OpenAI client (httpx)
       Eliminates the per-request TCP + TLS handshake overhead when multiple
       chunks are in-flight simultaneously. Falls back to HTTP/1.1 silently
       if httpx[http2] isn't installed.

  4. response_format="text"
       Skips Whisper returning a JSON envelope — the API streams the plain
       string directly, saving a small but free deserialization step.

  5. Calibration 2.5s → 1.5s
       Room-noise RMS stabilises in well under a second; 1.5s is plenty.

Hallucination fixes (v2):

  FIX 1 — MIN_CHUNK_RMS gate in process_chunk
       Rejects chunks whose raw RMS is below MIN_CHUNK_RMS before any
       normalization. Music/ambient noise that barely tripped VAD never
       reaches Whisper.

  FIX 2 — Conditional normalization in _preprocess
       Previously any RMS > 1e-6 triggered normalization, boosting near-
       silence up to SPEECH_NORM_RMS and feeding Whisper a convincingly
       loud-but-meaningless signal. Now only normalizes when RMS > 0.01
       (roughly the threshold for real speech energy).

  FIX 3 — Post-transcription hallucination blocklist
       Whisper-1 has well-known phantom outputs ("thank you for watching",
       FEMA URLs, engvid, etc.) that appear when audio is ambiguous. Any
       transcript containing a blocklist phrase is silently dropped.
"""

import collections
import io
import os
import re
import sys
import time
import traceback
from queue import Queue
from threading import Event, Lock, Thread

import numpy as np          # type: ignore
import sounddevice as sd    # type: ignore
import soundfile as sf      # type: ignore
from scipy import signal    # type: ignore

from .config import FS, MAX_THREADS, MICROPHONE_DEVICE_ID, SAVE_DIR

# ── VAD / gate parameters ─────────────────────────────────────────────────────
VAD_SILENCE_DURATION    = 0.30   # ⬇ was 0.55 — saves ~250 ms per utterance
VAD_MAX_SPEECH_DURATION = 15.0

NOISE_CALIBRATION_SECS  = 1.5   # ⬇ was 2.5
NOISE_GATE_MULTIPLIER   = 2.8
MIN_VAD_THRESHOLD       = 0.005
MAX_VAD_THRESHOLD       = 0.045

# ── Pre-roll ──────────────────────────────────────────────────────────────────
PRE_ROLL_SECS = 0.12   # 120 ms prepended to every flushed chunk

# ── Audio pre-processing ──────────────────────────────────────────────────────
HPF_CUTOFF_HZ   = 80.0
SPEECH_NORM_RMS = 0.08

# ── Hallucination guards ──────────────────────────────────────────────────────
# FIX 1: chunks whose raw RMS is below this are skipped before normalization
# can inflate them. Tune down to 0.005 if soft speech is being dropped.
MIN_CHUNK_RMS = 0.008

# FIX 2: normalization is only applied when RMS exceeds this floor.
# Prevents near-silence / music from being boosted to speech level.
MIN_NORM_RMS = 0.01

# FIX 3: Whisper-1 phantom outputs — transcripts matching any of these are
# silently dropped. All comparisons are lowercased substring matches.
_WHISPER_HALLUCINATIONS = {
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "don't forget to subscribe",
    "like and subscribe",
    "see you next time",
    "see you in the next video",
    "learn english",
    "www.",
    ".gov",
    ".com",
    "engvid",
    "office of the president",
    "this has been a presentation",
    "subtitles by",
    "translated by",
    "[ music ]",
    "[music]",
    "♪",
    # Whisper sometimes emits these when it hears only silence / noise
    "you",                          # single-word phantom — too short to be real anyway
    "thank you.",
    "thanks.",
}


def _build_openai_client():
    """
    Build an OpenAI client that uses HTTP/2 when httpx[http2] is available.
    Falls back to the default HTTP/1.1 client silently.
    """
    from api_keys import OPENAI_API_KEY  # noqa: PLC0415
    try:
        import httpx
        http_client = httpx.Client(http2=True)
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
        print("✅ [Mic] OpenAI client ready (HTTP/2)", flush=True)
    except Exception:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ [Mic] OpenAI client ready (HTTP/1.1)", flush=True)
    return client


class MicrophoneTranscriber:
    """
    Batch microphone transcriber with adaptive noise gate, HPF, pre-roll
    buffer, and OpenAI Whisper-1 transcription.

    Public interface is identical to the original so service.py needs no
    changes.
    """

    def __init__(self, keep_files=False, transcript_manager=None, device_id=None):
        self.FS          = FS
        self.SAVE_DIR    = SAVE_DIR
        self.MAX_THREADS = MAX_THREADS
        self.device_id   = device_id if device_id is not None else MICROPHONE_DEVICE_ID

        os.makedirs(self.SAVE_DIR, exist_ok=True)

        self.client = _build_openai_client()

        # High-pass filter (built once, reused everywhere)
        self._hpf_b, self._hpf_a = signal.butter(
            4, HPF_CUTOFF_HZ / (FS / 2), btype="high"
        )

        # Pre-roll ring buffer: stores the last PRE_ROLL_SECS of raw audio
        _pre_roll_samples = int(FS * PRE_ROLL_SECS)
        self._pre_roll: collections.deque = collections.deque(
            maxlen=_pre_roll_samples
        )

        # Thread-safety / state
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

        self.vad_threshold      = 0.012   # overwritten by calibration

        self.transcript_manager = transcript_manager
        self.volume_callback    = None

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
        self.volume_callback = callback

    def stop(self):
        self.stop_event.set()

    # ── Audio pre-processing ──────────────────────────────────────────────────

    def _apply_hpf(self, audio: np.ndarray) -> np.ndarray:
        return signal.lfilter(self._hpf_b, self._hpf_a, audio).astype(np.float32)

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        filtered = self._apply_hpf(audio)
        rms = np.sqrt(np.mean(filtered ** 2))
        # FIX 2: only normalize when there's real signal.
        # The old threshold (1e-6) boosted near-silence and music up to full
        # speech level, feeding Whisper a convincingly loud but meaningless
        # signal and triggering hallucinations.
        if rms > MIN_NORM_RMS:
            filtered = filtered * (SPEECH_NORM_RMS / rms)
        return np.clip(filtered, -1.0, 1.0).astype(np.float32)

    # ── Noise-floor calibration ───────────────────────────────────────────────

    def _calibrate_noise_floor(self, duration: float) -> float:
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
            f"✅ [Mic] Noise RMS={noise_rms:.5f}  VAD threshold={threshold:.5f}",
            flush=True,
        )
        return threshold

    # ── Audio callback ────────────────────────────────────────────────────────

    def audio_callback(self, indata, frames, timestamp, status):
        if self.stop_event.is_set():
            return

        raw      = indata.flatten().astype(np.float32)
        filtered = self._apply_hpf(raw)
        rms      = float(np.sqrt(np.mean(filtered ** 2)))

        if self.volume_callback:
            self.volume_callback(min(1.0, rms / SPEECH_NORM_RMS))

        with self.buffer_lock:
            if rms > self.vad_threshold:
                if not self.is_speaking:
                    self.is_speaking       = True
                    self.speech_start_time = time.time()
                    # Prepend pre-roll so we don't clip the first word
                    pre_roll_audio = np.array(list(self._pre_roll), dtype=np.float32)
                    if len(pre_roll_audio):
                        self.speech_buffer = pre_roll_audio

                self.speech_buffer      = np.concatenate([self.speech_buffer, raw])
                self.silence_start_time = None

                if time.time() - self.speech_start_time > VAD_MAX_SPEECH_DURATION:
                    self._flush_buffer()

            elif self.is_speaking:
                # Buffer trailing silence so the last word isn't clipped
                self.speech_buffer = np.concatenate([self.speech_buffer, raw])

                if self.silence_start_time is None:
                    self.silence_start_time = time.time()

                if time.time() - self.silence_start_time > VAD_SILENCE_DURATION:
                    self._flush_buffer()

            else:
                # Not speaking — keep the pre-roll ring buffer warm
                self._pre_roll.extend(raw.tolist())

    def _flush_buffer(self):
        """Called inside buffer_lock."""
        min_samples = int(self.FS * 0.3)

        if len(self.speech_buffer) > min_samples and self.active_threads < self.MAX_THREADS:
            chunk              = self.speech_buffer.copy()
            self.speech_buffer = np.array([], dtype=np.float32)
            self.is_speaking        = False
            self.silence_start_time = None
            self.speech_start_time  = None
            self._pre_roll.clear()   # stale after a flush

            self.active_threads += 1
            Thread(target=self.process_chunk, args=(chunk,), daemon=True).start()
        else:
            self.speech_buffer      = np.array([], dtype=np.float32)
            self.is_speaking        = False
            self.silence_start_time = None
            self.speech_start_time  = None

    # ── Transcription ─────────────────────────────────────────────────────────

    def process_chunk(self, chunk: np.ndarray):
        filename = None
        try:
            # FIX 1: reject low-energy chunks before normalization can inflate
            # them. Music / ambient noise that barely crossed the VAD gate
            # never reaches Whisper.
            raw_rms = float(np.sqrt(np.mean(chunk ** 2)))
            if raw_rms < MIN_CHUNK_RMS:
                print(
                    f"⏭️  [Mic] Chunk skipped — raw RMS {raw_rms:.5f} < {MIN_CHUNK_RMS} (not speech)",
                    flush=True,
                )
                return

            processed = self._preprocess(chunk)

            buf = io.BytesIO()
            sf.write(buf, processed, self.FS, format="WAV", subtype="PCM_16")
            buf.seek(0)

            if self.keep_files:
                filename = self.save_audio(chunk)

            # response_format="text" → plain string, no JSON envelope to parse
            text = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", buf, "audio/wav"),
                language="en",
                response_format="text",
                prompt=(
                    "Transcribe exactly as spoken. "
                    "The speaker may talk quickly, loudly, or trail off. "
                    "Do not add punctuation that wasn't spoken."
                ),
            )

            text = (text or "").strip()

            # FIX 3: drop known Whisper hallucination phrases
            if text:
                text_lower = text.lower()
                for phrase in _WHISPER_HALLUCINATIONS:
                    if phrase in text_lower:
                        print(f"🚫 [Mic] Hallucination blocked: {repr(text)}", flush=True)
                        return

            if text and len(text) >= 2:
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
            print(f"\n🎤 [Mic] Device {self.device_id}: {info['name']}", flush=True)
        except Exception as exc:
            print(f"⚠️  [Mic] Could not query device info: {exc}", flush=True)

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
            print(f"🎤 [Mic] Listener stopped (device {self.device_id}).", flush=True)