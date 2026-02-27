# transcriber_core/desktop_transcriber.py
import os
import time
import re
import uuid
import sys
import sounddevice as sd
import numpy as np
from threading import Thread, Event
from queue import Queue, Empty
from collections import deque
from faster_whisper import WhisperModel
from .config import (
    FS, DESKTOP_DEVICE_ID, WHISPER_MODEL_SIZE, 
    WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
)

# Optimized for Lyrical Transcription
CHUNK_DURATION = 5.0      
OVERLAP_DURATION = 2.0    
MIN_AUDIO_ENERGY = 0.0010 
MAX_BUFFER_SECONDS = 15.0 

class SpeechMusicTranscriber:
    def __init__(self, keep_files=False):
        self.FS = FS
        self.result_queue = Queue()
        self.audio_queue = Queue()
        self.stop_event = Event()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.word_context = deque(maxlen=50) 
        
        self.name_variations = {
            r'\bnaomi\b': 'Nami', r'\bnow may\b': 'Nami',
            r'\bnomi\b': 'Nami', r'\bnamy\b': 'Nami'
        }

        try:
            print(f"🤖 [Desktop] Loading {WHISPER_MODEL_SIZE}... (Hardware: M4 Pro)")
            self.model = WhisperModel(
                WHISPER_MODEL_SIZE, 
                device=WHISPER_DEVICE, 
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=8 # Utilize more of your M4 Pro cores
            )
            print(f"✅ [Desktop] Model ready.")
        except Exception as e:
            print(f"❌ [Desktop] Model failed: {e}")
            sys.stdout.flush()
            raise

    def _apply_name_correction(self, text):
        for variation, name in self.name_variations.items():
            text = re.sub(variation, name, text, flags=re.IGNORECASE)
        return text

    def _normalize_audio(self, audio):
        """Advanced normalization to bring vocals to the front."""
        max_val = np.max(np.abs(audio))
        if max_val > 0.0001:
            # Normalize and then slightly compress the peaks to help Whisper
            audio = audio / max_val
            return np.clip(audio * 1.2, -1.0, 1.0)
        return audio

    def audio_callback(self, indata, frames, timestamp, status):
        if self.stop_event.is_set(): return
        audio = np.mean(indata, axis=1).astype(np.float32) if indata.shape[1] > 1 else indata.flatten()
        self.audio_queue.put(audio)

    def _processing_loop(self):
        print("🔄 [Desktop] Transcription loop active.")
        sys.stdout.flush()
        
        window_samples = int((CHUNK_DURATION + OVERLAP_DURATION) * self.FS)
        step_samples = int(CHUNK_DURATION * self.FS)
        max_buffer_samples = int(MAX_BUFFER_SECONDS * self.FS)
        
        while not self.stop_event.is_set():
            new_chunks = []
            while not self.audio_queue.empty():
                new_chunks.append(self.audio_queue.get())
            
            if new_chunks:
                self.audio_buffer = np.concatenate([self.audio_buffer] + new_chunks)
            
            if len(self.audio_buffer) > max_buffer_samples:
                print(f"⚠️ [Desktop] Lag detected. Flushing...")
                sys.stdout.flush()
                self.audio_buffer = self.audio_buffer[-window_samples:]
            
            if len(self.audio_buffer) >= window_samples:
                window = self._normalize_audio(self.audio_buffer[:window_samples].copy())
                self.audio_buffer = self.audio_buffer[step_samples:]
                
                # Context Prompt: Add a 'music' hint to help the model expect lyrics
                prompt = "Transcribing lyrics and dialogue. " + " ".join(self.word_context)
                
                try:
                    # Optimized parameters for singing:
                    # - Beam Size 5: Better accuracy for complex lyrical choices
                    # - min_silence_duration_ms 500: Don't cut between long-held notes
                    segments, _ = self.model.transcribe(
                        window, 
                        initial_prompt=prompt,
                        vad_filter=True,
                        vad_parameters=dict(
                            min_speech_duration_ms=250,
                            min_silence_duration_ms=500,
                            speech_pad_ms=400
                        ),
                        language="en",
                        beam_size=5, 
                        best_of=5
                    )
                    
                    text = " ".join([s.text.strip() for s in segments]).strip()
                    if text and len(text) > 2:
                        text = self._apply_name_correction(text)
                        for word in text.split(): self.word_context.append(word)
                        
                        self.result_queue.put((text, str(uuid.uuid4()), "desktop", 0.9))
                        print(f"✅ [Desktop Output]: {text}")
                        sys.stdout.flush()
                except Exception as e:
                    print(f"⚠️ [Desktop] Transcription error: {e}")
                    sys.stdout.flush()
            
            time.sleep(0.05)

    def run(self):
        t = Thread(target=self._processing_loop, daemon=True)
        t.start()
        try:
            with sd.InputStream(
                device=DESKTOP_DEVICE_ID, 
                samplerate=self.FS, 
                channels=1, 
                callback=self.audio_callback,
                blocksize=int(self.FS * 0.1)
            ):
                while not self.stop_event.is_set() and t.is_alive(): 
                    time.sleep(0.1)
        except Exception as e:
            print(f"❌ [Desktop Fatal Error]: {e}")
            sys.stdout.flush()