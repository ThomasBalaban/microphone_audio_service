import os
import time
import re
import numpy as np
import soundfile as sf
import sounddevice as sd
from threading import Thread
from scipy import signal
from .config import FS, CHUNK_DURATION, OVERLAP, MAX_THREADS 

class AudioProcessor:
    def __init__(self, transcriber):
        self.transcriber = transcriber
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Get the native sample rate of the device
        try:
            device_info = sd.query_devices(transcriber.DESKTOP_DEVICE_ID)
            self.native_samplerate = int(device_info['default_samplerate'])
        except:
            self.native_samplerate = FS
        
        self.needs_resampling = (self.native_samplerate != FS)
        if self.needs_resampling:
            print(f"   Will resample: {self.native_samplerate} Hz → {FS} Hz")

    def resample_audio(self, audio, from_rate, to_rate):
        """Resample audio from one sample rate to another."""
        if from_rate == to_rate:
            return audio
        
        num_samples = int(len(audio) * to_rate / from_rate)
        resampled = signal.resample(audio, num_samples)
        return resampled.astype(np.float32)

    def audio_callback(self, indata, frames, timestamp, status):
        """Buffers audio and spawns processing threads for overlapping chunks."""
        try:
            if status:
                print(f"[Desktop Audio Status] {status}")
            
            if not self.transcriber.processing_lock.is_set() and self.transcriber.active_threads < MAX_THREADS * 0.5:
                self.transcriber.processing_lock.set()
                    
            if self.transcriber.stop_event.is_set():
                return
            
            # Convert to mono if stereo
            if len(indata.shape) > 1 and indata.shape[1] > 1:
                new_audio = np.mean(indata, axis=1).astype(np.float32)
            else:
                new_audio = indata.flatten().astype(np.float32)
            
            # Resample if needed
            if self.needs_resampling:
                new_audio = self.resample_audio(new_audio, self.native_samplerate, FS)
            
            self.audio_buffer = np.concatenate([self.audio_buffer, new_audio])
            
            target_samples = FS * CHUNK_DURATION
            if (self.transcriber.processing_lock.is_set() and 
                len(self.audio_buffer) >= target_samples and
                self.transcriber.active_threads < MAX_THREADS):
                    
                chunk = self.audio_buffer[:target_samples].copy()
                overlap_samples = int(FS * (CHUNK_DURATION - OVERLAP))
                self.audio_buffer = self.audio_buffer[overlap_samples:]
                    
                self.transcriber.active_threads += 1
                Thread(target=self.process_chunk, args=(chunk,)).start()
                self.transcriber.last_processed = time.time()
                
                if self.transcriber.active_threads >= MAX_THREADS:
                    self.transcriber.processing_lock.clear()
                    print(f"Pausing processing - too many active threads: {self.transcriber.active_threads}")
                    
        except Exception as e:
            print(f"Audio callback error: {e}")
            import traceback
            traceback.print_exc()
            self.audio_buffer = np.array([], dtype=np.float32)
    
    def process_chunk(self, chunk):
        """Processes a single audio chunk in a separate thread."""
        filename = None
        try:
            # Pre-process audio to filter out silence
            amplitude = np.abs(chunk).mean()
            if amplitude < 0.005:
                return

            # Check for valid audio
            if np.isnan(chunk).any() or np.isinf(chunk).any():
                print("Warning: Invalid audio data, skipping chunk")
                return

            filename = self.save_audio(chunk)
            
            if self.transcriber.auto_detect:
                audio_type, confidence = self.transcriber.classifier.classify(chunk)
            else:
                audio_type = self.transcriber.classifier.current_type
                confidence = 0.8
                
            if confidence < 0.4:
                return
            
            # Transcribe
            segments, info = self.transcriber.model.transcribe(
                chunk, 
                beam_size=1, 
                language="en"
            )
            text = "".join(seg.text for seg in segments).strip()
            
            # Remove repeated patterns
            text = re.sub(r'(\w)(\s*-\s*\1){3,}', r'\1...', text)
            
            if text and len(text) >= 2:
                # Apply name correction HERE before putting in queue
                corrected_text = text
                for variation, name in self.transcriber.name_variations.items():
                    corrected_text = re.sub(variation, name, corrected_text, flags=re.IGNORECASE)
                
                # Put CORRECTED text in queue for main.py to consume
                self.transcriber.result_queue.put((corrected_text, filename, audio_type, confidence))
            else:
                # Clean up if no text
                if not self.transcriber.keep_files and filename and os.path.exists(filename):
                    os.remove(filename)
                
        except Exception as e:
            print(f"Processing error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.transcriber.active_threads -= 1
            if filename and not self.transcriber.keep_files and os.path.exists(filename):
                try:
                    if not any(filename in item for item in list(self.transcriber.result_queue.queue)):
                         os.remove(filename)
                except:
                    pass

    def save_audio(self, chunk):
        """Saves audio chunk to file and returns filename."""
        timestamp = time.strftime("%Y%m%d-%H%M%S-%f")[:-3]
        filename = os.path.join(self.transcriber.SAVE_DIR, f"desktop_{timestamp}.wav")
        sf.write(filename, chunk, FS, subtype='PCM_16')
        self.transcriber.saved_files.append(filename)
        return filename