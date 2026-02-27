# nami/audio_utils/desktop_audio_processor.py
import os
import time
import numpy as np
import soundfile as sf
import re
from threading import Thread
from scipy import signal
import torch

class AudioProcessor:
    def __init__(self, transcriber):
        self.transcriber = transcriber
        self.audio_buffer = np.array([], dtype=np.float32)

    def audio_callback(self, indata, frames, timestamp, status):
        """Buffers audio and spawns processing threads for overlapping chunks."""
        try:
            from .config import FS, CHUNK_DURATION, OVERLAP, MAX_THREADS

            # Resume processing if thread count is low
            if not self.transcriber.processing_lock.is_set() and self.transcriber.active_threads < MAX_THREADS * 0.5:
                self.transcriber.processing_lock.set()
                    
            if self.transcriber.stop_event.is_set():
                return
                
            new_audio = indata.flatten().astype(np.float32)
            self.audio_buffer = np.concatenate([self.audio_buffer, new_audio])
            
            # Process when buffer reaches target duration and we're not overloaded
            if (self.transcriber.processing_lock.is_set() and 
                len(self.audio_buffer) >= FS * CHUNK_DURATION and
                self.transcriber.active_threads < MAX_THREADS):
                    
                # Copy chunk to prevent modification during processing
                chunk = self.audio_buffer[:FS*CHUNK_DURATION].copy()
                # Slide buffer forward to create overlap
                self.audio_buffer = self.audio_buffer[int(FS*(CHUNK_DURATION-OVERLAP)):]
                    
                self.transcriber.active_threads += 1
                Thread(target=self.process_chunk, args=(chunk,)).start()
                self.transcriber.last_processed = time.time()
                
                # If we get too many threads, temporarily pause processing
                if self.transcriber.active_threads >= MAX_THREADS:
                    self.transcriber.processing_lock.clear()
                    print(f"Pausing processing - too many active threads: {self.transcriber.active_threads}")
                    
        except Exception as e:
            print(f"Audio callback error: {e}")
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
            
            if self.auto_detect:
                audio_type, confidence = self.classifier.classify(chunk)
            else:
                audio_type = self.classifier.current_type
                confidence = 0.8
                
            if confidence < 0.4:
                return
            
            # Transcribe
            segments, info = self.model.transcribe(
                chunk, 
                beam_size=1, 
                language="en"
            )
            text = "".join(seg.text for seg in segments).strip()
            
            # Remove repeated patterns
            text = re.sub(r'(\w)(\s*-\s*\1){3,}', r'\1...', text)
            
            if text and len(text) >= 2:
                # Apply name correction HERE before queuing
                corrected_text = text
                for variation, name in self.name_variations.items():
                    corrected_text = re.sub(variation, name, corrected_text, flags=re.IGNORECASE)
                
                # Put corrected text in queue
                self.result_queue.put((corrected_text, filename, audio_type, confidence))
            else:
                # Clean up if no text
                if not self.keep_files and filename and os.path.exists(filename):
                    os.remove(filename)
                
        except Exception as e:
            print(f"Processing error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure the thread count is always decremented
            self.active_threads -= 1
            # Clean up unused files
            if filename and not self.keep_files and os.path.exists(filename):
                try:
                    if not any(filename in item for item in list(self.result_queue.queue)):
                        os.remove(filename)
                except:
                    pass
                
    def save_audio(self, chunk):
        """Saves audio chunk to file and returns filename."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.transcriber.SAVE_DIR, f"desktop_{timestamp}.wav")
        sf.write(filename, chunk, self.transcriber.FS, subtype='PCM_16')
        self.transcriber.saved_files.append(filename)
        return filename