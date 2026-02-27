import os
import time
import numpy as np
from threading import Thread
from .config import FS, CHUNK_DURATION, OVERLAP, MAX_THREADS

class AudioInputHandler:
    def __init__(self, transcriber, transcription_processor):
        self.transcriber = transcriber
        self.transcription_processor = transcription_processor
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_noise_log = 0  # For throttling noise messages
        
    def audio_callback(self, indata, frames, timestamp, status):
        """Process incoming audio data with enhanced error handling and noise control"""
        try:
            # Check if we should resume processing
            if not self.transcriber.processing_lock.is_set() and self.transcriber.active_threads < MAX_THREADS * 0.5:
                # Resume processing if thread count is low enough
                self.transcriber.processing_lock.set()
                print("Resuming audio processing")
                    
            if status:
                print(f"Audio status: {status}")
            
            # Skip if stopped
            if self.transcriber.stop_event.is_set():
                return
                
            # Process all incoming audio
            new_audio = np.squeeze(indata).astype(np.float32)
            
            # Skip processing if audio is too quiet (global noise gate)
            rms_amplitude = np.sqrt(np.mean(new_audio**2))
            if rms_amplitude < 0.0003:  # Very quiet audio
                if time.time() - self.last_noise_log > 5:
                    print(f"Input too quiet: {rms_amplitude:.6f} RMS, skipping")
                    self.last_noise_log = time.time()
                return
                
            # Check for invalid data early
            if np.isnan(new_audio).any() or np.isinf(new_audio).any():
                print("Warning: Input audio contains NaN or Inf values, replacing with zeros")
                new_audio = np.nan_to_num(new_audio, nan=0.0, posinf=0.0, neginf=0.0)
                
            # Add the new audio to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, new_audio])
            
            # Limit buffer size to prevent memory issues (max 30 seconds)
            max_buffer_size = FS * 30
            if len(self.audio_buffer) > max_buffer_size:
                # Keep only the most recent data
                self.audio_buffer = self.audio_buffer[-max_buffer_size:]
                print("Warning: Audio buffer too large, trimming")
            
            # Process when buffer reaches target duration and we're not overloaded
            if (self.transcriber.processing_lock.is_set() and 
                len(self.audio_buffer) >= FS * CHUNK_DURATION and
                self.transcriber.active_threads < MAX_THREADS):
                    
                # Check overall buffer energy before processing
                buffer_energy = np.abs(self.audio_buffer[:FS*CHUNK_DURATION]).mean()
                if buffer_energy < 0.0005:
                    # Skip low-energy chunks, but still advance buffer
                    self.audio_buffer = self.audio_buffer[FS*(CHUNK_DURATION-OVERLAP):]
                    return
                    
                # Copy chunk to prevent modification during processing
                chunk = self.audio_buffer[:FS*CHUNK_DURATION].copy()
                self.audio_buffer = self.audio_buffer[FS*(CHUNK_DURATION-OVERLAP):]
                    
                self.transcriber.active_threads += 1
                Thread(target=self.transcription_processor.process_chunk, args=(chunk,)).start()
                self.transcriber.last_processed = time.time()
                
                # If we get too many threads, temporary pause processing
                if self.transcriber.active_threads >= MAX_THREADS * 0.8:
                    self.transcriber.processing_lock.clear()
                    print(f"Pausing processing - too many active threads: {self.transcriber.active_threads}")
                    
        except Exception as e:
            print(f"Audio callback error: {e}")
            # Try to recover
            self.audio_buffer = np.array([], dtype=np.float32)
    
    def save_audio(self, chunk):
        """Save audio chunk to file and return filename"""
        from .config import SAVE_DIR
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(SAVE_DIR, f"fixed_{timestamp}.wav")
        
        import soundfile as sf
        sf.write(filename, chunk, FS, subtype='PCM_16')
        self.transcriber.saved_files.append(filename)
        
        return filename