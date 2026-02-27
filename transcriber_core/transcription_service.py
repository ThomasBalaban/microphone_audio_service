# transcriber_core/transcription_service.py
import time
import multiprocessing
import sys
import threading
from queue import Empty
from difflib import SequenceMatcher

class TranscriptionDeduplicator:
    """Advanced stitching that eliminates stutters with performance safety."""
    
    def __init__(self, similarity_threshold=0.6, time_window=4.0):
        self.recent_transcripts = {}  # source -> (history_text, timestamp)
        self.similarity_threshold = similarity_threshold
        self.time_window = time_window
        
    def _clean_overlap(self, prev_text, curr_text):
        """Finds where the new text actually starts by stripping the tail of the old one."""
        p_words = prev_text.strip().split()
        
        # We only care about the last 20 words of context for overlap detection
        p_tail = " ".join(p_words[-20:]).lower()
        c_full = curr_text.lower()
        
        matcher = SequenceMatcher(None, p_tail, c_full)
        match = matcher.find_longest_match(0, len(p_tail), 0, len(c_full))
        
        # If the match starts near the beginning of the NEW text and is significant
        if match.b < 15 and match.size > 5:
            # Slice the current text starting from the end of the match
            return curr_text[match.b + match.size:].strip()
            
        return curr_text

    def process(self, text, source):
        """Processes transcription and returns only unique, new content."""
        current_time = time.time()
        if not text or len(text.strip()) < 1:
            return False, None
            
        if source in self.recent_transcripts:
            prev_text, prev_timestamp = self.recent_transcripts[source]
            
            # Reset history if it's been a long pause (fresh start)
            if current_time - prev_timestamp > self.time_window:
                self.recent_transcripts[source] = (text, current_time)
                return True, text
            
            # Strip overlaps
            unique_content = self._clean_overlap(prev_text, text)
            
            if not unique_content or len(unique_content) < 1:
                return False, None
                
            # Update history (Capped at 1000 chars to prevent CPU bloat)
            new_history = (prev_text + " " + unique_content)[-1000:]
            self.recent_transcripts[source] = (new_history, current_time)
            return True, unique_content
        
        self.recent_transcripts[source] = (text, current_time)
        return True, text

def transcription_process_target(result_queue, stop_event):
    """The target function for the Transcription Process."""
    from transcriber_core import MicrophoneTranscriber, DesktopTranscriber
    
    _mic = MicrophoneTranscriber(keep_files=False)
    _desktop = DesktopTranscriber(keep_files=False)
    deduplicator = TranscriptionDeduplicator()

    # Start the engines in separate threads
    t_desktop = threading.Thread(target=_desktop.run, daemon=True, name="DesktopThread")
    t_mic = threading.Thread(target=_mic.run, daemon=True, name="MicThread")
    t_desktop.start()
    t_mic.start()

    print("🧠 [SENSES] Dual Engines Started. Monitoring for audio...")
    sys.stdout.flush()

    last_health_check = time.time()

    while not stop_event.is_set():
        # Health Monitoring
        if time.time() - last_health_check > 5:
            if not t_desktop.is_alive(): print("⚠️ [SENSES] Desktop Transcriber thread DIED.")
            if not t_mic.is_alive(): print("⚠️ [SENSES] Mic Transcriber thread DIED.")
            sys.stdout.flush()
            last_health_check = time.time()

        # Check Desktop Queue
        try:
            while not _desktop.result_queue.empty():
                text, sid, src, conf = _desktop.result_queue.get_nowait()
                ok, final = deduplicator.process(text, "desktop")
                if ok:
                    result_queue.put({"type": "transcript", "source": "desktop", "text": final, "timestamp": time.time()})
                _desktop.result_queue.task_done()
        except Empty: pass

        # Check Mic Queue
        try:
            while not _mic.result_queue.empty():
                text, fn, src, conf = _mic.result_queue.get_nowait()
                ok, final = deduplicator.process(text, "microphone")
                if ok:
                    result_queue.put({"type": "transcript", "source": "microphone", "text": final, "timestamp": time.time()})
                _mic.result_queue.task_done()
        except Empty: pass
        
        time.sleep(0.01)

class TranscriptionService:
    def __init__(self):
        self.process = None
        self.result_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()

    def start(self):
        if self.process and self.process.is_alive(): return
        self.stop_event.clear()
        self.process = multiprocessing.Process(
            target=transcription_process_target, 
            args=(self.result_queue, self.stop_event), 
            daemon=True
        )
        self.process.start()
    
    def stop(self):
        self.stop_event.set()
        if self.process: self.process.join(timeout=3)

    def get_results(self):
        results = []
        while not self.result_queue.empty():
            try: results.append(self.result_queue.get_nowait())
            except Empty: break
        return results