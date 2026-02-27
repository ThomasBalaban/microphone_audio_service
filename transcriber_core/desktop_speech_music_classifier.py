import numpy as np
from scipy import signal
from .config import FS

class SpeechMusicClassifier:
    """Classifies audio as either speech or music based on audio characteristics"""
    
    def __init__(self, max_history=5):
        # Default type
        self.current_type = "speech"
        
        # History for stabilization
        self.max_history = max_history
        self.history = ["speech"] * max_history
        
    def classify(self, audio_chunk):
        """
        Analyzes an audio chunk and classifies it as speech or music
        Returns: (type, confidence) where type is "speech" or "music"
        """
        # Skip empty chunks
        if len(audio_chunk) < FS * 0.5:  # Need at least 0.5 second
            return self.current_type, 0.5
            
        # Get various audio metrics for classification
        features = {}
        
        # 1. Basic audio energy
        features["energy"] = np.abs(audio_chunk).mean()
        
        # 2. Spectral features
        # Get frequency spectrum
        frequencies, times, spectrogram = signal.spectrogram(
            audio_chunk, 
            fs=FS,
            window='hamming',
            nperseg=1024,
            noverlap=512
        )
        
        # Calculate spectral properties
        # --- FIX: Added a small epsilon (1e-10) to prevent division by zero on silent audio chunks ---
        denominator = np.sum(spectrogram, axis=0) + 1e-10
        spectral_centroid = np.sum(frequencies[:, np.newaxis] * spectrogram, axis=0) / denominator
        features["mean_spectral_centroid"] = np.mean(spectral_centroid)
        
        # 3. Zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_chunk)))) / len(audio_chunk)
        features["zero_crossing_rate"] = zero_crossings
        
        # 4. Rhythm features - simplified beat detection
        # Calculate energy envelope
        frame_size = int(FS * 0.02)  # 20ms frames
        energy_frames = []
        
        for i in range(0, len(audio_chunk) - frame_size, frame_size):
            frame = audio_chunk[i:i+frame_size]
            energy_frames.append(np.sum(frame**2))
            
        energy_envelope = np.array(energy_frames)
        
        # Detect peaks in energy envelope
        peak_indices = signal.find_peaks(energy_envelope, distance=5)[0]
        
        # Calculate beat regularity (standard deviation of inter-onset intervals)
        if len(peak_indices) > 1:
            inter_onset_intervals = np.diff(peak_indices)
            mean_interval = np.mean(inter_onset_intervals)
            # --- FIX: Add a check for mean_interval being zero ---
            if mean_interval > 0:
                features["beat_regularity"] = np.std(inter_onset_intervals) / mean_interval
            else:
                features["beat_regularity"] = 1.0
        else:
            features["beat_regularity"] = 1.0  # No clear beats
        
        # 5. Spectral flatness (music tends to have lower flatness)
        log_power_spectrum = np.log(np.mean(spectrogram, axis=1) + 1e-10)
        # --- FIX: Added epsilon here as well for safety ---
        mean_exp_spectrum = np.mean(np.exp(log_power_spectrum)) + 1e-10
        features["spectral_flatness"] = np.exp(np.mean(log_power_spectrum)) / mean_exp_spectrum
        
        # 6. Spectral contrast (dynamic range in each frequency band)
        # Simplification: measure variance across time for each frequency band
        features["spectral_variance"] = np.mean(np.var(spectrogram, axis=1))
        
        # Classify based on features
        # These thresholds are determined empirically
        music_score = 0
        confidence = 0.5  # Default middle confidence
        
        # Music usually has more regular beats
        if features["beat_regularity"] < 0.5:
            music_score += 2
            confidence += 0.1
        
        # Music often has more spectral variance
        if features["spectral_variance"] > 0.2:
            music_score += 1
            confidence += 0.05
            
        # Speech has higher zero crossing rate typically
        if features["zero_crossing_rate"] > 0.1:
            music_score -= 1
            confidence += 0.05
            
        # Music typically has lower spectral flatness (more tonal)
        if features["spectral_flatness"] < 0.4:
            music_score += 1
            confidence += 0.05
            
        # Music tends to have a different spectral centroid range
        if 2000 < features["mean_spectral_centroid"] < 4000:
            music_score += 1
            confidence += 0.05
        
        # Classify based on overall score
        if music_score > 2:
            predicted_type = "music"
        elif music_score < 0:
            predicted_type = "speech"
        else:
            # Borderline case - use history
            previous_types_count = self.history.count("music")
            if previous_types_count > self.max_history / 2:
                predicted_type = "music"
            else:
                predicted_type = "speech"
                
        # Cap confidence
        confidence = min(confidence, 0.95)
        
        # Update history and current type
        self.history.append(predicted_type)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        # Smooth classification with history
        most_common = max(set(self.history), key=self.history.count)
        
        # Only change type if we're confident
        if self.history.count(most_common) >= int(self.max_history * 0.6):
            self.current_type = most_common
        
        return self.current_type, confidence