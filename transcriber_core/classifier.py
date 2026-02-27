import numpy as np # type: ignore
import torch # type: ignore
from .config import FS

class SpeechMusicClassifier:
    """Enhanced classifier to detect speech vs music with improved music detection"""
    
    def __init__(self):
        # Track the current audio type and history
        self.current_type = "speech"  # Default to speech
        self.history = []
        self.max_history = 5  # Increased history size for more stability
        self.min_confidence = 0.2  # Minimum confidence to change state
        
    def classify(self, audio_chunk):
        """Classify audio chunk as speech or music using enhanced features"""
        try:
            # Convert to numpy if tensor
            if torch.is_tensor(audio_chunk):
                audio_chunk = audio_chunk.cpu().numpy()
                
            # Normalize the audio
            audio_chunk = audio_chunk / (np.max(np.abs(audio_chunk)) + 1e-10)
            
            # Extract key features that differentiate speech and music
            
            # 1. Zero-crossing rate
            # Speech typically has moderate zero-crossing rate
            # Music often has higher or more consistent zero-crossing rate
            zero_crossings = np.sum(np.diff(np.signbit(audio_chunk)) != 0)
            zc_rate = zero_crossings / len(audio_chunk)
            
            # 2. Calculate spectral features
            # Get the frequency spectrum
            spectrum = np.abs(np.fft.rfft(audio_chunk))
            freqs = np.fft.rfftfreq(len(audio_chunk), 1/FS)
            
            # 3. Analyze energy in different frequency bands
            # Speech energy is typically concentrated in 300-3000 Hz
            # Music has wider frequency distribution and more energy in higher frequencies
            
            # Define frequency bands - more detailed for better discrimination
            bands = [0, 150, 300, 1000, 2000, 3000, 5000, 8000, 12000]
            band_energy = []
            
            for i in range(len(bands)-1):
                mask = (freqs >= bands[i]) & (freqs < bands[i+1])
                band_energy.append(np.sum(spectrum[mask]))
                
            # Normalize band energies
            total_energy = sum(band_energy) + 1e-10
            band_energy_ratio = [e/total_energy for e in band_energy]
            
            # 4. Calculate spectral flux (frame-to-frame spectral change)
            # Split the audio into frames
            frame_size = 400  # 25ms frames
            hop_size = 160    # 10ms hop
            num_frames = (len(audio_chunk) - frame_size) // hop_size
            
            if num_frames < 2:  # Need at least 2 frames for flux calculation
                frame_size = len(audio_chunk) // 4
                hop_size = frame_size // 2
                num_frames = 3  # Force creation of at least 3 frames
            
            frame_specs = []
            for i in range(num_frames):
                start_idx = min(i*hop_size, len(audio_chunk) - frame_size)
                frame = audio_chunk[start_idx:start_idx+frame_size]
                frame_spec = np.abs(np.fft.rfft(frame))
                frame_specs.append(frame_spec)
                
            # Calculate spectral flux between consecutive frames
            spectral_flux = []
            for i in range(1, len(frame_specs)):
                # Normalize spectra
                spec1 = frame_specs[i-1] / (np.sum(frame_specs[i-1]) + 1e-10)
                spec2 = frame_specs[i] / (np.sum(frame_specs[i]) + 1e-10)
                # Calculate flux
                flux = np.sum((spec2 - spec1)**2)
                spectral_flux.append(flux)
                
            # Average spectral flux
            avg_flux = np.mean(spectral_flux) if spectral_flux else 0
            
            # 5. Calculate spectral centroid (brightness)
            # Music often has higher spectral centroid
            centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
            normalized_centroid = centroid / (FS/2)  # Normalize by Nyquist frequency
            
            # 6. Calculate spectral flatness (tonality)
            # Music tends to have lower spectral flatness (more tonal)
            geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
            arithmetic_mean = np.mean(spectrum + 1e-10)
            flatness = geometric_mean / arithmetic_mean
            
            # 7. Rhythmic features - energy modulation
            # Calculate energy envelope
            frame_size = FS // 50  # 20 ms frames
            num_frames = len(audio_chunk) // frame_size
            energy_envelope = []
            
            for i in range(num_frames):
                frame = audio_chunk[i*frame_size:(i+1)*frame_size]
                energy = np.sum(frame**2)
                energy_envelope.append(energy)
                
            # Calculate variance of energy envelope (music tends to have more regular pattern)
            energy_variance = np.var(energy_envelope) if energy_envelope else 0
            
            # Collect decision features
            speech_features = 0
            music_features = 0
            
            # Feature 1: Zero crossing rate
            if 0.01 < zc_rate < 0.1:
                speech_features += 1
            elif zc_rate >= 0.1:
                music_features += 1.5  # Weight higher for music
                
            # Feature 2: Energy in speech band (1000-3000 Hz)
            speech_band_energy = band_energy_ratio[3] + band_energy_ratio[4]  # 1000-3000 Hz bands
            high_freq_energy = sum(band_energy_ratio[5:])  # Energy above 3000 Hz
            
            if speech_band_energy > 0.4:
                speech_features += 1
            
            # Music often has more energy in higher frequencies
            if high_freq_energy > 0.25:
                music_features += 1.5
                
            # Feature 3: Spectral flux
            # Speech has higher frame-to-frame variation than most music
            if avg_flux > 0.02:
                speech_features += 1
            elif avg_flux < 0.01:
                music_features += 1
                
            # Feature 4: Spectral centroid (brightness)
            if normalized_centroid > 0.2:
                music_features += 1
                
            # Feature 5: Spectral flatness
            if flatness < 0.1:  # More tonal (music)
                music_features += 1
            elif flatness > 0.2:  # More noise-like (speech)
                speech_features += 0.5
                
            # Feature 6: Energy variance
            # Music often has more regular energy patterns
            if energy_variance < 0.1 and len(energy_envelope) > 10:
                music_features += 1
                
            # Correct decision logic with improved weighting
            if speech_features > music_features:
                detected_type = "speech"
                confidence = min(0.5 + 0.1 * speech_features, 0.9)
            else:
                detected_type = "music"
                confidence = min(0.5 + 0.1 * music_features, 0.9)
                
            # Update history
            self.history.append(detected_type)
            if len(self.history) > self.max_history:
                self.history.pop(0)
                
            # Only change type if we have consistent evidence
            # More weighted toward detecting music (lower threshold)
            speech_count = self.history.count("speech")
            music_count = self.history.count("music")
            
            # Require at least 60% agreement for speech, but only 40% for music
            # This makes the classifier more sensitive to music
            if speech_count >= 0.6 * len(self.history) and self.current_type != "speech":
                print(f"Audio type changed: {self.current_type.upper()} → SPEECH (confidence: {confidence:.2f})")
                self.current_type = "speech"
            elif music_count >= 0.4 * len(self.history) and self.current_type != "music":
                print(f"Audio type changed: {self.current_type.upper()} → MUSIC (confidence: {confidence:.2f})")
                self.current_type = "music"
                
            return self.current_type, confidence
            
        except Exception as e:
            # If any error occurs during classification, stick with current type
            print(f"Classification error: {str(e)}")
            return self.current_type, 0.5