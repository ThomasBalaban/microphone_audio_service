# transcriber_core/__init__.py
"""
Transcription Core Module

Microphone: Uses Parakeet MLX streaming for real-time transcription.
Desktop Audio: Uses faster-whisper with VAD-based batch processing.
"""

from .audio_manager import TranscriptManager
from .classifier import SpeechMusicClassifier
from .microphone import MicrophoneTranscriber
from .desktop_transcriber import SpeechMusicTranscriber as DesktopTranscriber
from . import config

__all__ = [
    'TranscriptManager',
    'SpeechMusicClassifier', 
    'MicrophoneTranscriber',
    'DesktopTranscriber',
    'config'
]