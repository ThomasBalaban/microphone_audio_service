import time
import os
import datetime
from typing import Dict, Any, Optional

class TranscriptManager:
    """
    Manages transcription data without long-term storage functionality.
    """
    def __init__(self):
        self.recent_transcripts = []
        self.max_recent = 20  # Keep only the 20 most recent transcripts in memory
        
    def publish_transcript(self, source: str, text: str, 
                         timestamp: Optional[str] = None, 
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Process a transcript without long-term storage.
        
        Args:
            source: Source of the transcript (e.g., "desktop", "microphone")
            text: The transcribed text
            timestamp: ISO format timestamp (if None, current time is used)
            metadata: Additional metadata to store with the transcript
        """
        if not text.strip():
            return  # Skip empty transcripts
            
        # Create timestamp if not provided
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
            
        # Create the message
        message = {
            "source": source,
            "text": text,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        # Add to in-memory recent transcripts
        self.recent_transcripts.append(message)
        
        # Trim to keep only the most recent ones
        if len(self.recent_transcripts) > self.max_recent:
            self.recent_transcripts = self.recent_transcripts[-self.max_recent:]
    
    def get_recent_transcripts(self, limit=10, source=None):
        """
        Get recent transcripts from memory.
        
        Args:
            limit: Maximum number of transcripts to return
            source: Filter by source (e.g., "desktop", "microphone")
            
        Returns:
            List of transcript documents
        """
        if limit > self.max_recent:
            limit = self.max_recent
            
        # Filter by source if specified
        if source:
            filtered = [t for t in self.recent_transcripts if t["source"] == source]
        else:
            filtered = self.recent_transcripts
            
        # Return most recent first, up to limit
        return filtered[-limit:][::-1]
    
    def search_transcripts(self, query_text, limit=10):
        """
        Simple in-memory search for transcripts by text content.
        
        Args:
            query_text: Text to search for (case-insensitive)
            limit: Maximum number of results to return
            
        Returns:
            List of matching transcript documents
        """
        query_lower = query_text.lower()
        matches = []
        
        for transcript in self.recent_transcripts:
            if query_lower in transcript["text"].lower():
                matches.append(transcript)
                if len(matches) >= limit:
                    break
                    
        return matches
    
    def cleanup_audio_files(self):
        """
        Clean up all files in the audio_captures folder at the root level.
        """
        # Determine the path to the audio_captures folder (one folder up from current module)
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        captures_dir = os.path.join(current_dir, 'audio_captures')
        
        # Count files for reporting
        file_count = 0
        
        # Remove all files in the directory
        for filename in os.listdir(captures_dir):
            file_path = os.path.join(captures_dir, filename)
            # Check if it's a file (not a subdirectory)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    file_count += 1
                except Exception as e:
                    print(f"Error removing file {file_path}: {str(e)}")
        
        if file_count > 0:
            print(f"Cleaned up {file_count} audio files from {captures_dir}")
    
    def close(self):
        """Clean up resources"""
        # Clean up audio files
        self.cleanup_audio_files()
        
        # Clear memory
        self.recent_transcripts = []