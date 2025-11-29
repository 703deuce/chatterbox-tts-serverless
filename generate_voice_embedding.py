#!/usr/bin/env python3
"""
Standalone Voice Embedding Generator
Generate voice embeddings that work with the Chatterbox TTS handler.

This module can be used by external applications to create voice embeddings
that are compatible with the existing handler without any modifications.

Usage:
    from generate_voice_embedding import create_voice_embedding
    
    # From audio file
    result = create_voice_embedding.from_file(
        audio_file="my_voice.wav",
        voice_name="MyVoice",
        embeddings_dir="./voice_embeddings",
        gender="male"
    )
    
    # From audio array (numpy)
    result = create_voice_embedding.from_array(
        audio_array=audio_data,
        sample_rate=44100,
        voice_name="MyVoice",
        embeddings_dir="./voice_embeddings"
    )
    
    # From base64 audio
    result = create_voice_embedding.from_base64(
        audio_base64="UklGRiQAAABXQVZFZm10IBAAAAAB...",
        voice_name="MyVoice",
        embeddings_dir="./voice_embeddings"
    )
"""

import json
import pickle
import gzip
import hashlib
import time
import base64
import io
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np

# Optional dependencies - will raise helpful error if missing
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None

# Note: We don't need ChatTTS here - we store raw audio arrays.
# ChatTTS extracts embeddings internally when you pass it audio in the handler.


class VoiceEmbeddingGenerator:
    """
    Standalone voice embedding generator compatible with Chatterbox TTS handler.
    
    Creates embeddings in the exact format expected by the handler:
    - Compressed pickle files (.pkl.gz) with audio arrays
    - Metadata JSON files
    - Updated voice catalog and name mapping
    """
    
    # Standard sample rate for ChatterboxTTS
    TARGET_SAMPLE_RATE = 24000
    
    def __init__(self, embeddings_dir: str = "voice_embeddings"):
        """
        Initialize the generator
        
        Args:
            embeddings_dir: Directory where embeddings will be saved
                           (should match handler's voice_embeddings directory)
        
        Note: This stores raw audio arrays, not extracted embeddings.
        ChatterboxTTS extracts speaker embeddings internally when you pass it audio,
        so storing raw audio is the correct approach.
        """
        self.embeddings_dir = Path(embeddings_dir)
        self._ensure_dependencies()
    
    def _ensure_dependencies(self):
        """Check that required dependencies are available"""
        if librosa is None:
            raise ImportError(
                "librosa is required. Install with: pip install librosa"
            )
        if sf is None:
            raise ImportError(
                "soundfile is required. Install with: pip install soundfile"
            )
    
    def _generate_voice_id(self, audio_array: np.ndarray, voice_name: str) -> str:
        """
        Generate a unique 8-character hex voice ID
        
        Args:
            audio_array: Audio data
            voice_name: Voice name
            
        Returns:
            8-character hex string
        """
        # Create hash from audio characteristics and name
        audio_hash = hashlib.sha256(
            audio_array.tobytes() + voice_name.encode()
        ).hexdigest()
        return audio_hash[:8]
    
    def _resample_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Resample audio to target sample rate (24kHz)
        
        Args:
            audio_array: Audio data
            sample_rate: Current sample rate
            
        Returns:
            Resampled audio array at 24kHz
        """
        if sample_rate == self.TARGET_SAMPLE_RATE:
            return audio_array
        
        return librosa.resample(
            audio_array,
            orig_sr=sample_rate,
            target_sr=self.TARGET_SAMPLE_RATE
        )
    
    def _load_voice_catalog(self) -> Dict[str, Any]:
        """Load existing voice catalog"""
        catalog_file = self.embeddings_dir / "voice_catalog.json"
        if catalog_file.exists():
            with open(catalog_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_voice_catalog(self, catalog: Dict[str, Any]):
        """Save voice catalog"""
        catalog_file = self.embeddings_dir / "voice_catalog.json"
        with open(catalog_file, 'w') as f:
            json.dump(catalog, f, indent=2)
    
    def _load_name_mapping(self) -> Dict[str, str]:
        """Load name to ID mapping"""
        mapping_file = self.embeddings_dir / "name_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_name_mapping(self, mapping: Dict[str, str]):
        """Save name to ID mapping"""
        mapping_file = self.embeddings_dir / "name_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
    
    def _update_stats(self, catalog: Dict[str, Any]):
        """Update statistics file"""
        stats_file = self.embeddings_dir / "stats.json"
        
        # Calculate gender distribution
        gender_counts = {}
        for voice_info in catalog.values():
            gender = voice_info.get('gender', 'unknown')
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        
        stats = {
            "total_voices": len(catalog),
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gender_distribution": gender_counts
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def create(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        voice_name: str,
        gender: Optional[str] = None,
        voice_description: Optional[str] = None,
        original_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create and save a voice embedding from audio array
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            voice_name: Name for the voice (must be unique)
            gender: Optional gender ("male" or "female")
            voice_description: Optional description
            original_file: Optional original filename
            
        Returns:
            Dictionary with:
            - success: bool
            - voice_id: str (8-char hex ID)
            - voice_name: str
            - embedding_path: str (relative path to embedding file)
            - duration: float (seconds)
            - sample_rate: int
        """
        # Ensure directories exist
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        (self.embeddings_dir / "embeddings").mkdir(exist_ok=True)
        (self.embeddings_dir / "metadata").mkdir(exist_ok=True)
        
        # Resample to 24kHz
        audio_array = self._resample_audio(audio_array, sample_rate)
        sample_rate = self.TARGET_SAMPLE_RATE
        
        # Generate voice ID
        voice_id = self._generate_voice_id(audio_array, voice_name)
        
        # Load existing catalog
        catalog = self._load_voice_catalog()
        
        # Check if name already exists (warn but allow)
        name_mapping = self._load_name_mapping()
        if voice_name in name_mapping and name_mapping[voice_name] != voice_id:
            print(f"⚠️  Warning: Voice name '{voice_name}' already exists with different ID")
            print(f"   Existing ID: {name_mapping[voice_name]}, New ID: {voice_id}")
            print(f"   The new voice will overwrite the name mapping.")
        
        # Prepare embedding data (format expected by handler)
        # Note: We store raw audio arrays, not extracted embeddings.
        # ChatterboxTTS extracts speaker embeddings internally when you pass it audio,
        # so storing raw audio is the correct and simpler approach.
        embedding_data = {
            'audio': audio_array,  # Raw audio array (ChatTTS will extract embeddings internally)
            'sample_rate': sample_rate,
            'voice_name': voice_name,
            'created_at': time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        # Save compressed embedding file
        embedding_filename = f"{voice_id}.pkl.gz"
        embedding_path = self.embeddings_dir / "embeddings" / embedding_filename
        with gzip.open(embedding_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        # Calculate duration
        duration = len(audio_array) / sample_rate
        
        # Create metadata (format expected by handler)
        voice_metadata = {
            "voice_id": voice_id,
            "speaker_name": voice_name,
            "gender": gender or "unknown",
            "original_file": original_file or f"{voice_name}.wav",
            "embedding_file": f"embeddings/{embedding_filename}",  # Relative path (for reference)
            "duration": round(duration, 2),
            "sample_rate": sample_rate,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "type": "chattts_embedding"
            # Note: firebase_storage_path and firebase_storage_bucket should be added
            # by the external app after uploading to Firebase Storage
        }
        
        if voice_description:
            voice_metadata["description"] = voice_description
        
        # Save individual metadata file
        metadata_path = self.embeddings_dir / "metadata" / f"{voice_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(voice_metadata, f, indent=2)
        
        # Update catalog
        catalog[voice_id] = voice_metadata
        self._save_voice_catalog(catalog)
        
        # Update name mapping
        name_mapping[voice_name] = voice_id
        self._save_name_mapping(name_mapping)
        
        # Update stats
        self._update_stats(catalog)
        
        return {
            "success": True,
            "voice_id": voice_id,
            "voice_name": voice_name,
            "embedding_path": f"embeddings/{embedding_filename}",
            "metadata_path": f"metadata/{voice_id}.json",
            "duration": round(duration, 2),
            "sample_rate": sample_rate,
            "message": f"Voice '{voice_name}' saved successfully"
        }
    
    def from_file(
        self,
        audio_file: Union[str, Path],
        voice_name: str,
        gender: Optional[str] = None,
        voice_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create embedding from audio file
        
        Args:
            audio_file: Path to audio file (WAV, MP3, etc.)
            voice_name: Name for the voice
            gender: Optional gender
            voice_description: Optional description
            
        Returns:
            Result dictionary
        """
        # Load audio file
        audio_array, sample_rate = librosa.load(str(audio_file), sr=None)
        original_file = Path(audio_file).name
        
        return self.create(
            audio_array=audio_array,
            sample_rate=sample_rate,
            voice_name=voice_name,
            gender=gender,
            voice_description=voice_description,
            original_file=original_file
        )
    
    def from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        voice_name: str,
        gender: Optional[str] = None,
        voice_description: Optional[str] = None,
        original_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create embedding from numpy array
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate
            voice_name: Name for the voice
            gender: Optional gender
            voice_description: Optional description
            original_file: Optional original filename
            
        Returns:
            Result dictionary
        """
        return self.create(
            audio_array=audio_array,
            sample_rate=sample_rate,
            voice_name=voice_name,
            gender=gender,
            voice_description=voice_description,
            original_file=original_file
        )
    
    def from_base64(
        self,
        audio_base64: str,
        voice_name: str,
        gender: Optional[str] = None,
        voice_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create embedding from base64 encoded audio
        
        Args:
            audio_base64: Base64 encoded audio data
            voice_name: Name for the voice
            gender: Optional gender
            voice_description: Optional description
            
        Returns:
            Result dictionary
        """
        # Decode base64
        audio_data = base64.b64decode(audio_base64)
        audio_buffer = io.BytesIO(audio_data)
        
        # Load audio
        audio_array, sample_rate = sf.read(audio_buffer)
        
        return self.create(
            audio_array=audio_array,
            sample_rate=sample_rate,
            voice_name=voice_name,
            gender=gender,
            voice_description=voice_description
        )


# Convenience functions for easy import
def create_voice_embedding(
    audio_input: Union[str, Path, np.ndarray, bytes],
    voice_name: str,
    embeddings_dir: str = "voice_embeddings",
    gender: Optional[str] = None,
    voice_description: Optional[str] = None,
    sample_rate: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to create voice embedding from various input types
    
    Args:
        audio_input: Can be:
                    - File path (str/Path) → loads from file
                    - numpy array → uses directly
                    - base64 string → decodes and uses
        voice_name: Name for the voice
        embeddings_dir: Directory for embeddings
        gender: Optional gender
        voice_description: Optional description
        sample_rate: Required if audio_input is numpy array
        
    Returns:
        Result dictionary
    """
    generator = VoiceEmbeddingGenerator(embeddings_dir)
    
    if isinstance(audio_input, (str, Path)):
        # File path
        return generator.from_file(
            audio_file=audio_input,
            voice_name=voice_name,
            gender=gender,
            voice_description=voice_description
        )
    elif isinstance(audio_input, np.ndarray):
        # Numpy array
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio_input is a numpy array")
        return generator.from_array(
            audio_array=audio_input,
            sample_rate=sample_rate,
            voice_name=voice_name,
            gender=gender,
            voice_description=voice_description
        )
    elif isinstance(audio_input, str) and len(audio_input) > 100:
        # Assume base64 string
        return generator.from_base64(
            audio_base64=audio_input,
            voice_name=voice_name,
            gender=gender,
            voice_description=voice_description
        )
    else:
        raise ValueError(
            f"Unsupported audio_input type: {type(audio_input)}. "
            "Use file path, numpy array, or base64 string."
        )


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Standalone Voice Embedding Generator")
        print("=" * 50)
        print("\nUsage:")
        print("  python generate_voice_embedding.py <audio_file> <voice_name> [gender] [description]")
        print("\nExample:")
        print("  python generate_voice_embedding.py my_voice.wav MyVoice male 'My custom voice'")
        print("\nOr use as a module:")
        print("  from generate_voice_embedding import create_voice_embedding")
        print("  result = create_voice_embedding('my_voice.wav', 'MyVoice')")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    voice_name = sys.argv[2]
    gender = sys.argv[3] if len(sys.argv) > 3 else None
    description = sys.argv[4] if len(sys.argv) > 4 else None
    
    try:
        generator = VoiceEmbeddingGenerator()
        result = generator.from_file(
            audio_file=audio_file,
            voice_name=voice_name,
            gender=gender,
            voice_description=description
        )
        
        print("\n✅ Voice embedding created successfully!")
        print(f"   Voice ID: {result['voice_id']}")
        print(f"   Voice Name: {result['voice_name']}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Sample Rate: {result['sample_rate']}Hz")
        print(f"\n📁 Files created:")
        print(f"   - {result['embedding_path']}")
        print(f"   - {result['metadata_path']}")
        print(f"\n🎤 You can now use this voice in your handler:")
        print(f'   voice_name: "{voice_name}"')
        print(f'   voice_id: "{result["voice_id"]}"')
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

