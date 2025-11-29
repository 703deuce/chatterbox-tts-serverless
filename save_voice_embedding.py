#!/usr/bin/env python3
"""
Save Voice Embedding Utility
Saves audio files as voice embeddings that can be used later without the WAV file.
"""

import json
import pickle
import gzip
import logging
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)

def generate_voice_id(audio_array: np.ndarray, voice_name: str) -> str:
    """Generate a unique 8-character hex voice ID"""
    # Create a hash from audio data and name
    combined = f"{voice_name}_{audio_array.shape[0]}_{audio_array[0] if len(audio_array) > 0 else 0}"
    hash_obj = hashlib.md5(combined.encode())
    return hash_obj.hexdigest()[:8]

def save_voice_embedding(
    audio_array: np.ndarray,
    sample_rate: int,
    voice_name: str,
    embeddings_dir: str = "voice_embeddings",
    voice_description: Optional[str] = None,
    gender: Optional[str] = None,
    original_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save a voice embedding from audio array
    
    Args:
        audio_array: Audio data as numpy array
        sample_rate: Sample rate of the audio
        voice_name: Name for the voice (e.g., "Amy", "Benjamin")
        embeddings_dir: Directory to save embeddings
        voice_description: Optional description
        gender: Optional gender ("male" or "female")
        original_file: Optional original filename
    
    Returns:
        Dictionary with voice info including voice_id
    """
    try:
        embeddings_path = Path(embeddings_dir)
        embeddings_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        embeddings_subdir = embeddings_path / "embeddings"
        metadata_subdir = embeddings_path / "metadata"
        embeddings_subdir.mkdir(exist_ok=True)
        metadata_subdir.mkdir(exist_ok=True)
        
        # Resample to 24kHz if needed (standard for ChatterboxTTS)
        target_sr = 24000
        if sample_rate != target_sr:
            logger.info(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        
        # Generate voice ID
        voice_id = generate_voice_id(audio_array, voice_name)
        
        # Check if voice already exists
        catalog_file = embeddings_path / "voice_catalog.json"
        voice_catalog = {}
        if catalog_file.exists():
            with open(catalog_file, 'r') as f:
                voice_catalog = json.load(f)
        
        # If voice with same ID exists, update it
        if voice_id in voice_catalog:
            logger.warning(f"Voice ID {voice_id} already exists, updating...")
        
        # Prepare embedding data (store as dict with audio key)
        embedding_data = {
            'audio': audio_array,
            'sample_rate': sample_rate,
            'voice_name': voice_name,
            'created_at': time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        # Save compressed embedding file
        embedding_file = embeddings_subdir / f"{voice_id}.pkl.gz"
        with gzip.open(embedding_file, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        logger.info(f"Saved embedding to {embedding_file}")
        
        # Calculate duration
        duration = len(audio_array) / sample_rate
        
        # Create voice metadata
        voice_metadata = {
            "voice_id": voice_id,
            "speaker_name": voice_name,
            "gender": gender or "unknown",
            "original_file": original_file or f"{voice_name}.wav",
            "embedding_file": f"embeddings/{voice_id}.pkl.gz",
            "duration": round(duration, 2),
            "sample_rate": sample_rate,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "type": "chattts_embedding"
        }
        
        if voice_description:
            voice_metadata["description"] = voice_description
        
        # Save individual metadata file
        metadata_file = metadata_subdir / f"{voice_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(voice_metadata, f, indent=2)
        
        # Update voice catalog
        voice_catalog[voice_id] = voice_metadata
        
        # Save updated catalog
        with open(catalog_file, 'w') as f:
            json.dump(voice_catalog, f, indent=2)
        
        # Update name mapping
        name_mapping_file = embeddings_path / "name_mapping.json"
        name_mapping = {}
        if name_mapping_file.exists():
            with open(name_mapping_file, 'r') as f:
                name_mapping = json.load(f)
        
        name_mapping[voice_name] = voice_id
        
        with open(name_mapping_file, 'w') as f:
            json.dump(name_mapping, f, indent=2)
        
        # Update stats
        stats_file = embeddings_path / "stats.json"
        stats = {
            "total_voices": len(voice_catalog),
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generation_time": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        # Calculate gender distribution
        gender_counts = {}
        for voice_info in voice_catalog.values():
            gender = voice_info.get('gender', 'unknown')
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        stats['gender_distribution'] = gender_counts
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✅ Voice '{voice_name}' saved successfully as ID: {voice_id}")
        
        return {
            "success": True,
            "voice_id": voice_id,
            "voice_name": voice_name,
            "embedding_file": str(embedding_file),
            "metadata_file": str(metadata_file),
            "duration": duration,
            "sample_rate": sample_rate
        }
        
    except Exception as e:
        logger.error(f"Failed to save voice embedding: {e}")
        raise

def save_voice_from_file(
    audio_file_path: str,
    voice_name: str,
    embeddings_dir: str = "voice_embeddings",
    voice_description: Optional[str] = None,
    gender: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save a voice embedding from an audio file
    
    Args:
        audio_file_path: Path to WAV/MP3 audio file
        voice_name: Name for the voice
        embeddings_dir: Directory to save embeddings
        voice_description: Optional description
        gender: Optional gender
    
    Returns:
        Dictionary with voice info
    """
    # Load audio file
    audio_array, sample_rate = librosa.load(audio_file_path, sr=None)
    
    # Get original filename
    original_file = Path(audio_file_path).name
    
    return save_voice_embedding(
        audio_array=audio_array,
        sample_rate=sample_rate,
        voice_name=voice_name,
        embeddings_dir=embeddings_dir,
        voice_description=voice_description,
        gender=gender,
        original_file=original_file
    )

# Example usage
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 3:
        print("Usage: python save_voice_embedding.py <audio_file> <voice_name> [gender] [description]")
        print("Example: python save_voice_embedding.py my_voice.wav MyVoice male 'My custom voice'")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    voice_name = sys.argv[2]
    gender = sys.argv[3] if len(sys.argv) > 3 else None
    description = sys.argv[4] if len(sys.argv) > 4 else None
    
    result = save_voice_from_file(
        audio_file_path=audio_file,
        voice_name=voice_name,
        voice_description=description,
        gender=gender
    )
    
    print(f"\n✅ Success! Voice saved:")
    print(f"   Voice ID: {result['voice_id']}")
    print(f"   Voice Name: {result['voice_name']}")
    print(f"   Duration: {result['duration']:.2f}s")
    print(f"   Sample Rate: {result['sample_rate']}Hz")
    print(f"\nYou can now use this voice with:")
    print(f'   voice_name: "{voice_name}"')
    print(f'   voice_id: "{result["voice_id"]}"')

