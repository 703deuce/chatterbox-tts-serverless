#!/usr/bin/env python3
"""
Standalone Audio Preparation for External Applications

This script prepares audio arrays that can be sent directly to the Chatterbox TTS API.
It does NOT require ChatterboxTTS to be installed - only basic audio libraries.

The prepared audio can be:
1. Sent as base64 in the 'reference_audio' field
2. Or saved to the voice_embeddings directory for use with voice_id/voice_name

Usage:
    from standalone_audio_prep import prepare_audio_for_api
    
    # Prepare audio from file
    audio_array, sample_rate = prepare_audio_for_api("my_voice.wav")
    
    # Convert to base64 for API
    import base64
    import soundfile as sf
    import io
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format='WAV')
    audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Send to API
    api_request = {
        "text": "Hello world",
        "reference_audio": audio_b64
    }
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import audio libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - install with: pip install librosa")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("soundfile not available - install with: pip install soundfile")


# Standard sample rate for ChatterboxTTS
TARGET_SAMPLE_RATE = 24000
MAX_DURATION_SECONDS = 30


def prepare_audio_for_api(
    audio_input: Union[str, Path, np.ndarray, bytes],
    sample_rate: Optional[int] = None,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    max_duration_seconds: int = MAX_DURATION_SECONDS,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Prepare audio for use with Chatterbox TTS API
    
    This function:
    1. Loads audio from various formats
    2. Resamples to target sample rate (24kHz default)
    3. Trims to max duration
    4. Normalizes audio
    5. Returns audio array ready for API
    
    Args:
        audio_input: Audio file path, numpy array, or base64 bytes
        sample_rate: Sample rate (required if audio_input is numpy array)
        target_sample_rate: Target sample rate (default: 24000)
        max_duration_seconds: Maximum duration in seconds (default: 30)
        normalize: Whether to normalize audio (default: True)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa is required. Install with: pip install librosa soundfile"
        )
    
    # Handle different input types
    if isinstance(audio_input, (str, Path)):
        # File path
        logger.info(f"Loading audio from file: {audio_input}")
        audio_array, sr = librosa.load(str(audio_input), sr=None)
        
    elif isinstance(audio_input, np.ndarray):
        # Numpy array
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio_input is a numpy array")
        audio_array = audio_input.copy()
        sr = sample_rate
        
    elif isinstance(audio_input, bytes):
        # Base64 bytes or WAV bytes
        import io
        if not SOUNDFILE_AVAILABLE:
            raise ImportError("soundfile is required for bytes input")
        
        audio_buffer = io.BytesIO(audio_input)
        audio_array, sr = sf.read(audio_buffer)
        
    else:
        raise ValueError(f"Unsupported audio_input type: {type(audio_input)}")
    
    # Resample to target sample rate if needed
    if sr != target_sample_rate:
        logger.info(f"Resampling from {sr}Hz to {target_sample_rate}Hz")
        audio_array = librosa.resample(
            audio_array,
            orig_sr=sr,
            target_sr=target_sample_rate,
            res_type='kaiser_best'
        )
        sr = target_sample_rate
    
    # Trim to max duration
    max_samples = int(max_duration_seconds * sr)
    if len(audio_array) > max_samples:
        logger.info(f"Trimming audio from {len(audio_array)/sr:.2f}s to {max_duration_seconds}s")
        audio_array = audio_array[:max_samples]
    
    # Normalize if requested
    if normalize:
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            # Normalize to [-1, 1] range
            audio_array = audio_array / max_val
            logger.info("Audio normalized to [-1, 1] range")
    
    # Ensure mono
    if len(audio_array.shape) > 1:
        logger.info("Converting to mono")
        audio_array = np.mean(audio_array, axis=1)
    
    logger.info(f"Audio prepared: {len(audio_array)} samples at {sr}Hz ({len(audio_array)/sr:.2f}s)")
    
    return audio_array, sr


def audio_to_base64(
    audio_array: np.ndarray,
    sample_rate: int,
    format: str = 'WAV'
) -> str:
    """
    Convert audio array to base64 string for API
    
    Args:
        audio_array: Audio array
        sample_rate: Sample rate
        format: Audio format ('WAV', 'FLAC', etc.)
    
    Returns:
        Base64 encoded audio string
    """
    if not SOUNDFILE_AVAILABLE:
        raise ImportError("soundfile is required for base64 conversion")
    
    import base64
    import io
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format=format)
    buffer.seek(0)
    
    audio_bytes = buffer.read()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    return audio_b64


def base64_to_audio(audio_b64: str) -> Tuple[np.ndarray, int]:
    """
    Convert base64 string to audio array
    
    Args:
        audio_b64: Base64 encoded audio string
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    if not SOUNDFILE_AVAILABLE:
        raise ImportError("soundfile is required for base64 conversion")
    
    import base64
    import io
    
    audio_data = base64.b64decode(audio_b64)
    audio_buffer = io.BytesIO(audio_data)
    audio_array, sample_rate = sf.read(audio_buffer)
    
    return audio_array, sample_rate


def prepare_and_encode_for_api(
    audio_input: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    max_duration_seconds: int = MAX_DURATION_SECONDS
) -> str:
    """
    Prepare audio and encode to base64 in one step
    
    This is a convenience function that combines prepare_audio_for_api and audio_to_base64.
    
    Args:
        audio_input: Audio file path or numpy array
        sample_rate: Sample rate (required if audio_input is numpy array)
        target_sample_rate: Target sample rate (default: 24000)
        max_duration_seconds: Maximum duration in seconds (default: 30)
    
    Returns:
        Base64 encoded audio string ready for API
    """
    audio_array, sr = prepare_audio_for_api(
        audio_input,
        sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
        max_duration_seconds=max_duration_seconds
    )
    
    return audio_to_base64(audio_array, sr)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Standalone Audio Preparation for Chatterbox TTS API")
        print("=" * 60)
        print("\nThis script prepares audio for use with the Chatterbox TTS API.")
        print("It does NOT require ChatterboxTTS - only librosa and soundfile.")
        print("\nUsage:")
        print("  python standalone_audio_prep.py <audio_file> [output_file]")
        print("\nExample:")
        print("  python standalone_audio_prep.py my_voice.wav")
        print("\nOr use as a module:")
        print("  from standalone_audio_prep import prepare_and_encode_for_api")
        print("  audio_b64 = prepare_and_encode_for_api('my_voice.wav')")
        print("  # Send audio_b64 in API request as 'reference_audio'")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Prepare audio
        audio_array, sample_rate = prepare_audio_for_api(audio_file)
        
        # Encode to base64
        audio_b64 = audio_to_base64(audio_array, sample_rate)
        
        print(f"\n✅ Audio prepared successfully!")
        print(f"   Duration: {len(audio_array)/sample_rate:.2f}s")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Samples: {len(audio_array)}")
        print(f"   Base64 length: {len(audio_b64)} characters")
        
        # Save base64 to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(audio_b64)
            print(f"\n✅ Base64 saved to: {output_file}")
        else:
            print(f"\n📋 Base64 (first 100 chars): {audio_b64[:100]}...")
            print(f"\n💡 Use this in your API request:")
            print(f"   {{'text': 'Hello world', 'reference_audio': '{audio_b64[:50]}...'}}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

