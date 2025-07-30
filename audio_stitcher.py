#!/usr/bin/env python3
"""
Audio Stitching Module
Handles seamless stitching of audio segments with crossfade support
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not available - using basic numpy stitching")

import soundfile as sf
import io

logger = logging.getLogger(__name__)

@dataclass
class AudioSegmentData:
    """Represents an audio segment with metadata"""
    index: int
    audio: np.ndarray
    sample_rate: int
    engine: str
    tag_type: Optional[str] = None
    duration: float = 0.0

class AudioStitcher:
    """Handles stitching of audio segments with crossfades"""
    
    def __init__(self, crossfade_ms: int = 100, normalize_segments: bool = True):
        self.crossfade_ms = crossfade_ms
        self.normalize_segments = normalize_segments
        self.sample_rate = 24000  # Default sample rate
        
    def stitch_segments(
        self, 
        segments: List[AudioSegmentData],
        output_sample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Stitch audio segments together with crossfades
        
        Args:
            segments: List of AudioSegmentData to stitch
            output_sample_rate: Target sample rate for output
            
        Returns:
            Tuple of (stitched_audio, sample_rate)
        """
        if not segments:
            raise ValueError("No segments to stitch")
        
        # Sort segments by index to maintain order
        segments = sorted(segments, key=lambda x: x.index)
        
        # Use the sample rate from the first segment if not specified
        target_sr = output_sample_rate or segments[0].sample_rate
        self.sample_rate = target_sr
        
        logger.info(f"Stitching {len(segments)} segments with {self.crossfade_ms}ms crossfade")
        
        if PYDUB_AVAILABLE:
            return self._stitch_with_pydub(segments, target_sr)
        else:
            return self._stitch_with_numpy(segments, target_sr)
    
    def _stitch_with_pydub(
        self, 
        segments: List[AudioSegmentData], 
        target_sr: int
    ) -> Tuple[np.ndarray, int]:
        """Stitch segments using pydub for advanced audio processing"""
        try:
            # Convert first segment to pydub AudioSegment
            result_audio = self._numpy_to_pydub(segments[0].audio, segments[0].sample_rate)
            
            for i in range(1, len(segments)):
                current_segment = segments[i]
                previous_segment = segments[i-1]
                
                # Convert current segment to pydub
                current_audio = self._numpy_to_pydub(
                    current_segment.audio, 
                    current_segment.sample_rate
                )
                
                # Determine if crossfade is needed
                needs_crossfade = self._needs_crossfade(previous_segment, current_segment)
                
                if needs_crossfade and self.crossfade_ms > 0:
                    # Apply crossfade
                    logger.debug(f"Applying {self.crossfade_ms}ms crossfade between segments {i-1} and {i}")
                    result_audio = result_audio.append(current_audio, crossfade=self.crossfade_ms)
                else:
                    # Simple concatenation
                    result_audio = result_audio + current_audio
            
            # Normalize if requested
            if self.normalize_segments:
                result_audio = normalize(result_audio)
            
            # Convert back to numpy
            return self._pydub_to_numpy(result_audio, target_sr)
            
        except Exception as e:
            logger.warning(f"pydub stitching failed: {e}, falling back to numpy")
            return self._stitch_with_numpy(segments, target_sr)
    
    def _stitch_with_numpy(
        self, 
        segments: List[AudioSegmentData], 
        target_sr: int
    ) -> Tuple[np.ndarray, int]:
        """Stitch segments using numpy (fallback method)"""
        result_segments = []
        
        for segment in segments:
            # Resample if needed
            audio = self._resample_audio(segment.audio, segment.sample_rate, target_sr)
            
            # Trim silence from edges if needed
            audio = self._trim_silence(audio, target_sr)
            
            result_segments.append(audio)
        
        # Apply crossfades between segments
        if len(result_segments) > 1 and self.crossfade_ms > 0:
            result_audio = self._apply_numpy_crossfades(result_segments, target_sr)
        else:
            # Simple concatenation
            result_audio = np.concatenate(result_segments)
        
        # Normalize if requested
        if self.normalize_segments:
            max_val = np.max(np.abs(result_audio))
            if max_val > 0:
                result_audio = result_audio / max_val
        
        return result_audio, target_sr
    
    def _apply_numpy_crossfades(
        self, 
        segments: List[np.ndarray], 
        sample_rate: int
    ) -> np.ndarray:
        """Apply crossfades between numpy audio segments"""
        crossfade_samples = int(self.crossfade_ms * sample_rate / 1000)
        
        result = segments[0]
        
        for i in range(1, len(segments)):
            current_segment = segments[i]
            
            # Apply crossfade if both segments are long enough
            if len(result) >= crossfade_samples and len(current_segment) >= crossfade_samples:
                # Create fade-out for end of result
                fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                result[-crossfade_samples:] *= fade_out
                
                # Create fade-in for start of current segment
                fade_in = np.linspace(0.0, 1.0, crossfade_samples)
                current_segment[:crossfade_samples] *= fade_in
                
                # Overlap and add
                overlap_section = result[-crossfade_samples:] + current_segment[:crossfade_samples]
                result = np.concatenate([
                    result[:-crossfade_samples],
                    overlap_section,
                    current_segment[crossfade_samples:]
                ])
            else:
                # Simple concatenation if segments are too short for crossfade
                result = np.concatenate([result, current_segment])
        
        return result
    
    def _needs_crossfade(
        self, 
        prev_segment: AudioSegmentData, 
        current_segment: AudioSegmentData
    ) -> bool:
        """Determine if crossfade is needed between two segments"""
        # Apply crossfade when switching between different engines or tag types
        if prev_segment.engine != current_segment.engine:
            return True
        
        if prev_segment.tag_type != current_segment.tag_type:
            return True
        
        # Also apply for F5-TTS segments to ensure smooth transitions
        if current_segment.engine == 'f5':
            return True
        
        return False
    
    def _trim_silence(
        self, 
        audio: np.ndarray, 
        sample_rate: int, 
        threshold_db: float = -40.0
    ) -> np.ndarray:
        """Trim silence from beginning and end of audio"""
        if len(audio) == 0:
            return audio
        
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
        
        # Find non-silent regions
        non_silent = audio_db > threshold_db
        
        if not np.any(non_silent):
            return audio  # Return as-is if all silent
        
        # Find first and last non-silent samples
        non_silent_indices = np.where(non_silent)[0]
        start_idx = max(0, non_silent_indices[0] - int(0.1 * sample_rate))  # Keep 0.1s padding
        end_idx = min(len(audio), non_silent_indices[-1] + int(0.1 * sample_rate))
        
        return audio[start_idx:end_idx]
    
    def _resample_audio(
        self, 
        audio: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio
        
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            logger.warning("librosa not available for resampling - using simple interpolation")
            # Simple linear interpolation as fallback
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, range(len(audio)), audio)
    
    def _numpy_to_pydub(self, audio: np.ndarray, sample_rate: int) -> AudioSegment:
        """Convert numpy array to pydub AudioSegment"""
        # Ensure audio is in correct format for pydub
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Create pydub AudioSegment
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio_int16.dtype.itemsize,
            channels=1
        )
        
        return audio_segment
    
    def _pydub_to_numpy(
        self, 
        audio_segment: AudioSegment, 
        target_sr: int
    ) -> Tuple[np.ndarray, int]:
        """Convert pydub AudioSegment to numpy array"""
        # Resample if needed
        if audio_segment.frame_rate != target_sr:
            audio_segment = audio_segment.set_frame_rate(target_sr)
        
        # Convert to numpy
        audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        audio_array = audio_array / 32767.0  # Normalize to [-1, 1]
        
        return audio_array, target_sr

def create_audio_stitcher(
    crossfade_ms: int = 100, 
    normalize_segments: bool = True
) -> AudioStitcher:
    """Factory function to create AudioStitcher instance"""
    return AudioStitcher(crossfade_ms=crossfade_ms, normalize_segments=normalize_segments)

# Example usage and testing
if __name__ == "__main__":
    # Test the stitcher
    stitcher = create_audio_stitcher(crossfade_ms=150)
    
    # Create mock audio segments for testing
    sample_rate = 24000
    duration = 2.0  # seconds
    
    segments = []
    for i in range(3):
        # Generate test audio (sine wave)
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440 + i * 110  # Different frequencies
        audio = np.sin(2 * np.pi * frequency * t) * 0.5
        
        segment = AudioSegmentData(
            index=i,
            audio=audio,
            sample_rate=sample_rate,
            engine='chatterbox' if i % 2 == 0 else 'f5',
            tag_type='normal' if i % 2 == 0 else 'excited',
            duration=duration
        )
        segments.append(segment)
    
    # Test stitching
    try:
        result_audio, result_sr = stitcher.stitch_segments(segments)
        print(f"Stitched audio: {len(result_audio)} samples at {result_sr}Hz")
        print(f"Total duration: {len(result_audio) / result_sr:.2f}s")
        
        # Save test result
        output_path = "test_stitched_audio.wav"
        sf.write(output_path, result_audio, result_sr)
        print(f"Saved test audio to {output_path}")
        
    except Exception as e:
        print(f"Stitching test failed: {e}") 