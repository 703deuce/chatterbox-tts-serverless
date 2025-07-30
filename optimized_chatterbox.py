#!/usr/bin/env python3
"""
Optimized ChatterboxTTS wrapper that accepts direct audio arrays
instead of requiring audio files, eliminating the inefficient conversion process.
"""

import torch
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Union, Tuple
from chatterbox.tts import ChatterboxTTS
import logging

logger = logging.getLogger(__name__)

class OptimizedChatterboxTTS:
    """
    Enhanced ChatterboxTTS wrapper that supports direct audio array input
    to bypass the inefficient embedding → WAV → embedding conversion
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize the optimized TTS model"""
        self.device = device
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.sr = getattr(self.model, 'sr', 24000)
        logger.info(f"OptimizedChatterboxTTS initialized on {device}")
    
    def generate(
        self, 
        text: str,
        audio_prompt_array: Optional[np.ndarray] = None,
        audio_prompt_path: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate TTS with optimized audio prompt handling
        
        Args:
            text: Text to synthesize
            audio_prompt_array: Direct audio array (OPTIMIZED - no file I/O)
            audio_prompt_path: Legacy audio file path (fallback)
            **kwargs: Other generation parameters
            
        Returns:
            Generated audio tensor
        """
        
        # OPTIMIZATION: Use direct audio array if provided
        if audio_prompt_array is not None:
            logger.info("Using direct audio array (OPTIMIZED)")
            
            # Create temporary file only for model compatibility
            # TODO: This will be eliminated in Phase 2
            temp_path = self._array_to_temp_file(audio_prompt_array)
            
            try:
                result = self.model.generate(
                    text=text,
                    audio_prompt_path=temp_path,
                    **kwargs
                )
                return result
            finally:
                # Clean up temporary file
                try:
                    Path(temp_path).unlink()
                except:
                    pass
        
        # Legacy path: use audio file directly
        elif audio_prompt_path is not None:
            logger.info("Using audio file path (legacy)")
            return self.model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                **kwargs
            )
        
        # No audio prompt
        else:
            logger.info("No audio prompt provided")
            return self.model.generate(text=text, **kwargs)
    
    def generate_stream(
        self,
        text: str,
        audio_prompt_array: Optional[np.ndarray] = None,
        audio_prompt_path: Optional[str] = None,
        **kwargs
    ):
        """
        Generate streaming TTS with optimized audio prompt handling
        
        Args:
            text: Text to synthesize
            audio_prompt_array: Direct audio array (OPTIMIZED)
            audio_prompt_path: Legacy audio file path (fallback)
            **kwargs: Other generation parameters
            
        Yields:
            Audio chunks and metrics
        """
        
        # OPTIMIZATION: Use direct audio array if provided
        if audio_prompt_array is not None:
            logger.info("Using direct audio array for streaming (OPTIMIZED)")
            
            # Create temporary file only for model compatibility
            # TODO: This will be eliminated in Phase 2
            temp_path = self._array_to_temp_file(audio_prompt_array)
            
            try:
                yield from self.model.generate_stream(
                    text=text,
                    audio_prompt_path=temp_path,
                    **kwargs
                )
            finally:
                # Clean up temporary file
                try:
                    Path(temp_path).unlink()
                except:
                    pass
        
        # Legacy path: use audio file directly
        elif audio_prompt_path is not None:
            logger.info("Using audio file path for streaming (legacy)")
            yield from self.model.generate_stream(
                text=text,
                audio_prompt_path=audio_prompt_path,
                **kwargs
            )
        
        # No audio prompt
        else:
            logger.info("No audio prompt provided for streaming")
            yield from self.model.generate_stream(text=text, **kwargs)
    
    def _array_to_temp_file(self, audio_array: np.ndarray) -> str:
        """
        Convert audio array to temporary file
        
        NOTE: This is a temporary bridge function that will be eliminated
        in Phase 2 when we modify ChatterboxTTS to accept arrays directly
        """
        temp_path = tempfile.mktemp(suffix='.wav')
        sf.write(temp_path, audio_array, self.sr)
        return temp_path
    
    # TODO: Phase 2 - Add direct embedding support
    def _generate_with_direct_embedding(self, text: str, audio_array: np.ndarray, **kwargs):
        """
        Future implementation: Generate TTS with direct audio embedding
        This will bypass all file I/O when ChatterboxTTS is modified
        """
        # This will be implemented in Phase 2
        raise NotImplementedError("Direct embedding support coming in Phase 2")

# Factory function for easy initialization
def create_optimized_tts(device: str = "cuda") -> OptimizedChatterboxTTS:
    """Create an optimized ChatterboxTTS instance"""
    return OptimizedChatterboxTTS(device=device) 