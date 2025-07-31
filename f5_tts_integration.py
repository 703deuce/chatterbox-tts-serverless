#!/usr/bin/env python3
"""
F5-TTS Integration Module
Provides a wrapper for F5-TTS with expressive tag support
"""

import torch
import numpy as np
import logging
import time
from typing import Optional, Dict, Any, Union
from pathlib import Path

# Configure logging first
logger = logging.getLogger(__name__)

try:
    # Try the main F5-TTS API imports (official pattern)
    from f5_tts.api import F5TTS
    from f5_tts.infer.utils_infer import load_model, infer_process
    
    # Additional imports for direct inference
    import tempfile
    import torchaudio
    F5_AVAILABLE = True
    logger.info("F5-TTS successfully imported")
except ImportError as e:
    F5_AVAILABLE = False
    logger.warning(f"F5-TTS not available - expressive tags will fall back to Chatterbox. Import error: {e}")
except Exception as e:
    F5_AVAILABLE = False
    logger.warning(f"F5-TTS import failed - expressive tags will fall back to Chatterbox. Error: {e}")

class F5TTSWrapper:
    """Wrapper class for F5-TTS with expressive capabilities"""
    
    def __init__(self, device: str = "cuda", model_type: str = "F5TTS_v1_Base"):
        self.device = device
        self.model_type = model_type
        self.model = None
        self.is_loaded = False
        
        # Native F5-TTS expressive tags (built-in support)
        self.native_tags = {
            'whisper': {
                'description': 'Whispered speech',
                'builtin': True
            },
            'shout': {
                'description': 'Yelling/shouting', 
                'builtin': True
            },
            'happy': {
                'description': 'Happy tone',
                'builtin': True
            },
            'sad': {
                'description': 'Sad tone', 
                'builtin': True
            },
            'angry': {
                'description': 'Angry tone',
                'builtin': True
            }
        }
        
        # Extended tags mapped to native F5-TTS tags
        self.extended_tags = {
            'excited': 'happy',
            'calm': 'whisper', 
            'nervous': 'whisper',
            'confident': 'happy',
            'loud': 'shout',
            'quiet': 'whisper'
        }
        
        # Audio effects for expressive enhancement
        self.expressive_effects = {
            'whisper': {
                'speed': 0.9,
                'volume_scale': 0.7,
                'pitch_shift': -0.1,
                'emphasis': 0.3
            },
            'shout': {
                'speed': 1.1,
                'volume_scale': 1.3,
                'pitch_shift': 0.1,
                'emphasis': 0.8
            },
            'sad': {
                'speed': 0.85,
                'volume_scale': 0.8,
                'pitch_shift': -0.05,
                'emphasis': 0.4
            },
            'happy': {
                'speed': 1.05,
                'volume_scale': 1.1,
                'pitch_shift': 0.05,
                'emphasis': 0.7
            }
        }
    
    def load_model(self):
        """Load F5-TTS model from local files downloaded during Docker build"""
        if not F5_AVAILABLE:
            logger.error("F5-TTS not available - cannot load model")
            return False
        
        try:
            logger.info(f"Loading F5-TTS model ({self.model_type}) on {self.device}")
            start_time = time.time()
            
            # Check for pre-downloaded models (downloaded during Docker build like S3Gen)
            local_model_dir = "/workspace/f5_models/F5TTS_v1_Base"
            local_model_file = f"{local_model_dir}/model_1250000.safetensors"
            local_vocab_file = f"{local_model_dir}/vocab.txt"
            
            # Verify local files exist (should be downloaded during build)
            if not Path(local_model_file).exists():
                logger.error(f"F5-TTS model file not found: {local_model_file}")
                logger.error("F5-TTS models must be downloaded during Docker build")
                return False
                
            if not Path(local_vocab_file).exists():
                logger.error(f"F5-TTS vocab file not found: {local_vocab_file}")
                return False
            
            logger.info("Loading F5-TTS from pre-downloaded local models...")
            
            # Load F5-TTS with local files (like ChatterboxTTS with S3Gen)
            self.model = F5TTS(
                model_type=self.model_type,
                ckpt_file=local_model_file,   # Local checkpoint from build
                vocab_file=local_vocab_file,  # Local vocab from build
                ode_method="euler",
                use_ema=True,
                device=self.device
            )
            
            self.is_loaded = True
            load_time = time.time() - start_time
            logger.info(f"✅ F5-TTS model loaded from local files in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load F5-TTS model: {e}")
            self.is_loaded = False
            return False
    
    def generate_expressive(
        self,
        text: str,
        tag_type: str,
        speaker_embedding: Optional[np.ndarray] = None,
        reference_audio: Optional[np.ndarray] = None,
        custom_tag_audio: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Generate expressive speech using F5-TTS
        
        Args:
            text: Text to synthesize
            tag_type: Expressive tag type (whisper, shout, happy, sad, angry, etc.)
            speaker_embedding: Speaker embedding array
            reference_audio: Reference audio for voice cloning
            custom_tag_audio: Dict of custom reference audio for non-native tags
            **kwargs: Additional generation parameters
            
        Returns:
            Generated audio array or None if failed
        """
        if not self.is_loaded:
            if not self.load_model():
                logger.error("Cannot generate - F5-TTS model not loaded")
                return None
        
        try:
            logger.info(f"Generating expressive speech: {tag_type} - '{text[:50]}...'")
            start_time = time.time()
            
            # Determine if this is a native F5-TTS tag or needs mapping
            actual_tag = tag_type
            if tag_type in self.extended_tags:
                actual_tag = self.extended_tags[tag_type]
                logger.debug(f"Mapping extended tag '{tag_type}' to native tag '{actual_tag}'")
            
            # Get reference audio for voice cloning
            ref_audio = reference_audio if reference_audio is not None else speaker_embedding
            
            # For custom tags, use custom reference audio if provided
            if tag_type not in self.native_tags and custom_tag_audio and tag_type in custom_tag_audio:
                logger.info(f"Using custom reference audio for tag: {tag_type}")
                ref_audio = custom_tag_audio[tag_type]
            
            # Prepare generation parameters for F5-TTS
            gen_params = {
                'text': text,
                'ref_audio': ref_audio,
                'ref_text': "",  # Can be empty for voice cloning
                'exp_name': actual_tag,  # Use the expressive tag name
                'remove_silence': True,
                **kwargs
            }
            
            # Generate audio using F5-TTS with expressive tag
            audio_output, sample_rate = self.model.infer(**gen_params)
            
            # Apply additional expressive effects if needed
            if tag_type in self.expressive_effects:
                config = self.expressive_effects[tag_type]
                audio_output = self._apply_expressive_effects(
                    audio_output, 
                    config, 
                    sample_rate
                )
            
            gen_time = time.time() - start_time
            duration = len(audio_output) / sample_rate
            
            logger.info(f"F5-TTS expressive generation complete: {duration:.2f}s audio in {gen_time:.2f}s "
                       f"(RTF: {gen_time/duration:.3f}) using tag '{actual_tag}'")
            
            return audio_output
            
        except Exception as e:
            logger.error(f"F5-TTS expressive generation failed: {e}")
            return None
    
    def _apply_expressive_effects(
        self, 
        audio: np.ndarray, 
        config: Dict[str, float], 
        sample_rate: int
    ) -> np.ndarray:
        """Apply expressive effects to generated audio"""
        try:
            # Apply volume scaling
            audio = audio * config['volume_scale']
            
            # Apply emphasis (simple dynamic range adjustment)
            if config['emphasis'] > 0.5:
                # Increase dynamic range for emphasis
                audio = np.sign(audio) * np.power(np.abs(audio), 1.0 - config['emphasis'] * 0.3)
            else:
                # Reduce dynamic range for subtle effects
                audio = np.sign(audio) * np.power(np.abs(audio), 1.0 + config['emphasis'] * 0.5)
            
            # Ensure audio doesn't clip
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
            
            return audio
            
        except Exception as e:
            logger.warning(f"Failed to apply expressive effects: {e}")
            return audio
    
    def is_available(self) -> bool:
        """Check if F5-TTS is available and loaded"""
        return F5_AVAILABLE and self.is_loaded
    
    def get_supported_tags(self) -> Dict[str, Any]:
        """Get list of supported expressive tags with their capabilities"""
        return {
            'native_f5_tags': list(self.native_tags.keys()),
            'extended_tags': list(self.extended_tags.keys()),
            'all_supported': list(self.native_tags.keys()) + list(self.extended_tags.keys()),
            'custom_audio_support': True,
            'tag_details': {
                **self.native_tags,
                **{tag: {'description': f'Maps to {target}', 'builtin': False} 
                   for tag, target in self.extended_tags.items()}
            }
        }

class F5TTSFallback:
    """Fallback class when F5-TTS is not available"""
    
    def __init__(self, *args, **kwargs):
        self.is_loaded = False
        logger.warning("F5TTSFallback initialized - expressive tags will be processed by Chatterbox")
    
    def load_model(self):
        return False
    
    def generate_expressive(self, *args, **kwargs):
        return None
    
    def is_available(self):
        return False
    
    def get_supported_tags(self):
        return []

def create_f5_tts(device: str = "cuda") -> Union[F5TTSWrapper, F5TTSFallback]:
    """Factory function to create F5-TTS instance"""
    if F5_AVAILABLE:
        return F5TTSWrapper(device=device)
    else:
        return F5TTSFallback()

# Example usage
if __name__ == "__main__":
    # Test F5-TTS wrapper
    f5_tts = create_f5_tts()
    
    if f5_tts.is_available():
        print("F5-TTS available")
        print(f"Supported tags: {f5_tts.get_supported_tags()}")
        
        # Test generation (would need actual reference audio)
        # audio = f5_tts.generate_expressive(
        #     text="This is a test of expressive speech",
        #     tag_type="excited",
        #     reference_audio=None
        # )
    else:
        print("F5-TTS not available - using fallback") 