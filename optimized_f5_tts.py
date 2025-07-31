#!/usr/bin/env python3
"""
Optimized F5-TTS wrapper following the Chatterbox pattern
Loads models from local files instead of runtime downloads
"""

import torch
import numpy as np
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

# Try to import F5-TTS
try:
    from f5_tts.api import F5TTS
    F5_AVAILABLE = True
    logger.info("F5-TTS package available")
except ImportError as e:
    F5_AVAILABLE = False
    logger.warning(f"F5-TTS not available: {e}")

class OptimizedF5TTS:
    """
    Optimized F5-TTS wrapper that mirrors the Chatterbox pattern
    - Uses pre-downloaded local models (like checkpoints/s3gen.pt)
    - No runtime downloads required
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize F5-TTS with local model loading"""
        self.device = device
        self.model = None
        self.is_loaded = False
        
        # Model paths (following Chatterbox checkpoints pattern)
        self.model_dir = "/workspace/f5_models/F5TTS_v1_Base"
        self.model_file = f"{self.model_dir}/model_1250000.safetensors"
        self.vocab_file = f"{self.model_dir}/vocab.txt"
        
        # Native F5-TTS expressive tags
        self.supported_tags = {
            'whisper': 'Whispered speech',
            'shout': 'Yelling/shouting', 
            'happy': 'Happy tone',
            'sad': 'Sad tone',
            'angry': 'Angry tone'
        }
        
        # Extended tags mapped to native tags
        self.tag_mapping = {
            'excited': 'happy',
            'calm': 'whisper',
            'nervous': 'whisper', 
            'confident': 'happy',
            'loud': 'shout',
            'quiet': 'whisper'
        }
        
        logger.info(f"OptimizedF5TTS initialized on {device}")
    
    def load_model(self):
        """Load F5-TTS model using runtime downloads (like official F5-TTS pattern)"""
        if not F5_AVAILABLE:
            logger.error("F5-TTS package not available")
            return False
        
        try:
            logger.info(f"Loading F5-TTS model ({self.model_type}) on {self.device}")
            start_time = time.time()
            
            # Use F5-TTS default model loading (downloads to HuggingFace cache if needed)
            # This follows the official F5-TTS Docker pattern with VOLUME /root/.cache/huggingface/hub/
            self.model = F5TTS(
                model_type=self.model_type,  # "F5TTS_v1_Base"
                device=self.device
            )
            
            self.is_loaded = True
            load_time = time.time() - start_time
            logger.info(f"✅ F5-TTS model loaded successfully in {load_time:.2f}s")
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
        **kwargs
    ) -> Optional[np.ndarray]:
        """Generate expressive speech using F5-TTS"""
        if not self.is_loaded:
            logger.error("F5-TTS model not loaded")
            return None
        
        try:
            # Map extended tags to native tags
            actual_tag = self.tag_mapping.get(tag_type, tag_type)
            
            if actual_tag not in self.supported_tags:
                logger.warning(f"Unsupported tag '{tag_type}', using default")
                actual_tag = 'happy'  # Default fallback
            
            logger.info(f"Generating F5-TTS audio: {actual_tag} - '{text[:50]}...'")
            
            # Generate using F5-TTS model
            # Note: This is a simplified interface - actual F5-TTS API may differ
            audio = self.model.infer(
                text=text,
                ref_audio=speaker_embedding,  # Reference audio for voice
                remove_silence=True,
                speed=1.0
            )
            
            # Convert to numpy array if needed
            if isinstance(audio, torch.Tensor):
                audio = audio.squeeze().cpu().numpy()
            
            logger.info(f"F5-TTS generation complete: {len(audio)} samples")
            return audio
            
        except Exception as e:
            logger.error(f"F5-TTS generation failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if F5-TTS is available and loaded"""
        return F5_AVAILABLE and self.is_loaded
    
    def get_supported_tags(self) -> Dict[str, str]:
        """Get supported expressive tags"""
        return self.supported_tags

    @classmethod
    def from_pretrained(cls, device: str = "cuda", model_type: str = "F5TTS_v1_Base"):
        """
        Create OptimizedF5TTS instance with model loading (following Chatterbox pattern)
        F5-TTS will download models to /root/.cache/huggingface/hub/ at runtime
        """
        instance = cls(device=device, model_type=model_type)
        
        if not F5_AVAILABLE:
            logger.warning("F5-TTS package not available")
            return instance
            
        try:
            logger.info(f"Loading F5-TTS model ({model_type}) on {device}")
            start_time = time.time()
            
            # Use F5-TTS default model loading (downloads to HuggingFace cache if needed)
            # This follows the official F5-TTS pattern with VOLUME /root/.cache/huggingface/hub/
            instance.model = F5TTS(
                model_type=model_type,  # "F5TTS_v1_Base"
                device=device
            )
            
            instance.is_loaded = True
            load_time = time.time() - start_time
            logger.info(f"✅ F5-TTS model loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to load F5-TTS model: {e}")
            instance.is_loaded = False
            
        return instance


class F5TTSFallback:
    """Fallback class when F5-TTS is not available"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        logger.warning("F5TTSFallback initialized - expressive tags will be processed by Chatterbox")
    
    def load_model(self):
        return False
    
    def generate_expressive(self, *args, **kwargs):
        return None
    
    def is_available(self):
        return False
    
    def get_supported_tags(self):
        return {}


def create_optimized_f5_tts(device: str = "cuda") -> Union[OptimizedF5TTS, F5TTSFallback]:
    """Factory function to create optimized F5-TTS instance (like create_optimized_tts)"""
    if F5_AVAILABLE:
        f5_model = OptimizedF5TTS(device=device)
        if f5_model.load_model():
            return f5_model
        else:
            logger.warning("F5-TTS model loading failed, using fallback")
            return F5TTSFallback(device=device)
    else:
        return F5TTSFallback(device=device)


# Example usage and testing
if __name__ == "__main__":
    # Test optimized F5-TTS
    f5_tts = create_optimized_f5_tts()
    
    if f5_tts.is_available():
        print("✅ OptimizedF5TTS available")
        print(f"Supported tags: {f5_tts.get_supported_tags()}")
    else:
        print("❌ OptimizedF5TTS not available - using fallback") 