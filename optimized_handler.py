#!/usr/bin/env python3
"""
Optimized Handler using direct audio arrays instead of temporary files
This eliminates the embedding → WAV → embedding conversion inefficiency
"""

import runpod
import torch
import base64
import io
import logging
import time
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Generator, Tuple, List

# Import optimized components
from optimized_chatterbox import OptimizedChatterboxTTS, create_optimized_tts
from optimized_voice_library import OptimizedVoiceLibrary, initialize_optimized_voice_library

# Import legacy components for fallback
from local_voice_library import LocalVoiceLibrary
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.s3gen import S3GEN_SR, S3Gen

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
optimized_tts_model: Optional[OptimizedChatterboxTTS] = None
optimized_voice_library: Optional[OptimizedVoiceLibrary] = None
legacy_voice_library: Optional[LocalVoiceLibrary] = None  # Fallback
s3gen_model: Optional[S3Gen] = None

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def load_optimized_models():
    """Load optimized TTS model and voice library"""
    global optimized_tts_model, optimized_voice_library, legacy_voice_library, s3gen_model
    
    logger.info(f"Loading optimized models on device: {device}")
    
    try:
        # Load optimized ChatterboxTTS
        logger.info("Loading OptimizedChatterboxTTS...")
        optimized_tts_model = create_optimized_tts(device=device)
        logger.info("OptimizedChatterboxTTS loaded successfully")
        
        # Load optimized voice library
        logger.info("Loading OptimizedVoiceLibrary...")
        optimized_voice_library = initialize_optimized_voice_library()
        if optimized_voice_library.is_available():
            stats = optimized_voice_library.get_stats()
            logger.info(f"OptimizedVoiceLibrary loaded with {stats['total_voices']} voices")
            logger.info(f"Cache settings: {stats['cache_size']}/{stats['max_cache_size']}")
        else:
            logger.warning("Optimized voice library not available, falling back to legacy")
        
        # Load legacy voice library as fallback
        logger.info("Loading legacy voice library as fallback...")
        try:
            from local_voice_library import initialize_local_voice_library
            legacy_voice_library = initialize_local_voice_library()
            logger.info("Legacy voice library loaded as fallback")
        except Exception as e:
            logger.warning(f"Legacy voice library failed to load: {e}")
        
        # Load S3Gen model for voice conversion (unchanged)
        logger.info("Loading S3Gen model for voice conversion...")
        s3gen_model = S3Gen()
        s3g_checkpoint = "checkpoints/s3gen.pt"
        
        Path("checkpoints").mkdir(exist_ok=True)
        map_location = torch.device('cpu') if device in ['cpu', 'mps'] else None
        
        if Path(s3g_checkpoint).exists():
            logger.info(f"Loading S3Gen model from {s3g_checkpoint}")
            s3gen_model.load_state_dict(torch.load(s3g_checkpoint, map_location=map_location))
            s3gen_model.to(device)
            s3gen_model.eval()
            logger.info("S3Gen model loaded successfully")
        else:
            logger.warning("S3Gen model not available - voice conversion disabled")
            s3gen_model = None
        
    except Exception as e:
        logger.error(f"Failed to load optimized models: {e}")
        raise e

def handle_voice_cloning_source_optimized(job_input: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    OPTIMIZED: Handle voice cloning source and return audio array directly
    This eliminates the temp file creation step
    
    Supports:
    - voice_id: Load from local filesystem or Firebase Storage
    - voice_name: Load from local filesystem or Firebase Storage
    - reference_audio: Direct base64 audio
    - voice_metadata: Add new voice to catalog dynamically (for RunPod Serverless)
    """
    try:
        voice_id = job_input.get('voice_id')
        voice_name = job_input.get('voice_name')
        reference_audio_b64 = job_input.get('reference_audio')
        voice_metadata = job_input.get('voice_metadata')  # NEW: For dynamic voice addition
        max_reference_duration_sec = job_input.get('max_reference_duration_sec', 30)
        
        # NEW: Option 0: Add voice metadata dynamically (for RunPod Serverless)
        if voice_metadata and optimized_voice_library:
            logger.info("Adding voice metadata to catalog dynamically")
            if optimized_voice_library.add_voice_to_catalog(voice_metadata):
                voice_id = voice_metadata.get('voice_id')
                logger.info(f"Voice {voice_id} added to catalog, now loading...")
            else:
                logger.warning("Failed to add voice metadata to catalog")
        
        # Option 1: Use voice from optimized embeddings by ID
        if voice_id:
            if optimized_voice_library:
                # Check if voice exists in catalog, if not try to add from metadata
                if not optimized_voice_library.is_available() or voice_id not in optimized_voice_library.voice_catalog:
                    # Try to get metadata from job_input if provided
                    if voice_metadata and voice_metadata.get('voice_id') == voice_id:
                        optimized_voice_library.add_voice_to_catalog(voice_metadata)
                
                audio_array = optimized_voice_library.get_voice_audio_direct(voice_id)
                if audio_array is not None:
                    logger.info(f"Using voice from optimized embeddings: {voice_id} (DIRECT)")
                    return audio_array
                else:
                    raise ValueError(f"Voice ID '{voice_id}' not found. Ensure voice_metadata is provided or voice exists in catalog.")
            else:
                raise ValueError("Optimized voice library not available")
        
        # Option 2: Use voice from optimized library by name
        elif voice_name:
            if optimized_voice_library:
                audio_array = optimized_voice_library.get_voice_audio_by_name_direct(voice_name)
                if audio_array is not None:
                    logger.info(f"Using voice from optimized embeddings by name: {voice_name} (DIRECT)")
                    return audio_array
            
            # Fallback to legacy library
            if legacy_voice_library and legacy_voice_library.is_available():
                audio_array = legacy_voice_library.get_voice_audio_by_name(voice_name)
                if audio_array is not None:
                    logger.info(f"Using voice from legacy library: {voice_name} (fallback)")
                    return audio_array
            
            raise ValueError(f"Voice '{voice_name}' not found in any library")
        
        # Option 3: Use provided reference audio
        elif reference_audio_b64:
            logger.info("Using provided reference audio")
            # Decode reference audio (inline utility)
            def base64_to_audio(audio_b64):
                import base64
                import io
                import soundfile as sf
                audio_data = base64.b64decode(audio_b64)
                audio_buffer = io.BytesIO(audio_data)
                audio_array, sample_rate = sf.read(audio_buffer)
                return audio_array, sample_rate
                
            reference_audio, ref_sr = base64_to_audio(reference_audio_b64)
            
            # Trim to max duration
            max_samples = int(max_reference_duration_sec * ref_sr)
            if len(reference_audio) > max_samples:
                reference_audio = reference_audio[:max_samples]
                logger.info(f"Trimmed reference audio to {max_reference_duration_sec} seconds")
            
            return reference_audio
        
        # No voice cloning source provided
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error handling optimized voice cloning source: {e}")
        raise e

def generate_basic_tts_optimized(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """OPTIMIZED: Generate basic TTS using direct audio arrays"""
    try:
        # Extract and validate text
        text = job_input.get('text', 'Hello, this is a test.')
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(text) > 5000:
            raise ValueError("Text too long (max 5000 characters)")
        
        # Basic parameters
        sample_rate = job_input.get('sample_rate', None)
        audio_normalization = job_input.get('audio_normalization', None)
        
        logger.info(f"Optimized Basic TTS request - Text: {len(text)} chars")
        
        # OPTIMIZATION: Get voice audio array directly (no temp files!)
        audio_prompt_array = handle_voice_cloning_source_optimized(job_input)
        voice_cloning_enabled = audio_prompt_array is not None
        
        # Generate audio using optimized TTS
        logger.info("Starting optimized basic TTS generation...")
        wav = optimized_tts_model.generate(
            text=text,
            audio_prompt_array=audio_prompt_array  # DIRECT ARRAY INPUT!
        )
        logger.info(f"Optimized basic TTS generation complete - shape: {wav.shape}")
        
        # Process audio (inline utility functions)
        def process_audio_tensor(wav_tensor, target_sr, normalize=None):
            if isinstance(wav_tensor, torch.Tensor):
                wav_numpy = wav_tensor.squeeze().cpu().numpy()
            else:
                wav_numpy = wav_tensor
            
            # Resample if needed
            if target_sr != 24000:
                import librosa
                wav_numpy = librosa.resample(wav_numpy, orig_sr=24000, target_sr=target_sr)
            
            # Normalize if requested
            if normalize == "peak":
                max_val = np.max(np.abs(wav_numpy))
                if max_val > 0:
                    wav_numpy = wav_numpy / max_val
            elif normalize == "rms":
                rms = np.sqrt(np.mean(wav_numpy**2))
                if rms > 0:
                    wav_numpy = wav_numpy / rms * 0.1
            
            return wav_numpy
        
        def audio_to_base64(audio_array, sample_rate):
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='WAV')
            buffer.seek(0)
            audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            return audio_b64
        output_sr = sample_rate if sample_rate else getattr(optimized_tts_model, 'sr', 24000)
        wav_numpy = process_audio_tensor(wav, output_sr, audio_normalization)
        
        # Convert to base64
        audio_base64 = audio_to_base64(wav_numpy, output_sr)
        
        # Prepare response
        response = {
            "audio": audio_base64,
            "sample_rate": output_sr,
            "text": text,
            "mode": "basic_optimized",
            "parameters": {
                "voice_cloning": voice_cloning_enabled,
                "voice_id": job_input.get('voice_id', 'none'),
                "voice_name": job_input.get('voice_name', 'none'),
                "optimization": "direct_audio_array"  # Indicate optimization used
            }
        }
        
        logger.info(f"Optimized basic TTS successful - Duration: {len(wav_numpy) / output_sr:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Optimized basic TTS generation failed: {e}")
        raise RuntimeError(f"Optimized basic TTS generation failed: {str(e)}")

def generate_streaming_tts_optimized(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """OPTIMIZED: Generate streaming TTS using direct audio arrays"""
    try:
        # Extract and validate text
        text = job_input.get('text', 'Hello, this is a test.')
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(text) > 5000:
            raise ValueError("Text too long (max 5000 characters)")
        
        # Streaming parameters
        chunk_size = job_input.get('chunk_size', 50)
        exaggeration = job_input.get('exaggeration', 0.5)
        cfg_weight = job_input.get('cfg_weight', job_input.get('cfg', 0.5))
        temperature = job_input.get('temperature', 0.8)
        print_metrics = job_input.get('print_metrics', True)
        
        # Audio output parameters
        sample_rate = job_input.get('sample_rate', None)
        audio_normalization = job_input.get('audio_normalization', None)
        
        logger.info(f"Optimized Streaming TTS request - Text: {len(text)} chars, Chunk size: {chunk_size}")
        
        # OPTIMIZATION: Get voice audio array directly (no temp files!)
        audio_prompt_array = handle_voice_cloning_source_optimized(job_input)
        voice_cloning_enabled = audio_prompt_array is not None
        
        # Generate audio using optimized streaming
        logger.info("Starting optimized streaming TTS generation...")
        audio_chunks = []
        chunk_metrics = []
        
        start_time = time.time()
        first_chunk_time = None
        
        # Use optimized streaming generation
        for audio_chunk, metrics in optimized_tts_model.generate_stream(
            text=text,
            audio_prompt_array=audio_prompt_array,  # DIRECT ARRAY INPUT!
            chunk_size=chunk_size,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            print_metrics=print_metrics
        ):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
            
            audio_chunks.append(audio_chunk)
            
            # Collect metrics
            chunk_info = {
                'chunk_count': metrics.chunk_count,
                'rtf': metrics.rtf if hasattr(metrics, 'rtf') and metrics.rtf else None,
                'chunk_shape': list(audio_chunk.shape) if hasattr(audio_chunk, 'shape') else None
            }
            chunk_metrics.append(chunk_info)
        
        # Combine all chunks
        final_audio = torch.cat(audio_chunks, dim=-1)
        total_time = time.time() - start_time
        
        logger.info(f"Optimized streaming TTS complete - Total chunks: {len(audio_chunks)}, Total time: {total_time:.3f}s")
        
        # Process audio (inline utility functions)
        def process_audio_tensor(wav_tensor, target_sr, normalize=None):
            if isinstance(wav_tensor, torch.Tensor):
                wav_numpy = wav_tensor.squeeze().cpu().numpy()
            else:
                wav_numpy = wav_tensor
            
            # Resample if needed
            if target_sr != 24000:
                import librosa
                wav_numpy = librosa.resample(wav_numpy, orig_sr=24000, target_sr=target_sr)
            
            # Normalize if requested
            if normalize == "peak":
                max_val = np.max(np.abs(wav_numpy))
                if max_val > 0:
                    wav_numpy = wav_numpy / max_val
            elif normalize == "rms":
                rms = np.sqrt(np.mean(wav_numpy**2))
                if rms > 0:
                    wav_numpy = wav_numpy / rms * 0.1
            
            return wav_numpy
        
        def audio_to_base64(audio_array, sample_rate):
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='WAV')
            buffer.seek(0)
            audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            return audio_b64
        
        output_sr = sample_rate if sample_rate else getattr(optimized_tts_model, 'sr', 24000)
        wav_numpy = process_audio_tensor(final_audio, output_sr, audio_normalization)
        
        # Convert to base64
        audio_base64 = audio_to_base64(wav_numpy, output_sr)
        
        # Prepare response
        response = {
            "audio": audio_base64,
            "sample_rate": output_sr,
            "text": text,
            "mode": "streaming_optimized",
            "parameters": {
                "chunk_size": chunk_size,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "voice_cloning": voice_cloning_enabled,
                "optimization": "direct_audio_array"  # Indicate optimization used
            },
            "streaming_metrics": {
                "total_chunks": len(audio_chunks),
                "first_chunk_latency": first_chunk_time,
                "total_generation_time": total_time,
                "audio_duration": len(wav_numpy) / output_sr,
                "rtf": total_time / (len(wav_numpy) / output_sr) if len(wav_numpy) > 0 else None,
                "chunk_details": chunk_metrics
            }
        }
        
        logger.info(f"Optimized streaming TTS successful - Duration: {len(wav_numpy) / output_sr:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Optimized streaming TTS generation failed: {e}")
        raise RuntimeError(f"Optimized streaming TTS generation failed: {str(e)}")

def handler_optimized(job):
    """OPTIMIZED: Main handler function using direct audio arrays"""
    try:
        # Load models if not already loaded
        if optimized_tts_model is None:
            load_optimized_models()
        
        # Get job input
        job_input = job.get('input', {})
        
        # Determine operation type
        operation = job_input.get('operation', 'tts')
        
        if operation == 'tts':
            mode = job_input.get('mode', 'basic')
            
            if mode == 'basic':
                return generate_basic_tts_optimized(job_input)
            elif mode == 'streaming':
                return generate_streaming_tts_optimized(job_input)
            elif mode == 'streaming_voice_cloning':
                # Use streaming with voice cloning enabled
                return generate_streaming_tts_optimized(job_input)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        elif operation == 'list_local_voices':
            # List voices from optimized library
            if optimized_voice_library and optimized_voice_library.is_available():
                local_voices = optimized_voice_library.list_voices()
                stats = optimized_voice_library.get_stats()
                return {
                    "local_voices": local_voices,
                    "total_local_voices": len(local_voices),
                    "stats": stats,
                    "optimization": "direct_audio_arrays"
                }
            else:
                return {
                    "error": "Optimized voice library not available",
                    "message": "Optimized voice library failed to initialize"
                }
        
        # For other operations (voice_conversion, voice_transfer, voice_cloning)
        # fall back to backup handler for now
        else:
            logger.info(f"Falling back to backup handler for operation: {operation}")
            try:
                import sys
                import importlib.util
                spec = importlib.util.spec_from_file_location("handler_backup", "/workspace/handler_backup.py")
                backup_handler = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(backup_handler)
                return backup_handler.handler(job)
            except Exception as e:
                logger.error(f"Fallback handler error: {e}")
                return {"error": f"Operation '{operation}' not supported in optimized handler yet"}
            
    except Exception as e:
        logger.error(f"Optimized handler error: {e}")
        return {"error": str(e)}

# Performance comparison function
def compare_performance():
    """Compare optimized vs legacy performance"""
    logger.info("=== PERFORMANCE COMPARISON ===")
    
    # Test optimized approach
    logger.info("Testing optimized approach...")
    if optimized_voice_library:
        start = time.time()
        audio_array = optimized_voice_library.get_voice_audio_by_name_direct("Amy")
        optimized_time = time.time() - start
        logger.info(f"Optimized voice loading: {optimized_time:.4f}s")
    
    # Test legacy approach
    logger.info("Testing legacy approach...")
    if legacy_voice_library:
        start = time.time()
        audio_array = legacy_voice_library.get_voice_audio_by_name("Amy")
        legacy_time = time.time() - start
        logger.info(f"Legacy voice loading: {legacy_time:.4f}s")
        
        if optimized_time and legacy_time:
            improvement = ((legacy_time - optimized_time) / legacy_time) * 100
            logger.info(f"Performance improvement: {improvement:.1f}%")

# Load models on startup
if __name__ == "__main__":
    logger.info("Starting Optimized Chatterbox TTS serverless handler...")
    
    # Load optimized models
    load_optimized_models()
    
    # Compare performance
    compare_performance()
    
    # Start the optimized handler
    runpod.serverless.start({"handler": handler_optimized}) 