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
f5_tts_model = None  # F5-TTS for expressive tags

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def load_optimized_models():
    """Load optimized TTS model and voice library"""
    global optimized_tts_model, optimized_voice_library, legacy_voice_library, s3gen_model, f5_tts_model
    
    logger.info(f"Loading optimized models on device: {device}")
    
    try:
        # Load optimized ChatterboxTTS
        logger.info("Loading OptimizedChatterboxTTS...")
        optimized_tts_model = create_optimized_tts(device=device)
        logger.info("OptimizedChatterboxTTS loaded successfully")
        
        # Load F5-TTS for expressive tags
        logger.info("Loading F5-TTS...")
        try:
            from f5_tts_integration import F5TTSWrapper
            f5_tts_model = F5TTSWrapper(device=device)
            if f5_tts_model.load_model():
                logger.info("F5-TTS loaded successfully")
            else:
                logger.error("❌ F5-TTS failed to load - check f5_tts_integration.py logs for details")
                logger.error("   F5-TTS will be unavailable for this session")
                f5_tts_model = None
        except Exception as e:
            logger.error(f"❌ F5-TTS initialization failed: {e}")
            logger.error("   F5-TTS will be unavailable for this session")
            f5_tts_model = None
        
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
        
        logger.info(f"Optimized voice library initialized with {len(optimized_voice_library.voice_catalog) if optimized_voice_library else 0} voices")
        
    except Exception as e:
        logger.error(f"Failed to load optimized models: {e}")
        raise e

def handle_voice_cloning_source_optimized(job_input: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    OPTIMIZED: Handle voice cloning source and return audio array directly
    This eliminates the temp file creation step
    """
    try:
        voice_id = job_input.get('voice_id')
        voice_name = job_input.get('voice_name')
        reference_audio_b64 = job_input.get('reference_audio')
        max_reference_duration_sec = job_input.get('max_reference_duration_sec', 30)
        
        # Option 1: Use voice from optimized embeddings by ID
        if voice_id:
            if optimized_voice_library and optimized_voice_library.is_available():
                audio_array = optimized_voice_library.get_voice_audio_direct(voice_id)
                if audio_array is not None:
                    logger.info(f"Using voice from optimized embeddings: {voice_id} (DIRECT)")
                    return audio_array
                else:
                    raise ValueError(f"Voice ID '{voice_id}' not found in optimized embeddings")
            else:
                raise ValueError("Optimized voice library not available")
        
        # Option 2: Use voice from optimized library by name
        elif voice_name:
            if optimized_voice_library and optimized_voice_library.is_available():
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

def generate_voice_conversion_optimized(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """OPTIMIZED: Generate voice conversion using direct audio arrays"""
    try:
        # Extract input audio
        input_audio_b64 = job_input.get('input_audio')
        if not input_audio_b64:
            raise ValueError("input_audio is required")
        
        # Decode input audio (inline utility)
        def base64_to_audio(audio_b64):
            import base64
            import io
            import soundfile as sf
            audio_data = base64.b64decode(audio_b64)
            audio_buffer = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_buffer)
            return audio_array, sample_rate
        
        input_audio_array, input_sr = base64_to_audio(input_audio_b64)
        logger.info(f"Loaded input audio: {input_sr}Hz, {len(input_audio_array)} samples")
        
        # Get target voice parameters
        voice_name = job_input.get('voice_name')
        voice_id = job_input.get('voice_id')
        target_speaker_b64 = job_input.get('target_speaker')
        
        # Handle target audio source
        if target_speaker_b64:
            # Direct audio input (no optimization needed)
            try:
                target_audio_array, target_sr = base64_to_audio(target_speaker_b64)
                logger.info(f"Using provided target speaker audio: {target_sr}Hz, {len(target_audio_array)} samples")
            except Exception as e:
                return {
                    "error": f"Failed to decode target speaker audio: {str(e)}",
                    "message": "Please ensure target_speaker is valid base64 encoded audio data"
                }
        
        elif voice_name or voice_id:
            # OPTIMIZATION: Use voice from library with direct audio arrays
            audio_prompt_array = handle_voice_cloning_source_optimized(job_input)
            if audio_prompt_array is not None:
                target_audio_array = audio_prompt_array
                target_sr = 24000  # Default sample rate for embeddings
                logger.info(f"OPTIMIZED: Loaded target audio from library directly: {len(target_audio_array)} samples")
            else:
                return {
                    "error": "Voice not found in library",
                    "message": f"Voice '{voice_name or voice_id}' not found. Use list_local_voices operation to see available voices."
                }
        
        else:
            return {
                "error": "target_speaker, voice_name, or voice_id is required",
                "message": "Please provide either target_speaker (base64 audio), voice_name, or voice_id"
            }
        
        # Process audio for S3Gen model
        logger.info("Processing audio for voice conversion...")
        
        # Load and resample input audio to S3_SR (16kHz)
        input_audio_16 = librosa.resample(input_audio_array, orig_sr=input_sr, target_sr=S3_SR)
        
        # Load and resample target audio to S3GEN_SR (24kHz), limit to 10 seconds
        target_audio_24 = librosa.resample(target_audio_array, orig_sr=target_sr, target_sr=S3GEN_SR)
        max_samples = S3GEN_SR * 10  # 10 seconds
        if len(target_audio_24) > max_samples:
            target_audio_24 = target_audio_24[:max_samples]
        
        # Process audio for S3Gen model
        input_audio_16 = torch.tensor(input_audio_16).float().to(device)[None, :]
        target_audio_24 = torch.tensor(target_audio_24).float().to(device)
        
        # Use proper inference mode context
        with torch.inference_mode():
            logger.info("Tokenizing input audio...")
            s3_tokens, _ = s3gen_model.tokenizer(input_audio_16.clone())
            
            # Generate voice conversion
            logger.info("Generating voice conversion...")
            converted_wav = s3gen_model(s3_tokens.clone(), target_audio_24.clone(), S3GEN_SR)
            converted_wav = converted_wav.detach().cpu().numpy().flatten()
        
        # Apply watermark if requested
        no_watermark = job_input.get('no_watermark', False)
        if not no_watermark:
            try:
                import perth
                watermarker = perth.PerthImplicitWatermarker()
                converted_wav = watermarker.apply_watermark(converted_wav, sample_rate=S3GEN_SR)
                logger.info("Watermark applied to converted audio")
            except ImportError:
                logger.warning("Perth watermarker not available, skipping watermark")
            except Exception as e:
                logger.warning(f"Failed to apply watermark: {e}")
        
        # Convert to base64 (inline utility)
        def audio_to_base64(audio_array, sample_rate):
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='WAV')
            buffer.seek(0)
            audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            return audio_b64
        
        converted_audio_b64 = audio_to_base64(converted_wav, S3GEN_SR)
        
        # Calculate duration
        duration = len(converted_wav) / S3GEN_SR
        
        logger.info(f"Voice conversion completed successfully. Duration: {duration:.2f}s")
        
        # Prepare response with optimization indicators
        response = {
            "audio": converted_audio_b64,
            "sample_rate": S3GEN_SR,
            "duration": duration,
            "format": "wav",
            "model": "s3gen",
            "operation": "voice_conversion"
        }
        
        # Add optimization indicator if voice library was used
        if voice_name or voice_id:
            response.update({
                "optimization": "direct_audio_array",
                "target_voice_info": {
                    "voice_name": voice_name or "unknown",
                    "loaded_via": "optimized_embeddings"
                }
            })
        
        return response
        
    except Exception as e:
        logger.error(f"Voice conversion failed: {e}")
        return {
            "error": f"Voice conversion failed: {str(e)}",
            "message": "An unexpected error occurred during voice conversion"
        }

def generate_voice_transfer_optimized(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """ENHANCED: Generate voice transfer supporting both base64 and file URLs"""
    try:
        # Import enhanced voice transfer functionality
        from enhanced_voice_transfer import generate_voice_transfer_enhanced
        
        # Use the enhanced version that supports URLs and file paths
        return generate_voice_transfer_enhanced(job_input)
        
        # Process audio for S3Gen model (same logic for both modes)
        logger.info("Processing audio for voice transfer...")
        
        # Resample input audio to S3_SR (16kHz)
        input_audio_16 = librosa.resample(input_audio_array, orig_sr=input_sr, target_sr=S3_SR)
        
        # Resample target audio to S3GEN_SR (24kHz), limit to 10 seconds
        target_audio_24 = librosa.resample(target_audio_array, orig_sr=target_sr, target_sr=S3GEN_SR)
        max_samples = S3GEN_SR * 10  # 10 seconds
        if len(target_audio_24) > max_samples:
            target_audio_24 = target_audio_24[:max_samples]
            logger.info("Target audio trimmed to 10 seconds for processing efficiency")
        
        # Convert to tensors
        input_audio_16 = torch.tensor(input_audio_16).float().to(device)[None, :]
        target_audio_24 = torch.tensor(target_audio_24).float().to(device)
        
        # Generate voice transfer using S3Gen
        with torch.inference_mode():
            logger.info("Tokenizing input audio...")
            s3_tokens, _ = s3gen_model.tokenizer(input_audio_16.clone())
            
            logger.info("Generating voice transfer...")
            transferred_wav = s3gen_model(s3_tokens.clone(), target_audio_24.clone(), S3GEN_SR)
            transferred_wav = transferred_wav.detach().cpu().numpy().flatten()
        
        # Apply watermark if requested
        no_watermark = job_input.get('no_watermark', False)
        if not no_watermark:
            try:
                import perth
                watermarker = perth.PerthImplicitWatermarker()
                transferred_wav = watermarker.apply_watermark(transferred_wav, sample_rate=S3GEN_SR)
                logger.info("Watermark applied to transferred audio")
            except ImportError:
                logger.warning("Perth watermarker not available, skipping watermark")
            except Exception as e:
                logger.warning(f"Failed to apply watermark: {e}")
        
        # Convert to base64 (inline utility)
        def audio_to_base64(audio_array, sample_rate):
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='WAV')
            buffer.seek(0)
            audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            return audio_b64
        
        transferred_audio_b64 = audio_to_base64(transferred_wav, S3GEN_SR)
        
        # Calculate durations
        input_duration = len(input_audio_array) / input_sr
        output_duration = len(transferred_wav) / S3GEN_SR
        
        logger.info(f"Voice transfer completed successfully. Duration: {output_duration:.2f}s")
        
        # Prepare response
        response = {
            "audio": transferred_audio_b64,
            "sample_rate": S3GEN_SR,
            "duration": output_duration,
            "format": "wav",
            "model": "s3gen",
            "operation": "voice_transfer",
            "transfer_info": transfer_info,
            "input_duration": input_duration,
            "processing_time": "30-90 seconds typical"
        }
        
        # Add optimization indicator for embedding mode
        if optimization_indicator:
            response["optimization"] = optimization_indicator
        
        return response
        
    except Exception as e:
        logger.error(f"Voice transfer failed: {e}")
        return {
            "error": f"Voice transfer failed: {str(e)}",
            "message": "An unexpected error occurred during voice transfer"
        }

def generate_f5_tts_optimized(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """OPTIMIZED: Generate F5-TTS using the loaded F5TTSWrapper"""
    try:
        # Extract and validate text
        text = job_input.get('text', 'Hello, this is a test.')
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(text) > 5000:
            raise ValueError("Text too long (max 5000 characters)")
        
        logger.info(f"F5-TTS request - Text: {len(text)} chars")
        
        # Get reference audio if provided
        reference_audio = job_input.get('reference_audio')
        voice_id = job_input.get('voice_id')
        
        # Get speaker embedding
        speaker_embedding = None
        if voice_id:
            try:
                speaker_embedding = get_speaker_embedding(voice_id)
                logger.info(f"Using voice embedding for: {voice_id}")
            except Exception as e:
                logger.warning(f"Failed to get speaker embedding for {voice_id}: {e}")
        
        # Generate audio using F5-TTS
        logger.info("Starting F5-TTS generation...")
        start_time = time.time()
        
        # Use F5TTSWrapper to generate audio
        audio_array = f5_tts_model.generate_expressive(
            text=text,
            tag_type="normal",  # Default tag
            speaker_embedding=speaker_embedding,
            reference_audio=reference_audio
        )
        
        if audio_array is None:
            raise RuntimeError("F5-TTS returned no audio")
        
        total_time = time.time() - start_time
        logger.info(f"F5-TTS generation complete - Duration: {total_time:.2f}s")
        
        # Convert to base64
        sample_rate = 24000  # F5-TTS default
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Prepare response
        response = {
            "audio": audio_base64,
            "sample_rate": sample_rate,
            "text": text,
            "mode": "f5_tts",
            "generation_time": total_time,
            "audio_duration": len(audio_array) / sample_rate
        }
        
        logger.info(f"F5-TTS successful - Audio: {len(audio_array):,} samples")
        return response
        
    except Exception as e:
        logger.error(f"F5-TTS generation failed: {e}")
        raise RuntimeError(f"F5-TTS generation failed: {str(e)}")

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
            # Check for explicit model selection
            model = job_input.get('model', 'chatterbox')
            
            # Check if F5-TTS is explicitly requested
            if model == 'f5_tts' and f5_tts_model is not None:
                # Use F5-TTS for this request - no fallback, show real errors
                return generate_f5_tts_optimized(job_input)
            elif model == 'f5_tts' and f5_tts_model is None:
                # F5-TTS requested but not available
                return {"error": "F5-TTS model not available - check startup logs for loading errors"}

            # Check if text contains expressive tags
            text = job_input.get('text', '')
            if '{' in text and '}' in text:
                # Use enhanced expressive TTS
                try:
                    from enhanced_handler import EnhancedTTSHandler
                    enhanced_handler = EnhancedTTSHandler()
                    return enhanced_handler.generate_expressive_tts(job_input)
                except Exception as e:
                    logger.warning(f"Enhanced expressive TTS failed, falling back to standard: {e}")
                    # Fall through to standard TTS processing
            
            # Standard TTS processing
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
        
        elif operation == 'expressive_tts':
            # Explicit expressive TTS operation
            try:
                from enhanced_handler import EnhancedTTSHandler
                enhanced_handler = EnhancedTTSHandler()
                return enhanced_handler.generate_expressive_tts(job_input)
            except Exception as e:
                logger.error(f"Enhanced expressive TTS failed: {e}")
                return {"error": f"Expressive TTS failed: {str(e)}"}
        
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
        
        elif operation == 'voice_conversion':
            # OPTIMIZED: Voice conversion with direct audio arrays
            return generate_voice_conversion_optimized(job_input)
        
        elif operation == 'voice_transfer':
            # OPTIMIZED: Voice transfer with direct audio arrays  
            return generate_voice_transfer_optimized(job_input)
        
        # For voice_cloning operation, fall back to backup handler (no optimization needed - uses direct user audio)
        elif operation == 'voice_cloning':
            logger.info(f"Falling back to backup handler for voice_cloning (no optimization needed)")
            try:
                import sys
                import importlib.util
                spec = importlib.util.spec_from_file_location("handler_backup", "/workspace/handler_backup.py")
                backup_handler = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(backup_handler)
                return backup_handler.handler(job)
            except Exception as e:
                logger.error(f"Fallback handler error: {e}")
                return {"error": f"Voice cloning operation failed: {str(e)}"}
        
        else:
            return {"error": f"Unknown operation: {operation}"}
            
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