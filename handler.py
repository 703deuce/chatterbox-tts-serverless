import runpod
import torch
import tempfile
import base64
import io
import logging
import time
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Generator, Tuple, List
from chatterbox.tts import ChatterboxTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
tts_model: Optional[ChatterboxTTS] = None
# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def load_models():
    """Load TTS model on startup"""
    global tts_model
    
    logger.info(f"Loading models on device: {device}")
    
    try:
        logger.info("Loading ChatterboxTTS model...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("ChatterboxTTS model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise e

def audio_to_base64(audio_data, sample_rate=24000):
    """Convert audio numpy array to base64 string"""
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def base64_to_audio(base64_string):
    """Convert base64 string to audio numpy array"""
    audio_bytes = base64.b64decode(base64_string)
    audio_buffer = io.BytesIO(audio_bytes)
    audio_data, sample_rate = sf.read(audio_buffer)
    return audio_data, sample_rate

def process_audio_tensor(wav_tensor, sample_rate_target=None, audio_normalization=None):
    """Process audio tensor to numpy array with optional resampling and normalization"""
    # Convert tensor to numpy for processing
    if torch.is_tensor(wav_tensor):
        wav_numpy = wav_tensor.detach().cpu().numpy()
    else:
        wav_numpy = wav_tensor
    
    # Ensure audio is in correct shape (flatten if needed)
    if wav_numpy.ndim > 1:
        wav_numpy = wav_numpy.flatten()
    
    # Get model sample rate
    model_sr = getattr(tts_model, 'sr', 24000)
    
    # Resample if different sample rate requested
    if sample_rate_target and sample_rate_target != model_sr:
        wav_numpy = librosa.resample(wav_numpy, orig_sr=model_sr, target_sr=sample_rate_target)
    
    # Apply audio normalization if specified
    if audio_normalization == 'peak':
        wav_numpy = wav_numpy / np.max(np.abs(wav_numpy))
    elif audio_normalization == 'rms':
        rms = np.sqrt(np.mean(wav_numpy**2))
        wav_numpy = wav_numpy / rms * 0.1
    
    return wav_numpy

def generate_basic_tts(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Generate basic TTS as shown in GitHub documentation"""
    try:
        # Extract and validate text
        text = job_input.get('text', 'Hello, this is a test.')
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(text) > 5000:
            raise ValueError("Text too long (max 5000 characters)")
        
        # Basic parameters (minimal as per GitHub docs)
        sample_rate = job_input.get('sample_rate', None)
        audio_normalization = job_input.get('audio_normalization', None)
        
        # Voice cloning support
        reference_audio_b64 = job_input.get('reference_audio', None)
        max_reference_duration_sec = job_input.get('max_reference_duration_sec', 30)
        
        logger.info(f"Basic TTS request - Text: {len(text)} chars")
        
        # Prepare generation parameters (minimal as per GitHub docs)
        generation_params = {'text': text}
        
        # Handle voice cloning
        temp_ref_path = None
        if reference_audio_b64:
            # Decode reference audio
            reference_audio, ref_sr = base64_to_audio(reference_audio_b64)
            
            # Trim to max duration
            max_samples = int(max_reference_duration_sec * ref_sr)
            if len(reference_audio) > max_samples:
                reference_audio = reference_audio[:max_samples]
                logger.info(f"Trimmed reference audio to {max_reference_duration_sec} seconds")
            
            # Save temporarily for model usage
            temp_ref_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_ref_path, reference_audio, ref_sr)
            generation_params['audio_prompt_path'] = temp_ref_path
        
        # Generate audio using basic method
        logger.info("Starting basic TTS generation...")
        wav = tts_model.generate(**generation_params)
        logger.info(f"Basic TTS generation complete - shape: {wav.shape}")
        
        # Clean up temporary file
        if temp_ref_path:
            try:
                Path(temp_ref_path).unlink()
            except:
                pass
        
        # Process audio
        output_sr = sample_rate if sample_rate else getattr(tts_model, 'sr', 24000)
        wav_numpy = process_audio_tensor(wav, output_sr, audio_normalization)
        
        # Convert to base64
        audio_base64 = audio_to_base64(wav_numpy, output_sr)
        
        # Prepare response
        response = {
            "audio": audio_base64,
            "sample_rate": output_sr,
            "text": text,
            "mode": "basic",
            "parameters": {
                "voice_cloning": reference_audio_b64 is not None
            }
        }
        
        logger.info(f"Basic TTS generation successful - Duration: {len(wav_numpy) / output_sr:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Basic TTS generation failed: {e}")
        raise RuntimeError(f"Basic TTS generation failed: {str(e)}")

def generate_streaming_tts(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Generate streaming TTS as shown in GitHub documentation"""
    try:
        # Extract and validate text
        text = job_input.get('text', 'Hello, this is a test.')
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(text) > 5000:
            raise ValueError("Text too long (max 5000 characters)")
        
        # Streaming parameters
        chunk_size = job_input.get('chunk_size', 50)  # Default from GitHub docs
        exaggeration = job_input.get('exaggeration', 0.5)
        cfg_weight = job_input.get('cfg_weight', job_input.get('cfg', 0.5))
        temperature = job_input.get('temperature', 0.8)
        print_metrics = job_input.get('print_metrics', True)
        
        # Audio output parameters
        sample_rate = job_input.get('sample_rate', None)
        audio_normalization = job_input.get('audio_normalization', None)
        
        # Voice cloning support
        reference_audio_b64 = job_input.get('reference_audio', None)
        max_reference_duration_sec = job_input.get('max_reference_duration_sec', 30)
        
        # Validate parameters
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("Chunk size must be a positive integer")
        if not 0.0 <= exaggeration <= 1.0:
            raise ValueError("Exaggeration must be between 0.0 and 1.0")
        if not 0.0 <= cfg_weight <= 1.0:
            raise ValueError("CFG weight must be between 0.0 and 1.0")
        if not 0.1 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.1 and 2.0")
        
        logger.info(f"Streaming TTS request - Text: {len(text)} chars, Chunk size: {chunk_size}")
        
        # Prepare generation parameters
        generation_params = {
            'text': text,
            'chunk_size': chunk_size,
            'exaggeration': exaggeration,
            'cfg_weight': cfg_weight,
            'temperature': temperature,
            'print_metrics': print_metrics
        }
        
        # Handle voice cloning
        temp_ref_path = None
        if reference_audio_b64:
            # Decode reference audio
            reference_audio, ref_sr = base64_to_audio(reference_audio_b64)
            
            # Trim to max duration
            max_samples = int(max_reference_duration_sec * ref_sr)
            if len(reference_audio) > max_samples:
                reference_audio = reference_audio[:max_samples]
                logger.info(f"Trimmed reference audio to {max_reference_duration_sec} seconds")
            
            # Save temporarily for model usage
            temp_ref_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_ref_path, reference_audio, ref_sr)
            generation_params['audio_prompt_path'] = temp_ref_path
        
        # Generate audio using streaming method
        logger.info("Starting streaming TTS generation...")
        audio_chunks = []
        chunk_metrics = []
        
        start_time = time.time()
        first_chunk_time = None
        
        for audio_chunk, metrics in tts_model.generate_stream(**generation_params):
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
            
            logger.info(f"Generated chunk {metrics.chunk_count}, RTF: {metrics.rtf:.3f}" if hasattr(metrics, 'rtf') and metrics.rtf else f"Chunk {metrics.chunk_count}")
        
        # Combine all chunks into final audio
        final_audio = torch.cat(audio_chunks, dim=-1)
        total_time = time.time() - start_time
        
        logger.info(f"Streaming TTS generation complete - Total chunks: {len(audio_chunks)}, Total time: {total_time:.3f}s")
        
        # Clean up temporary file
        if temp_ref_path:
            try:
                Path(temp_ref_path).unlink()
            except:
                pass
        
        # Process audio
        output_sr = sample_rate if sample_rate else getattr(tts_model, 'sr', 24000)
        wav_numpy = process_audio_tensor(final_audio, output_sr, audio_normalization)
        
        # Convert to base64
        audio_base64 = audio_to_base64(wav_numpy, output_sr)
        
        # Prepare response
        response = {
            "audio": audio_base64,
            "sample_rate": output_sr,
            "text": text,
            "mode": "streaming",
            "parameters": {
                "chunk_size": chunk_size,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "voice_cloning": reference_audio_b64 is not None
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
        
        logger.info(f"Streaming TTS generation successful - Duration: {len(wav_numpy) / output_sr:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Streaming TTS generation failed: {e}")
        raise RuntimeError(f"Streaming TTS generation failed: {str(e)}")

def generate_streaming_voice_cloning_tts(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Generate streaming TTS with voice cloning as shown in GitHub documentation"""
    try:
        # Extract and validate text
        text = job_input.get('text', 'Hello, this is a test.')
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(text) > 5000:
            raise ValueError("Text too long (max 5000 characters)")
        
        # Streaming parameters (with GitHub defaults)
        chunk_size = job_input.get('chunk_size', 25)  # Smaller chunks for lower latency as per docs
        exaggeration = job_input.get('exaggeration', 0.7)  # Default from GitHub example
        cfg_weight = job_input.get('cfg_weight', job_input.get('cfg', 0.3))  # Default from GitHub example
        temperature = job_input.get('temperature', 0.8)
        print_metrics = job_input.get('print_metrics', True)
        
        # Audio output parameters
        sample_rate = job_input.get('sample_rate', None)
        audio_normalization = job_input.get('audio_normalization', None)
        
        # Voice cloning - required for this mode
        reference_audio_b64 = job_input.get('reference_audio', None)
        max_reference_duration_sec = job_input.get('max_reference_duration_sec', 30)
        
        if not reference_audio_b64:
            raise ValueError("Reference audio is required for streaming voice cloning mode")
        
        # Validate parameters
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("Chunk size must be a positive integer")
        if not 0.0 <= exaggeration <= 1.0:
            raise ValueError("Exaggeration must be between 0.0 and 1.0")
        if not 0.0 <= cfg_weight <= 1.0:
            raise ValueError("CFG weight must be between 0.0 and 1.0")
        if not 0.1 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.1 and 2.0")
        
        logger.info(f"Streaming Voice Cloning TTS request - Text: {len(text)} chars, Chunk size: {chunk_size}")
        
        # Handle voice cloning - required for this mode
        # Decode reference audio
        reference_audio, ref_sr = base64_to_audio(reference_audio_b64)
        
        # Trim to max duration
        max_samples = int(max_reference_duration_sec * ref_sr)
        if len(reference_audio) > max_samples:
            reference_audio = reference_audio[:max_samples]
            logger.info(f"Trimmed reference audio to {max_reference_duration_sec} seconds")
        
        # Save temporarily for model usage
        temp_ref_path = tempfile.mktemp(suffix='.wav')
        sf.write(temp_ref_path, reference_audio, ref_sr)
        
        # Prepare generation parameters (as per GitHub docs)
        generation_params = {
            'text': text,
            'audio_prompt_path': temp_ref_path,
            'exaggeration': exaggeration,
            'cfg_weight': cfg_weight,
            'chunk_size': chunk_size,
            'print_metrics': print_metrics
        }
        
        # Generate audio using streaming method with voice cloning
        logger.info("Starting streaming TTS generation with voice cloning...")
        audio_chunks = []
        chunk_metrics = []
        
        start_time = time.time()
        first_chunk_time = None
        
        for audio_chunk, metrics in tts_model.generate_stream(**generation_params):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
            
            audio_chunks.append(audio_chunk)
            
            # Collect metrics
            chunk_info = {
                'chunk_count': metrics.chunk_count,
                'rtf': metrics.rtf if hasattr(metrics, 'rtf') and metrics.rtf else None,
                'chunk_shape': list(audio_chunk.shape) if hasattr(audio_chunk, 'shape') else None,
                'latency_to_first_chunk': metrics.latency_to_first_chunk if hasattr(metrics, 'latency_to_first_chunk') else None
            }
            chunk_metrics.append(chunk_info)
            
            # Log first chunk latency if available
            if hasattr(metrics, 'latency_to_first_chunk') and metrics.latency_to_first_chunk:
                logger.info(f"First chunk latency: {metrics.latency_to_first_chunk:.3f}s")
            
            logger.info(f"Generated chunk {metrics.chunk_count}, RTF: {metrics.rtf:.3f}" if hasattr(metrics, 'rtf') and metrics.rtf else f"Chunk {metrics.chunk_count}")
        
        # Combine all chunks into final audio
        final_audio = torch.cat(audio_chunks, dim=-1)
        total_time = time.time() - start_time
        
        logger.info(f"Streaming voice cloning TTS generation complete - Total chunks: {len(audio_chunks)}, Total time: {total_time:.3f}s")
        
        # Clean up temporary file
        try:
            Path(temp_ref_path).unlink()
        except:
            pass
        
        # Process audio
        output_sr = sample_rate if sample_rate else getattr(tts_model, 'sr', 24000)
        wav_numpy = process_audio_tensor(final_audio, output_sr, audio_normalization)
        
        # Convert to base64
        audio_base64 = audio_to_base64(wav_numpy, output_sr)
        
        # Prepare response
        response = {
            "audio": audio_base64,
            "sample_rate": output_sr,
            "text": text,
            "mode": "streaming_voice_cloning",
            "parameters": {
                "chunk_size": chunk_size,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "voice_cloning": True
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
        
        logger.info(f"Streaming voice cloning TTS generation successful - Duration: {len(wav_numpy) / output_sr:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Streaming voice cloning TTS generation failed: {e}")
        raise RuntimeError(f"Streaming voice cloning TTS generation failed: {str(e)}")



def generate_tts(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Route to appropriate TTS generation method based on mode"""
    mode = job_input.get('mode', 'basic')
    
    if mode == 'basic':
        return generate_basic_tts(job_input)
    elif mode == 'streaming':
        return generate_streaming_tts(job_input)
    elif mode == 'streaming_voice_cloning':
        return generate_streaming_voice_cloning_tts(job_input)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'basic', 'streaming', or 'streaming_voice_cloning'")

def handler(job):
    """Main handler function for Runpod"""
    try:
        # Load models if not already loaded
        if tts_model is None:
            load_models()
        
        # Get job input
        job_input = job.get('input', {})
        
        # Determine operation type
        operation = job_input.get('operation', 'tts')
        
        if operation == 'tts':
            return generate_tts(job_input)
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": str(e)}

# Load models on startup
if __name__ == "__main__":
    logger.info("Starting Chatterbox TTS serverless handler...")
    
    # Load models
    load_models()
    
    # Start the handler
    runpod.serverless.start({"handler": handler}) 