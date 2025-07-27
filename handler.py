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

# Import voice conversion models
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.s3gen import S3GEN_SR, S3Gen

# Import local voice library
from local_voice_library import initialize_local_voice_library, LocalVoiceLibrary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
tts_model: Optional[ChatterboxTTS] = None
s3gen_model: Optional[S3Gen] = None
local_voice_library: Optional[LocalVoiceLibrary] = None
# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def load_models():
    """Load TTS model, voice conversion model, and voice library on startup"""
    global tts_model, s3gen_model, local_voice_library
    
    logger.info(f"Loading models on device: {device}")
    
    try:
        logger.info("Loading ChatterboxTTS model...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("ChatterboxTTS model loaded successfully")
        
        # Load S3Gen model for voice conversion
        logger.info("Loading S3Gen model for voice conversion...")
        s3gen_model = S3Gen()
        s3g_checkpoint = "checkpoints/s3gen.pt"
        
        # Create checkpoints directory if it doesn't exist
        Path("checkpoints").mkdir(exist_ok=True)
        
        # Determine map_location for loading
        map_location = torch.device('cpu') if device in ['cpu', 'mps'] else None
        
        if Path(s3g_checkpoint).exists():
            logger.info(f"Loading S3Gen model from {s3g_checkpoint}")
            s3gen_model.load_state_dict(torch.load(s3g_checkpoint, map_location=map_location))
            s3gen_model.to(device)
            s3gen_model.eval()
            logger.info("S3Gen model loaded successfully")
        else:
            logger.warning(f"S3Gen checkpoint not found at {s3g_checkpoint}")
            
            # Try to download the model from the chatterbox-streaming repository
            try:
                import urllib.request
                logger.info("Attempting to download S3Gen model...")
                
                # This is a placeholder - the actual download URL would need to be determined
                # For now, we'll set s3gen_model to None
                logger.warning("S3Gen model download not implemented yet")
                logger.warning("Voice conversion will not be available")
                s3gen_model = None
                
            except Exception as e:
                logger.error(f"Failed to download S3Gen model: {e}")
                logger.warning("Voice conversion will not be available")
                s3gen_model = None
        
        # Initialize local voice library
        logger.info("Initializing local voice library...")
        local_voice_library = initialize_local_voice_library()
        if local_voice_library.is_available():
            stats = local_voice_library.get_stats()
            logger.info(f"Local voice library loaded with {stats['total_voices']} voices")
        else:
            logger.warning("No local voice library found - voice_id parameters will not work")
            logger.warning("Run generate_local_embeddings.py first to create voice embeddings")
        
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

# Voice Library - Store reference voices on the server
VOICE_LIBRARY_PATH = Path("/tmp/voice_library")
VOICE_LIBRARY_PATH.mkdir(exist_ok=True)

# Built-in voice library (add your voices here)
BUILT_IN_VOICES = {
    "amy": "default female voice",
    "john": "default male voice",
    "sarah": "professional female voice",
    "mike": "professional male voice"
}

def save_voice_to_library(voice_name: str, audio_data: np.ndarray, sample_rate: int) -> bool:
    """Save a voice to the server-side voice library"""
    try:
        voice_path = VOICE_LIBRARY_PATH / f"{voice_name}.wav"
        sf.write(voice_path, audio_data, sample_rate)
        logger.info(f"Voice '{voice_name}' saved to library: {voice_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save voice '{voice_name}': {e}")
        return False

def get_voice_from_library(voice_name: str) -> Optional[str]:
    """Get a voice file path from the library"""
    try:
        voice_path = VOICE_LIBRARY_PATH / f"{voice_name}.wav"
        if voice_path.exists():
            return str(voice_path)
        else:
            logger.warning(f"Voice '{voice_name}' not found in library")
            return None
    except Exception as e:
        logger.error(f"Error accessing voice '{voice_name}': {e}")
        return None

def list_available_voices() -> Dict[str, Any]:
    """List all available voices in the library"""
    try:
        voices = {}
        
        # Add built-in voices
        for voice_name, description in BUILT_IN_VOICES.items():
            voices[voice_name] = {
                "type": "built-in",
                "description": description,
                "available": get_voice_from_library(voice_name) is not None
            }
        
        # Add user-uploaded voices
        for voice_file in VOICE_LIBRARY_PATH.glob("*.wav"):
            voice_name = voice_file.stem
            if voice_name not in BUILT_IN_VOICES:
                voices[voice_name] = {
                    "type": "user-uploaded",
                    "description": f"User uploaded voice: {voice_name}",
                    "available": True
                }
        
        return voices
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        return {}

def upload_voice_to_library(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Upload a voice to the server library"""
    try:
        voice_name = job_input.get('voice_name')
        reference_audio_b64 = job_input.get('reference_audio')
        voice_description = job_input.get('voice_description', f"User uploaded voice: {voice_name}")
        
        if not voice_name:
            raise ValueError("voice_name is required")
        if not reference_audio_b64:
            raise ValueError("reference_audio is required")
        
        # Decode the audio
        reference_audio, ref_sr = base64_to_audio(reference_audio_b64)
        
        # Save to library
        if save_voice_to_library(voice_name, reference_audio, ref_sr):
            return {
                "success": True,
                "message": f"Voice '{voice_name}' uploaded to library",
                "voice_name": voice_name,
                "description": voice_description
            }
        else:
            return {
                "success": False,
                "message": f"Failed to upload voice '{voice_name}'"
            }
            
    except Exception as e:
        logger.error(f"Voice upload failed: {e}")
        return {
            "success": False,
            "message": f"Voice upload failed: {str(e)}"
        }

def handle_voice_cloning_source(job_input: Dict[str, Any]) -> Optional[str]:
    """Handle voice cloning source - by voice_id, voice_name, or reference audio"""
    try:
        voice_id = job_input.get('voice_id')
        voice_name = job_input.get('voice_name')
        reference_audio_b64 = job_input.get('reference_audio')
        max_reference_duration_sec = job_input.get('max_reference_duration_sec', 30)
        
        # Option 1: Use voice from local embeddings by ID
        if voice_id:
            if local_voice_library and local_voice_library.is_available():
                audio_array = local_voice_library.get_voice_audio(voice_id)
                if audio_array is not None:
                    logger.info(f"Using voice from local embeddings: {voice_id}")
                    # Save temporarily for model usage
                    temp_ref_path = tempfile.mktemp(suffix='.wav')
                    sf.write(temp_ref_path, audio_array, 24000)
                    return temp_ref_path
                else:
                    raise ValueError(f"Voice ID '{voice_id}' not found in local embeddings")
            else:
                raise ValueError("Local voice library not available. Run generate_local_embeddings.py first.")
        
        # Option 2: Use voice from library by name
        elif voice_name:
            # First try local voice library
            if local_voice_library and local_voice_library.is_available():
                audio_array = local_voice_library.get_voice_audio_by_name(voice_name)
                if audio_array is not None:
                    logger.info(f"Using voice from local embeddings by name: {voice_name}")
                    # Save temporarily for model usage
                    temp_ref_path = tempfile.mktemp(suffix='.wav')
                    sf.write(temp_ref_path, audio_array, 24000)
                    return temp_ref_path
            
            # Fall back to old library
            voice_path = get_voice_from_library(voice_name)
            if voice_path:
                logger.info(f"Using voice from old library: {voice_name}")
                return voice_path
            else:
                raise ValueError(f"Voice '{voice_name}' not found in any library. Try voice_id or reference_audio instead.")
        
        # Option 3: Use provided reference audio
        elif reference_audio_b64:
            logger.info("Using provided reference audio")
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
            return temp_ref_path
        
        # No voice cloning source provided
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error handling voice cloning source: {e}")
        raise e

def generate_basic_tts(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Generate basic TTS with voice library support"""
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
        
        logger.info(f"Basic TTS request - Text: {len(text)} chars")
        
        # Prepare generation parameters
        generation_params = {'text': text}
        
        # Handle voice cloning (by name or by audio)
        temp_ref_path = handle_voice_cloning_source(job_input)
        voice_cloning_enabled = temp_ref_path is not None
        
        if temp_ref_path:
            generation_params['audio_prompt_path'] = temp_ref_path
        
        # Generate audio
        logger.info("Starting basic TTS generation...")
        wav = tts_model.generate(**generation_params)
        logger.info(f"Basic TTS generation complete - shape: {wav.shape}")
        
        # Clean up temporary file if we created one
        if temp_ref_path and temp_ref_path.startswith('/tmp/'):
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
                "voice_cloning": voice_cloning_enabled,
                "voice_id": job_input.get('voice_id', 'none'),
                "voice_name": job_input.get('voice_name', 'none')
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
        temp_ref_path = handle_voice_cloning_source(job_input)
        voice_cloning_enabled = temp_ref_path is not None
        
        if temp_ref_path:
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
        if temp_ref_path and temp_ref_path.startswith('/tmp/'):
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
                "voice_cloning": voice_cloning_enabled
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
        temp_ref_path = handle_voice_cloning_source(job_input)
        
        if not temp_ref_path:
            raise ValueError("Reference audio is required for streaming voice cloning mode")
        
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

def generate_voice_conversion(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert input audio to match target speaker voice using S3Gen model
    
    Args:
        job_input: Dictionary containing:
            - input_audio: Base64 encoded input audio to convert
            - target_speaker: Base64 encoded target speaker reference OR voice_name from library
            - voice_name: Optional voice name from voice library
            - voice_id: Optional voice ID from voice library
            - no_watermark: Optional boolean to skip watermarking
    
    Returns:
        Dictionary with converted audio in base64 format
    """
    try:
        # Check if S3Gen model is available
        if s3gen_model is None:
            return {
                "error": "Voice conversion not available. S3Gen model not loaded.",
                "message": "The S3Gen model checkpoint (checkpoints/s3gen.pt) is required for voice conversion.",
                "instructions": [
                    "1. Download the S3Gen model from the chatterbox-streaming repository",
                    "2. Place it at: checkpoints/s3gen.pt",
                    "3. Rebuild the RunPod container",
                    "4. Alternative: Use the ChatterboxTTS voice cloning instead with operation: 'tts', mode: 'streaming_voice_cloning'"
                ],
                "alternative_tts": {
                    "operation": "tts",
                    "mode": "streaming_voice_cloning", 
                    "text": "Your text here",
                    "voice_name": "Amy",
                    "chunk_size": 25,
                    "exaggeration": 0.7
                }
            }
        
        # Get input audio
        input_audio_b64 = job_input.get('input_audio')
        if not input_audio_b64:
            return {
                "error": "input_audio is required for voice conversion",
                "message": "Please provide the input audio as base64 encoded data"
            }
        
        # Decode input audio
        try:
            input_audio_data = base64.b64decode(input_audio_b64)
            input_audio_buffer = io.BytesIO(input_audio_data)
            input_audio_array, input_sr = sf.read(input_audio_buffer)
        except Exception as e:
            return {
                "error": f"Failed to decode input audio: {str(e)}",
                "message": "Please ensure input_audio is valid base64 encoded audio data"
            }
        
        # Get target speaker reference
        target_speaker_b64 = job_input.get('target_speaker')
        voice_name = job_input.get('voice_name')
        voice_id = job_input.get('voice_id')
        
        target_audio_array = None
        
        # Try to get target speaker from different sources
        if target_speaker_b64:
            # Use provided target speaker audio
            try:
                target_audio_data = base64.b64decode(target_speaker_b64)
                target_audio_buffer = io.BytesIO(target_audio_data)
                target_audio_array, target_sr = sf.read(target_audio_buffer)
            except Exception as e:
                return {
                    "error": f"Failed to decode target speaker audio: {str(e)}",
                    "message": "Please ensure target_speaker is valid base64 encoded audio data"
                }
        
        elif voice_name or voice_id:
            # Use voice from library
            voice_reference = handle_voice_cloning_source(job_input)
            if voice_reference:
                try:
                    # handle_voice_cloning_source returns a file path, not base64 data
                    target_audio_array, target_sr = sf.read(voice_reference)
                    logger.info(f"Loaded target audio from library: {target_sr}Hz, {len(target_audio_array)} samples")
                except Exception as e:
                    return {
                        "error": f"Failed to load library voice: {str(e)}",
                        "message": "Voice library returned invalid audio file"
                    }
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
        
        # Process audio for S3Gen model - proper PyTorch inference mode handling
        # Convert to tensors
        input_audio_16 = torch.tensor(input_audio_16).float().to(device)[None, :]
        target_audio_24 = torch.tensor(target_audio_24).float().to(device)
        
        # Use proper inference mode context and clone tensors to avoid inference tensor issues
        with torch.inference_mode():
            # Clone tensors to ensure they are not inference tensors
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
        
        # Convert to base64
        converted_audio_b64 = audio_to_base64(converted_wav, S3GEN_SR)
        
        # Calculate duration
        duration = len(converted_wav) / S3GEN_SR
        
        logger.info(f"Voice conversion completed successfully. Duration: {duration:.2f}s")
        
        return {
            "audio": converted_audio_b64,
            "sample_rate": S3GEN_SR,
            "duration": duration,
            "format": "wav",
            "model": "s3gen",
            "operation": "voice_conversion"
        }
        
    except Exception as e:
        logger.error(f"Voice conversion failed: {e}")
        return {
            "error": f"Voice conversion failed: {str(e)}",
            "message": "An unexpected error occurred during voice conversion"
        }


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
    """Main handler function for Runpod with voice library support"""
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
        elif operation == 'upload_voice':
            return upload_voice_to_library(job_input)
        elif operation == 'list_voices':
            return {
                "available_voices": list_available_voices(),
                "total_voices": len(list_available_voices())
            }
        elif operation == 'list_local_voices':
            # List voices from local embeddings
            if local_voice_library and local_voice_library.is_available():
                local_voices = local_voice_library.list_voices()
                stats = local_voice_library.get_stats()
                return {
                    "local_voices": local_voices,
                    "total_local_voices": len(local_voices),
                    "stats": stats
                }
            else:
                return {
                    "error": "Local voice library not available",
                    "message": "Run generate_local_embeddings.py first to create voice embeddings"
                }
        elif operation == 'search_local_voices':
            # Search voices in local embeddings
            if local_voice_library and local_voice_library.is_available():
                query = job_input.get('query', '')
                category = job_input.get('category', '')
                speaker = job_input.get('speaker', '')
                
                results = local_voice_library.search_voices(query, category, speaker)
                return {
                    "search_results": results,
                    "total_results": len(results),
                    "query": query,
                    "category": category,
                    "speaker": speaker
                }
            else:
                return {
                    "error": "Local voice library not available",
                    "message": "Run generate_local_embeddings.py first to create voice embeddings"
                }
        elif operation == 'voice_conversion':
            return generate_voice_conversion(job_input)
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