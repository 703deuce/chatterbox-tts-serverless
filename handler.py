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
from typing import Optional, Dict, Any, Generator, Tuple
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

def generate_tts(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Generate text-to-speech audio with basic parameters"""
    try:
        # Extract and validate text
        text = job_input.get('text', 'Hello, this is a test.')
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(text) > 5000:
            raise ValueError("Text too long (max 5000 characters)")
        
        # Basic synthesis parameters (using correct parameter names)
        exaggeration = job_input.get('exaggeration', 0.5)
        cfg_weight = job_input.get('cfg_weight', job_input.get('cfg', 0.5))  # Support both names
        temperature = job_input.get('temperature', 0.8)
        seed = job_input.get('seed', job_input.get('random_seed', None))
        
        # Voice parameters
        voice_mode = job_input.get('voice_mode', 'predefined')
        reference_audio_b64 = job_input.get('reference_audio', None)
        max_reference_duration_sec = job_input.get('max_reference_duration_sec', 30)
        
        # Audio output parameters
        output_format = job_input.get('output_format', 'wav')
        sample_rate = job_input.get('sample_rate', None)
        audio_normalization = job_input.get('audio_normalization', None)
        
        # Validate core parameters
        if not 0.0 <= exaggeration <= 1.0:
            raise ValueError("Exaggeration must be between 0.0 and 1.0")
        if not 0.0 <= cfg_weight <= 1.0:
            raise ValueError("CFG weight must be between 0.0 and 1.0")
        if not 0.1 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.1 and 2.0")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("Seed must be an integer")
        
        # Validate voice parameters
        if voice_mode not in ['predefined', 'clone']:
            raise ValueError("Voice mode must be 'predefined' or 'clone'")
        if voice_mode == 'clone' and not reference_audio_b64:
            raise ValueError("Reference audio required for voice cloning")
        if not 1 <= max_reference_duration_sec <= 60:
            raise ValueError("Max reference duration must be between 1 and 60 seconds")
        
        # Validate audio parameters
        if output_format not in ['wav', 'opus', 'mp3']:
            raise ValueError("Output format must be 'wav', 'opus', or 'mp3'")
        if sample_rate is not None and sample_rate not in [16000, 22050, 24000, 44100, 48000]:
            raise ValueError("Sample rate must be 16000, 22050, 24000, 44100, or 48000")
        
        # Validate streaming parameters
        chunk_size = job_input.get('chunk_size', 25)
        context_window = job_input.get('context_window', 50)
        fade_duration = job_input.get('fade_duration', 0.02)
        
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("Chunk size must be a positive integer")
        if not isinstance(context_window, int) or context_window < 1:
            raise ValueError("Context window must be a positive integer")
        if not isinstance(fade_duration, (int, float)) or fade_duration < 0:
            raise ValueError("Fade duration must be a non-negative number")
        
        logger.info(f"TTS request - Text: {len(text)} chars, Mode: {voice_mode}, Exaggeration: {exaggeration}")
        
        # Prepare generation parameters (using correct streaming API parameter names)
        generation_params = {
            'text': text,
            'exaggeration': exaggeration,
            'cfg_weight': cfg_weight,  # Streaming API uses cfg_weight, not cfg
            'temperature': temperature,
            'chunk_size': chunk_size,  # tokens per chunk
            'context_window': context_window,  # context window for each chunk
            'fade_duration': fade_duration,  # fade-in duration for each chunk
            'print_metrics': job_input.get('print_metrics', True)
        }
        
        # Note: The streaming API doesn't support seed parameter
        
        # Handle voice cloning
        temp_ref_path = None
        if voice_mode == 'clone' and reference_audio_b64:
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
        
        # Generate speech using streaming API
        streamed_chunks = []
        chunk_count = 0
        
        logger.info("Starting streaming TTS generation...")
        for audio_chunk, metrics in tts_model.generate_stream(**generation_params):
            chunk_count += 1
            streamed_chunks.append(audio_chunk)
            
            if chunk_count == 1:
                logger.info(f"First chunk received - shape: {audio_chunk.shape}")
            
            # Log metrics if available
            if metrics and job_input.get('print_metrics', False):
                logger.info(f"Chunk {chunk_count} metrics: {metrics}")
        
        # Concatenate all streaming chunks
        if streamed_chunks:
            wav = torch.cat(streamed_chunks, dim=-1)
            logger.info(f"Streaming complete - {chunk_count} chunks, final shape: {wav.shape}")
        else:
            raise RuntimeError("No audio chunks generated from streaming API")
        
        # Clean up temporary file
        if temp_ref_path:
            try:
                Path(temp_ref_path).unlink()
            except:
                pass
        
        # Get model sample rate
        output_sr = sample_rate if sample_rate else getattr(tts_model, 'sr', 24000)
        
        # Resample if different sample rate requested
        if sample_rate and sample_rate != getattr(tts_model, 'sr', 24000):
            wav = librosa.resample(wav, orig_sr=getattr(tts_model, 'sr', 24000), target_sr=sample_rate)
        
        # Apply audio normalization if specified
        if audio_normalization == 'peak':
            wav = wav / np.max(np.abs(wav))
        elif audio_normalization == 'rms':
            rms = np.sqrt(np.mean(wav**2))
            wav = wav / rms * 0.1
        
        # Convert to base64
        audio_base64 = audio_to_base64(wav, output_sr)
        
        # Prepare response
        response = {
            "audio": audio_base64,
            "sample_rate": output_sr,
            "text": text,
            "parameters": {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "voice_mode": voice_mode,
                "chunk_size": generation_params['chunk_size'],
                "context_window": generation_params['context_window'],
                "fade_duration": generation_params['fade_duration']
            }
        }
        
        logger.info(f"TTS generation successful - Duration: {len(wav) / output_sr:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise RuntimeError(f"TTS generation failed: {str(e)}")

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