import runpod
import torch
import tempfile
import base64
import io
import logging
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
tts_model: Optional[ChatterboxTTS] = None
vc_model: Optional[ChatterboxVC] = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    """Load TTS and VC models on startup"""
    global tts_model, vc_model
    
    logger.info(f"Loading models on device: {device}")
    
    try:
        logger.info("Loading ChatterboxTTS model...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("ChatterboxTTS model loaded successfully")
        
        logger.info("Loading ChatterboxVC model...")
        vc_model = ChatterboxVC.from_pretrained(device=device)
        logger.info("ChatterboxVC model loaded successfully")
        
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
    """Generate text-to-speech audio with full parameter control"""
    try:
        # Extract and validate text
        text = job_input.get('text', 'Hello, this is a test.')
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(text) > 5000:  # Increased limit for chunking support
            raise ValueError("Text too long (max 5000 characters)")
        
        # Core synthesis parameters
        exaggeration = job_input.get('exaggeration', 0.5)
        cfg_weight = job_input.get('cfg_weight', 0.5)
        temperature = job_input.get('temperature', 0.8)
        speed_factor = job_input.get('speed_factor', 1.0)
        seed = job_input.get('seed', job_input.get('random_seed', None))
        
        # Voice parameters
        voice_mode = job_input.get('voice_mode', 'predefined')
        predefined_voice_id = job_input.get('predefined_voice_id', None)
        reference_audio_filename = job_input.get('reference_audio_filename', None)
        reference_audio_b64 = job_input.get('reference_audio', None)
        max_reference_duration_sec = job_input.get('max_reference_duration_sec', 30)
        
        # Audio output parameters
        output_format = job_input.get('output_format', 'wav')
        sample_rate = job_input.get('sample_rate', None)
        audio_normalization = job_input.get('audio_normalization', None)
        
        # Text processing parameters
        split_text = job_input.get('split_text', False)
        chunk_size = job_input.get('chunk_size', 120)
        batching = job_input.get('batching', False)
        candidates_per_chunk = job_input.get('candidates_per_chunk', 1)
        retries = job_input.get('retries', 1)
        parallel_workers = job_input.get('parallel_workers', 1)
        
        # Language and post-processing
        language = job_input.get('language', 'en')
        sound_word_remove_replace = job_input.get('sound_word_remove_replace', None)
        auto_editor_margin = job_input.get('auto_editor_margin', None)
        whisper_model = job_input.get('whisper_model', None)
        
        # Validate core parameters
        if not 0.25 <= exaggeration <= 2.0:
            raise ValueError("Exaggeration must be between 0.25 and 2.0")
        if not 0.2 <= cfg_weight <= 1.0:
            raise ValueError("CFG weight must be between 0.2 and 1.0")
        if not 0.05 <= temperature <= 5.0:
            raise ValueError("Temperature must be between 0.05 and 5.0")
        if not 0.1 <= speed_factor <= 3.0:
            raise ValueError("Speed factor must be between 0.1 and 3.0")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("Seed must be an integer")
        
        # Validate voice parameters
        if voice_mode not in ['predefined', 'clone']:
            raise ValueError("Voice mode must be 'predefined' or 'clone'")
        if voice_mode == 'clone' and not reference_audio_b64 and not reference_audio_filename:
            raise ValueError("Reference audio required for voice cloning")
        if not 1 <= max_reference_duration_sec <= 60:
            raise ValueError("Max reference duration must be between 1 and 60 seconds")
        
        # Validate processing parameters
        if not 10 <= chunk_size <= 500:
            raise ValueError("Chunk size must be between 10 and 500 characters")
        if not 1 <= candidates_per_chunk <= 5:
            raise ValueError("Candidates per chunk must be between 1 and 5")
        if not 1 <= retries <= 5:
            raise ValueError("Retries must be between 1 and 5")
        if not 1 <= parallel_workers <= 8:
            raise ValueError("Parallel workers must be between 1 and 8")
        
        # Validate audio parameters
        if output_format not in ['wav', 'opus', 'mp3']:
            raise ValueError("Output format must be 'wav', 'opus', or 'mp3'")
        if sample_rate is not None and sample_rate not in [16000, 22050, 24000, 44100, 48000]:
            raise ValueError("Sample rate must be 16000, 22050, 24000, 44100, or 48000")
        
        logger.info(f"TTS request - Text: {len(text)} chars, Mode: {voice_mode}, Exaggeration: {exaggeration}")
        
        # Prepare generation parameters
        generation_params = {
            'text': text,
            'exaggeration': exaggeration,
            'cfg_weight': cfg_weight,
            'temperature': temperature
        }
        
        # Add seed if specified
        if seed is not None:
            generation_params['seed'] = seed
        
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
        
        # Generate speech
        wav = tts_model.generate(**generation_params)
        
        # Clean up temporary file
        if temp_ref_path:
            try:
                Path(temp_ref_path).unlink()
            except:
                pass
        
        # Apply speed factor post-processing if needed
        if speed_factor != 1.0:
            wav = librosa.effects.time_stretch(wav, rate=speed_factor)
        
        # Resample if different sample rate requested
        output_sr = sample_rate or tts_model.sr
        if output_sr != tts_model.sr:
            wav = librosa.resample(wav, orig_sr=tts_model.sr, target_sr=output_sr)
        
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
            "duration": len(wav) / output_sr,
            "parameters": {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "speed_factor": speed_factor,
                "voice_mode": voice_mode,
                "output_format": output_format,
                "language": language
            }
        }
        
        # Add optional parameters to response if used
        if seed is not None:
            response["parameters"]["seed"] = seed
        if sample_rate is not None:
            response["parameters"]["sample_rate"] = sample_rate
        if audio_normalization is not None:
            response["parameters"]["audio_normalization"] = audio_normalization
        if split_text:
            response["parameters"]["split_text"] = split_text
            response["parameters"]["chunk_size"] = chunk_size
        
        return response
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise e

def voice_conversion(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Perform voice conversion with advanced parameters"""
    try:
        # Extract parameters
        source_audio_b64 = job_input.get('source_audio')
        target_audio_b64 = job_input.get('target_audio')
        
        # Advanced VC parameters
        max_source_duration = job_input.get('max_source_duration_sec', 60)
        max_target_duration = job_input.get('max_target_duration_sec', 60)
        output_format = job_input.get('output_format', 'wav')
        sample_rate = job_input.get('sample_rate', None)
        audio_normalization = job_input.get('audio_normalization', None)
        
        if not source_audio_b64 or not target_audio_b64:
            raise ValueError("Both source_audio and target_audio are required")
        
        # Convert base64 to audio
        source_audio, source_sr = base64_to_audio(source_audio_b64)
        target_audio, target_sr = base64_to_audio(target_audio_b64)
        
        # Trim to max duration
        max_source_samples = int(max_source_duration * source_sr)
        max_target_samples = int(max_target_duration * target_sr)
        
        if len(source_audio) > max_source_samples:
            source_audio = source_audio[:max_source_samples]
            logger.info(f"Trimmed source audio to {max_source_duration} seconds")
        
        if len(target_audio) > max_target_samples:
            target_audio = target_audio[:max_target_samples]
            logger.info(f"Trimmed target audio to {max_target_duration} seconds")
        
        logger.info(f"VC request - Source: {len(source_audio)/source_sr:.2f}s, Target: {len(target_audio)/target_sr:.2f}s")
        
        # Resample if needed
        if source_sr != 16000:
            source_audio = librosa.resample(source_audio, orig_sr=source_sr, target_sr=16000)
            logger.info(f"Resampling source audio from {source_sr}Hz to 16000Hz")
        
        if target_sr != 24000:
            target_audio = librosa.resample(target_audio, orig_sr=target_sr, target_sr=24000)
            logger.info(f"Resampling target audio from {target_sr}Hz to 24000Hz")
        
        # Perform voice conversion
        converted_audio = vc_model.convert(
            source_audio=source_audio,
            target_audio=target_audio
        )
        
        # Apply audio normalization if specified
        if audio_normalization == 'peak':
            converted_audio = converted_audio / np.max(np.abs(converted_audio))
        elif audio_normalization == 'rms':
            rms = np.sqrt(np.mean(converted_audio**2))
            converted_audio = converted_audio / rms * 0.1
        
        # Resample if different sample rate requested
        output_sr = sample_rate or 24000
        if output_sr != 24000:
            converted_audio = librosa.resample(converted_audio, orig_sr=24000, target_sr=output_sr)
        
        # Convert to base64
        audio_base64 = audio_to_base64(converted_audio, output_sr)
        
        return {
            "audio": audio_base64,
            "sample_rate": output_sr,
            "duration": len(converted_audio) / output_sr,
            "source_duration": len(source_audio) / 16000,
            "target_duration": len(target_audio) / 24000,
            "parameters": {
                "output_format": output_format,
                "audio_normalization": audio_normalization
            }
        }
        
    except Exception as e:
        logger.error(f"Voice conversion failed: {e}")
        raise e

def handler(job):
    """Main handler function for Runpod serverless"""
    try:
        job_input = job.get('input', {})
        task_type = job_input.get('task', 'tts')
        
        if task_type == 'tts':
            result = generate_tts(job_input)
        elif task_type == 'vc':
            result = voice_conversion(job_input)
        else:
            raise ValueError(f"Unknown task type: {task_type}. Use 'tts' or 'vc'")
        
        return result
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

# Load models when the handler starts
load_models()

# Start the serverless handler
if __name__ == "__main__":
    logger.info("Starting Runpod serverless handler...")
    runpod.serverless.start({"handler": handler}) 