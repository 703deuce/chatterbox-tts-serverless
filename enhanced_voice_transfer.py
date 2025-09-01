import requests
import base64
import io
import soundfile as sf
import librosa
import torch
import logging
from typing import Dict, Any, Optional, Union
from urllib.parse import urlparse
import tempfile
import os
import re

logger = logging.getLogger(__name__)

def is_firebase_storage_url(url: str) -> bool:
    """Check if URL is a Firebase Storage URL"""
    return 'firebasestorage.app' in url or 'googleapis.com' in url

def construct_firebase_url(storage_bucket: str, path: str) -> str:
    """
    Construct Firebase Storage download URL from bucket and path
    
    Args:
        storage_bucket: Firebase storage bucket name
        path: File path in storage (e.g., "users/tts/user123/audio.wav")
        
    Returns:
        str: Direct download URL
    """
    # Encode the path for URL
    import urllib.parse
    encoded_path = urllib.parse.quote(path, safe='/')
    
    # Firebase Storage direct download URL format
    firebase_url = f"https://firebasestorage.googleapis.com/v0/b/{storage_bucket}/o/{encoded_path}?alt=media"
    
    logger.info(f"Constructed Firebase URL: {firebase_url}")
    return firebase_url

def download_audio_from_url(url: str, timeout: int = 30) -> tuple:
    """
    Download audio file from URL and return audio array and sample rate
    Supports regular URLs and Firebase Storage URLs
    
    Args:
        url: URL to download audio from
        timeout: Request timeout in seconds
        
    Returns:
        tuple: (audio_array, sample_rate)
    """
    try:
        logger.info(f"Downloading audio from URL: {url}")
        
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: {url}")
        
        # Add special handling for Firebase Storage URLs
        headers = {}
        if is_firebase_storage_url(url):
            logger.info("Detected Firebase Storage URL")
            # Firebase Storage may require specific headers
            headers['User-Agent'] = 'Mozilla/5.0 (compatible; TTS-API/1.0)'
        
        # Download file
        response = requests.get(url, timeout=timeout, stream=True, headers=headers)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        try:
            # Load audio using librosa
            audio_array, sample_rate = librosa.load(temp_path, sr=None)
            logger.info(f"Successfully downloaded audio: {sample_rate}Hz, {len(audio_array)} samples")
            return audio_array, sample_rate
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download audio from URL: {e}")
    except Exception as e:
        raise ValueError(f"Error processing audio from URL: {e}")

def load_audio_from_input(audio_input: Union[str, bytes], is_url: bool = False, storage_bucket: str = None, storage_path: str = None) -> tuple:
    """
    Load audio from either base64 string, URL, or Firebase Storage path
    
    Args:
        audio_input: Base64 string, URL, or Firebase path
        is_url: True if input is a URL, False if base64
        storage_bucket: Firebase storage bucket name
        storage_path: Firebase storage path
        
    Returns:
        tuple: (audio_array, sample_rate)
    """
    if storage_bucket and storage_path:
        # Use Firebase Storage path
        firebase_url = construct_firebase_url(storage_bucket, storage_path)
        return download_audio_from_url(firebase_url)
    elif is_url:
        return download_audio_from_url(audio_input)
    else:
        # Handle base64 input (existing logic)
        try:
            audio_data = base64.b64decode(audio_input)
            audio_buffer = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_buffer)
            return audio_array, sample_rate
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio: {e}")

def generate_voice_transfer_enhanced(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    ENHANCED: Generate voice transfer supporting base64, URLs, and Firebase Storage paths
    """
    try:
        # Extract and validate required parameters
        input_audio = job_input.get('input_audio')
        transfer_mode = job_input.get('transfer_mode')
        
        if not input_audio:
            raise ValueError("input_audio is required")
        if transfer_mode not in ['embedding', 'audio']:
            raise ValueError("transfer_mode must be 'embedding' or 'audio'")
        
        # Extract Firebase Storage configuration
        storage_bucket = job_input.get('storage_bucket', 'aitts-d4c6d.firebasestorage.app')  # Default bucket
        input_storage_path = job_input.get('input_storage_path')  # e.g., "users/tts/user123/input.wav"
        
        # Determine input method: Firebase path, URL, or base64
        input_is_url = job_input.get('input_is_url', False)
        if isinstance(input_audio, str) and (input_audio.startswith('http://') or input_audio.startswith('https://')):
            input_is_url = True
        
        # Load input audio
        input_audio_array, input_sr = load_audio_from_input(
            audio_input=input_audio,
            is_url=input_is_url,
            storage_bucket=storage_bucket,
            storage_path=input_storage_path
        )
        logger.info(f"Voice transfer: {transfer_mode} mode - Input audio: {input_sr}Hz, {len(input_audio_array)} samples")
        
        if transfer_mode == 'embedding':
            # WAV to Voice Embedding mode
            voice_name = job_input.get('voice_name')
            voice_id = job_input.get('voice_id')
            
            if not voice_name and not voice_id:
                raise ValueError("voice_name or voice_id is required for embedding mode")
            
            # Use optimized voice loading
            audio_prompt_array = handle_voice_cloning_source_optimized(job_input)
            if audio_prompt_array is not None:
                target_audio_array = audio_prompt_array
                target_sr = 24000  # Default sample rate for embeddings
                logger.info(f"OPTIMIZED: Loaded target voice directly from embeddings: {len(target_audio_array)} samples")
                
                # Determine input source type for response info
                input_source_type = "firebase_path" if input_storage_path else ("url" if input_is_url else "base64")
                
                transfer_info = {
                    "transfer_mode": "embedding",
                    "target_voice": voice_name or voice_id,
                    "source": "voice_library",
                    "voice_loading": "optimized_direct_access",
                    "input_source": input_source_type
                }
                optimization_indicator = "direct_audio_array"
            else:
                return {
                    "error": "Voice not found in library",
                    "message": f"Voice '{voice_name or voice_id}' not found. Use list_local_voices operation to see available voices."
                }
                
        elif transfer_mode == 'audio':
            # WAV to WAV mode
            target_audio = job_input.get('target_audio')
            if not target_audio:
                raise ValueError("target_audio is required for audio mode")
            
            # Extract target Firebase Storage path
            target_storage_path = job_input.get('target_storage_path')  # e.g., "users/tts/user123/target.wav"
            
            # Determine if target is Firebase path, URL, or base64
            target_is_url = job_input.get('target_is_url', False)
            if isinstance(target_audio, str) and (target_audio.startswith('http://') or target_audio.startswith('https://')):
                target_is_url = True
            
            # Load target audio
            target_audio_array, target_sr = load_audio_from_input(
                audio_input=target_audio,
                is_url=target_is_url,
                storage_bucket=storage_bucket,
                storage_path=target_storage_path
            )
            logger.info(f"Using provided target audio: {target_sr}Hz, {len(target_audio_array)} samples")
            
            # Determine source types for response info
            input_source_type = "firebase_path" if input_storage_path else ("url" if input_is_url else "base64")
            target_source_type = "firebase_path" if target_storage_path else ("url" if target_is_url else "base64")
            
            transfer_info = {
                "transfer_mode": "audio",
                "target_source": "user_provided_audio",
                "target_duration": len(target_audio_array) / target_sr,
                "input_source": input_source_type,
                "target_source_type": target_source_type
            }
            optimization_indicator = None
        
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
        
        # Convert to base64 for response
        def audio_to_base64(audio_array, sample_rate):
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='WAV')
            buffer.seek(0)
            audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            return audio_b64
        
        audio_b64 = audio_to_base64(transferred_wav, S3GEN_SR)
        
        # Return response
        response = {
            "audio": audio_b64,
            "sample_rate": S3GEN_SR,
            "duration": len(transferred_wav) / S3GEN_SR,
            "format": "wav",
            "model": "s3gen",
            "operation": "voice_transfer",
            "transfer_info": transfer_info,
            "input_duration": len(input_audio_array) / input_sr,
            "processing_time": "30-90 seconds typical"
        }
        
        if optimization_indicator:
            response["optimization"] = optimization_indicator
        
        logger.info(f"Voice transfer completed successfully: {response['duration']:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Voice transfer failed: {e}")
        return {
            "error": str(e),
            "message": "Voice transfer processing failed"
        }

# Example usage functions for the API
def submit_voice_transfer_with_firebase_path(api_key: str, input_path: str, target_voice_name: str, 
                                           storage_bucket: str = "aitts-d4c6d.firebasestorage.app", 
                                           no_watermark: bool = False) -> str:
    """
    Submit voice transfer job using Firebase Storage path (embedding mode)
    
    Args:
        api_key: RunPod API key
        input_path: Firebase Storage path (e.g., "users/tts/user123/audio.wav")
        target_voice_name: Target voice name from library
        storage_bucket: Firebase storage bucket name
        no_watermark: Skip watermarking
        
    Returns:
        str: Job ID
    """
    url = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "embedding",
            "input_audio": "",  # Not used when using Firebase path
            "input_storage_path": input_path,
            "storage_bucket": storage_bucket,
            "voice_name": target_voice_name,
            "no_watermark": no_watermark
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()['id']

def submit_voice_transfer_with_url(api_key: str, input_url: str, target_voice_name: str, no_watermark: bool = False) -> str:
    """
    Submit voice transfer job using input audio URL (embedding mode)
    
    Args:
        api_key: RunPod API key
        input_url: URL to input audio file
        target_voice_name: Target voice name from library
        no_watermark: Skip watermarking
        
    Returns:
        str: Job ID
    """
    url = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "embedding",
            "input_audio": input_url,
            "input_is_url": True,
            "voice_name": target_voice_name,
            "no_watermark": no_watermark
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()['id']

def submit_audio_to_audio_with_firebase_paths(api_key: str, input_path: str, target_path: str,
                                             storage_bucket: str = "aitts-d4c6d.firebasestorage.app",
                                             no_watermark: bool = False) -> str:
    """
    Submit voice transfer job using Firebase Storage paths for both input and target audio
    
    Args:
        api_key: RunPod API key
        input_path: Firebase Storage path for input audio (e.g., "users/tts/user123/input.wav")
        target_path: Firebase Storage path for target audio (e.g., "users/tts/user123/target.wav")
        storage_bucket: Firebase storage bucket name
        no_watermark: Skip watermarking
        
    Returns:
        str: Job ID
    """
    url = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "audio",
            "input_audio": "",  # Not used when using Firebase paths
            "input_storage_path": input_path,
            "target_audio": "",  # Not used when using Firebase paths
            "target_storage_path": target_path,
            "storage_bucket": storage_bucket,
            "no_watermark": no_watermark
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()['id']

def submit_audio_to_audio_with_urls(api_key: str, input_url: str, target_url: str, no_watermark: bool = False) -> str:
    """
    Submit voice transfer job using URLs for both input and target audio
    
    Args:
        api_key: RunPod API key
        input_url: URL to input audio file
        target_url: URL to target audio file
        no_watermark: Skip watermarking
        
    Returns:
        str: Job ID
    """
    url = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "audio",
            "input_audio": input_url,
            "input_is_url": True,
            "target_audio": target_url,
            "target_is_url": True,
            "no_watermark": no_watermark
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()['id']

# Example usage
if __name__ == "__main__":
    api_key = "YOUR_API_KEY"
    
    # Example 1: Transfer using Firebase Storage path to voice embedding
    try:
        job_id = submit_voice_transfer_with_firebase_path(
            api_key=api_key,
            input_path="users/tts/user123/input_audio.wav",
            target_voice_name="Amy",
            storage_bucket="aitts-d4c6d.firebasestorage.app",
            no_watermark=True
        )
        print(f"Firebase path job submitted: {job_id}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Transfer between two Firebase Storage audio files
    try:
        job_id = submit_audio_to_audio_with_firebase_paths(
            api_key=api_key,
            input_path="users/tts/user123/source.wav",
            target_path="users/tts/user123/target.wav",
            storage_bucket="aitts-d4c6d.firebasestorage.app",
            no_watermark=True
        )
        print(f"Firebase audio-to-audio job submitted: {job_id}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Transfer using input URL to voice embedding (original method)
    try:
        job_id = submit_voice_transfer_with_url(
            api_key=api_key,
            input_url="https://example.com/input_audio.wav",
            target_voice_name="Amy",
            no_watermark=True
        )
        print(f"URL job submitted: {job_id}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 4: Transfer between two audio URLs (original method)
    try:
        job_id = submit_audio_to_audio_with_urls(
            api_key=api_key,
            input_url="https://example.com/source_audio.wav",
            target_url="https://example.com/target_voice.wav",
            no_watermark=True
        )
        print(f"URL audio-to-audio job submitted: {job_id}")
        
    except Exception as e:
        print(f"Error: {e}")
