# Guide for External Applications

## Overview

This guide explains how external applications can prepare audio for the Chatterbox TTS API **without requiring ChatterboxTTS to be installed**.

## Key Point: Your API Accepts Audio Arrays, Not Embeddings

Your handler accepts:
- ✅ `reference_audio` (base64 encoded audio)
- ✅ `voice_id` (from voice library)
- ✅ `voice_name` (from voice library)

The handler then passes these audio arrays to ChatterboxTTS, which extracts embeddings internally.

## Solution: Standalone Audio Preparation

Use `standalone_audio_prep.py` to prepare audio without ChatterboxTTS:

### Installation (External App)

```bash
pip install librosa soundfile numpy
```

**That's it!** No ChatterboxTTS needed.

### Usage

#### Option 1: Prepare Audio and Send as Base64

```python
from standalone_audio_prep import prepare_and_encode_for_api

# Prepare audio from file and encode to base64
audio_b64 = prepare_and_encode_for_api("my_voice.wav")

# Send to your API
import requests

api_request = {
    "text": "Hello world!",
    "reference_audio": audio_b64  # <-- Send as base64
}

response = requests.post("https://your-api-endpoint.com/generate", json=api_request)
```

#### Option 2: Prepare Audio Array and Encode Manually

```python
from standalone_audio_prep import prepare_audio_for_api, audio_to_base64

# Prepare audio
audio_array, sample_rate = prepare_audio_for_api("my_voice.wav")

# Convert to base64
audio_b64 = audio_to_base64(audio_array, sample_rate)

# Send to API
api_request = {
    "text": "Hello world!",
    "reference_audio": audio_b64
}
```

#### Option 3: Use Existing `generate_voice_embedding.py` (Saves to Library)

If you want to save the voice to the library (so it can be used with `voice_id` or `voice_name`):

```python
from generate_voice_embedding import create_voice_embedding

# This saves to voice_embeddings/ directory
result = create_voice_embedding(
    audio_file="my_voice.wav",
    voice_name="MyVoice",
    embeddings_dir="voice_embeddings"  # Must match your API's directory
)

# Now you can use voice_id or voice_name in API calls
api_request = {
    "text": "Hello world!",
    "voice_id": result["voice_id"]  # or "voice_name": "MyVoice"
}
```

## Complete Example: External App Integration

```python
"""
Example: External app preparing audio and calling Chatterbox TTS API
"""

from standalone_audio_prep import prepare_and_encode_for_api
import requests
import json

class ChatterboxTTSClient:
    def __init__(self, api_endpoint: str):
        self.api_endpoint = api_endpoint
    
    def generate_tts(
        self,
        text: str,
        reference_audio_file: str = None,
        reference_audio_b64: str = None,
        voice_id: str = None,
        voice_name: str = None
    ):
        """
        Generate TTS with voice cloning
        
        Args:
            text: Text to synthesize
            reference_audio_file: Path to audio file (will be prepared automatically)
            reference_audio_b64: Pre-prepared base64 audio (alternative to file)
            voice_id: Voice ID from library
            voice_name: Voice name from library
        """
        # Prepare request
        request_data = {
            "text": text
        }
        
        # Handle reference audio
        if reference_audio_file:
            # Prepare audio from file
            audio_b64 = prepare_and_encode_for_api(reference_audio_file)
            request_data["reference_audio"] = audio_b64
        elif reference_audio_b64:
            # Use pre-prepared base64
            request_data["reference_audio"] = reference_audio_b64
        elif voice_id:
            # Use voice from library
            request_data["voice_id"] = voice_id
        elif voice_name:
            # Use voice from library by name
            request_data["voice_name"] = voice_name
        
        # Call API
        response = requests.post(
            f"{self.api_endpoint}/generate",
            json=request_data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")


# Usage
client = ChatterboxTTSClient("https://your-api-endpoint.com")

# Option 1: Use audio file directly
result = client.generate_tts(
    text="Hello world!",
    reference_audio_file="my_voice.wav"
)

# Option 2: Use pre-prepared base64
audio_b64 = prepare_and_encode_for_api("my_voice.wav")
result = client.generate_tts(
    text="Hello world!",
    reference_audio_b64=audio_b64
)

# Option 3: Use voice from library (if saved)
result = client.generate_tts(
    text="Hello world!",
    voice_name="MyVoice"
)
```

## What Happens Internally

1. **External App**: Prepares audio (resamples to 24kHz, normalizes, trims)
2. **External App**: Encodes to base64
3. **External App**: Sends to API with `reference_audio` field
4. **Your API**: Decodes base64 to audio array
5. **Your API**: Passes audio array to ChatterboxTTS
6. **ChatterboxTTS**: Extracts embeddings internally
7. **ChatterboxTTS**: Generates TTS

## File Structure for External App

```
external_app/
├── standalone_audio_prep.py    # Copy this file
├── requirements.txt            # librosa, soundfile, numpy
└── your_app.py                 # Your application code
```

## Requirements for External App

**Minimal dependencies:**
```txt
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0
```

**No ChatterboxTTS needed!**

## Comparison: Embeddings vs Audio Arrays

| Approach | Requires ChatTTS? | File Size | Complexity | Recommended |
|----------|------------------|-----------|------------|-------------|
| **Audio Arrays (Current)** | ❌ No | Medium | ⭐ Simple | ✅ **Yes** |
| Pre-extracted Embeddings | ✅ Yes | Small | ⭐⭐⭐ Complex | ❌ No |

## Summary

✅ **Use `standalone_audio_prep.py`** - it prepares audio without ChatterboxTTS  
✅ **Send audio as base64** in the `reference_audio` field  
✅ **Your API handles the rest** - extracts embeddings internally  
✅ **No ChatterboxTTS needed** in external app - only librosa/soundfile  

This is the simplest and most practical approach for external applications!

