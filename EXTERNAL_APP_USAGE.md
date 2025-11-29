# Using `generate_voice_embedding.py` in External Apps

## ✅ YES - It Works in Other Apps!

`generate_voice_embedding.py` is **completely standalone** and can be used in any external application.

## Dependencies (External App Only Needs These)

```bash
pip install librosa soundfile numpy
```

**NO ChatterboxTTS required!**  
**NO access to your codebase needed!**

## How to Use in External App

### Step 1: Copy the File

Copy `generate_voice_embedding.py` to your external app:

```
external_app/
├── generate_voice_embedding.py  # Copy this file
├── requirements.txt
└── your_app.py
```

### Step 2: Install Dependencies

```bash
pip install librosa soundfile numpy
```

### Step 3: Use It

```python
# In your external app
from generate_voice_embedding import create_voice_embedding

# Create voice embedding
result = create_voice_embedding.from_file(
    audio_file="user_voice.wav",
    voice_name="UserVoice",
    embeddings_dir="./voice_embeddings",  # Where to save
    gender="female"
)

print(f"Voice ID: {result['voice_id']}")
print(f"Voice Name: {result['voice_name']}")
print(f"Saved to: {result['embedding_path']}")
```

### Step 4: Use with Your API

```python
import requests

# Use the created voice with your API
api_request = {
    "text": "Hello world!",
    "voice_id": result['voice_id']  # ✅ Works!
}

response = requests.post("https://your-api.com/generate", json=api_request)
```

## What It Creates

The script creates files in the `voice_embeddings/` directory:

```
voice_embeddings/
├── embeddings/
│   └── {voice_id}.pkl.gz      # Audio array (compressed)
├── metadata/
│   └── {voice_id}.json        # Voice metadata
├── voice_catalog.json         # Updated catalog
└── name_mapping.json          # Updated name mapping
```

**These are the EXACT same files your handler expects!**

## Complete Example

```python
"""
External app creating voices for Chatterbox TTS API
"""

from generate_voice_embedding import create_voice_embedding
import requests
import os

class VoiceManager:
    def __init__(self, embeddings_dir="./voice_embeddings", api_endpoint=None):
        self.embeddings_dir = embeddings_dir
        self.api_endpoint = api_endpoint
    
    def create_voice(self, audio_file, voice_name, gender="unknown"):
        """Create a new voice embedding"""
        result = create_voice_embedding.from_file(
            audio_file=audio_file,
            voice_name=voice_name,
            embeddings_dir=self.embeddings_dir,
            gender=gender
        )
        return result
    
    def generate_tts(self, text, voice_id=None, voice_name=None):
        """Generate TTS using created voice"""
        if not self.api_endpoint:
            raise ValueError("API endpoint not set")
        
        request_data = {"text": text}
        
        if voice_id:
            request_data["voice_id"] = voice_id
        elif voice_name:
            request_data["voice_name"] = voice_name
        else:
            raise ValueError("Must provide voice_id or voice_name")
        
        response = requests.post(
            f"{self.api_endpoint}/generate",
            json=request_data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error: {response.status_code}")


# Usage
manager = VoiceManager(
    embeddings_dir="./voice_embeddings",
    api_endpoint="https://your-api.com"
)

# Create voice
voice = manager.create_voice(
    audio_file="my_voice.wav",
    voice_name="MyVoice",
    gender="male"
)

print(f"✅ Created voice: {voice['voice_id']}")

# Use voice
tts_result = manager.generate_tts(
    text="Hello! This is my cloned voice.",
    voice_id=voice['voice_id']
)

print("✅ TTS generated!")
```

## File Structure for External App

```
external_app/
├── generate_voice_embedding.py    # Copy from this repo
├── requirements.txt                # librosa, soundfile, numpy
├── voice_embeddings/              # Created by script
│   ├── embeddings/
│   ├── metadata/
│   ├── voice_catalog.json
│   └── name_mapping.json
└── your_app.py                    # Your application code
```

## Requirements.txt for External App

```txt
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0
requests>=2.31.0  # If calling API
```

## Key Points

✅ **Standalone** - No ChatterboxTTS needed  
✅ **No handler code** - Doesn't import anything from your codebase  
✅ **Same format** - Creates files your handler expects  
✅ **Works immediately** - Voices work with `voice_id`/`voice_name` in API  
✅ **Simple** - Only needs librosa + soundfile  

## Verification

The script:
- ✅ Only imports standard libraries + librosa/soundfile
- ✅ No imports from your handler code
- ✅ Creates exact format handler expects
- ✅ Can be copied to any app
- ✅ Works independently

**You can use it in any external app!** 🎉

