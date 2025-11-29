# Standalone Voice Embedding Generator - Usage Guide

## Overview

`generate_voice_embedding.py` is a **standalone module** that can be used by external applications to create voice embeddings compatible with your Chatterbox TTS handler. No modifications to the handler are needed.

## Installation

The module requires these dependencies:
```bash
pip install librosa soundfile numpy
```

## Quick Start

### Option 1: Use as a Module (Recommended)

```python
from generate_voice_embedding import VoiceEmbeddingGenerator

# Initialize generator (point to your handler's voice_embeddings directory)
generator = VoiceEmbeddingGenerator(embeddings_dir="./voice_embeddings")

# Create embedding from audio file
result = generator.from_file(
    audio_file="my_voice.wav",
    voice_name="MyVoice",
    gender="male",
    voice_description="My custom voice"
)

print(f"Voice ID: {result['voice_id']}")
print(f"Voice Name: {result['voice_name']}")
```

### Option 2: Use Convenience Function

```python
from generate_voice_embedding import create_voice_embedding

# Automatically detects input type
result = create_voice_embedding(
    audio_input="my_voice.wav",  # Can be file path, numpy array, or base64
    voice_name="MyVoice",
    gender="male"
)
```

### Option 3: Command Line

```bash
python generate_voice_embedding.py my_voice.wav MyVoice male "My custom voice"
```

## Input Methods

### From Audio File
```python
generator = VoiceEmbeddingGenerator()
result = generator.from_file(
    audio_file="path/to/audio.wav",
    voice_name="VoiceName",
    gender="male"  # Optional: "male" or "female"
)
```

### From Numpy Array
```python
import numpy as np
import soundfile as sf

audio_array, sample_rate = sf.read("audio.wav")

generator = VoiceEmbeddingGenerator()
result = generator.from_array(
    audio_array=audio_array,
    sample_rate=sample_rate,
    voice_name="VoiceName"
)
```

### From Base64 String
```python
import base64

with open("audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

generator = VoiceEmbeddingGenerator()
result = generator.from_base64(
    audio_base64=audio_base64,
    voice_name="VoiceName"
)
```

## Output Structure

The generator creates files in the `voice_embeddings` directory:

```
voice_embeddings/
├── embeddings/
│   └── {voice_id}.pkl.gz      # Compressed audio array
├── metadata/
│   └── {voice_id}.json        # Voice metadata
├── voice_catalog.json          # Updated catalog
├── name_mapping.json           # Updated name→ID mapping
└── stats.json                  # Updated statistics
```

## Return Value

```python
{
    "success": True,
    "voice_id": "a1b2c3d4",           # 8-char hex ID
    "voice_name": "MyVoice",
    "embedding_path": "embeddings/a1b2c3d4.pkl.gz",
    "metadata_path": "metadata/a1b2c3d4.json",
    "duration": 23.45,                # seconds
    "sample_rate": 24000,
    "message": "Voice 'MyVoice' saved successfully"
}
```

## Using Generated Voices in Handler

After generating the embedding, you can use it immediately in your handler:

```python
# In your API request
{
    "operation": "tts",
    "mode": "basic",
    "text": "Hello world!",
    "voice_name": "MyVoice"  # Uses the generated embedding
}

# Or by ID
{
    "operation": "tts",
    "mode": "streaming",
    "text": "Hello world!",
    "voice_id": "a1b2c3d4"
}
```

## Integration Examples

### Example 1: Frontend App (Python)

```python
# frontend_app.py
from generate_voice_embedding import VoiceEmbeddingGenerator
import requests

# User uploads audio file
audio_file = "user_uploaded_voice.wav"
voice_name = "UserVoice123"

# Generate embedding
generator = VoiceEmbeddingGenerator(embeddings_dir="./shared_voice_embeddings")
result = generator.from_file(audio_file, voice_name, gender="female")

# Now use in TTS API
api_response = requests.post(
    "https://your-api-endpoint/tts",
    json={
        "operation": "tts",
        "mode": "basic",
        "text": "Hello!",
        "voice_name": voice_name  # Works immediately!
    }
)
```

### Example 2: Batch Processing

```python
from generate_voice_embedding import VoiceEmbeddingGenerator
from pathlib import Path

generator = VoiceEmbeddingGenerator()

# Process multiple files
audio_files = [
    ("voice1.wav", "Voice1", "male"),
    ("voice2.wav", "Voice2", "female"),
    ("voice3.wav", "Voice3", "male"),
]

results = []
for audio_file, name, gender in audio_files:
    if Path(audio_file).exists():
        result = generator.from_file(audio_file, name, gender=gender)
        results.append(result)
        print(f"✅ Created: {name} ({result['voice_id']})")
    else:
        print(f"❌ File not found: {audio_file}")

print(f"\nTotal voices created: {len(results)}")
```

### Example 3: From Web Upload (Base64)

```python
# web_app.py
from flask import Flask, request, jsonify
from generate_voice_embedding import VoiceEmbeddingGenerator

app = Flask(__name__)
generator = VoiceEmbeddingGenerator(embeddings_dir="./voice_embeddings")

@app.route('/create_voice', methods=['POST'])
def create_voice():
    data = request.json
    audio_base64 = data['audio']  # Base64 encoded audio
    voice_name = data['voice_name']
    gender = data.get('gender')
    
    try:
        result = generator.from_base64(
            audio_base64=audio_base64,
            voice_name=voice_name,
            gender=gender
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
```

## Important Notes

1. **Directory Location**: Point `embeddings_dir` to the same directory your handler uses (usually `voice_embeddings`)

2. **Voice Names**: Must be unique. If a name already exists, it will overwrite the name mapping.

3. **Audio Quality**: 
   - Recommended: 15-30 seconds of clear speech
   - Minimum: 5 seconds
   - Maximum: 60 seconds (will be trimmed)
   - Sample rate: Any (automatically resampled to 24kHz)

4. **File Format**: Supports WAV, MP3, FLAC, and other formats supported by librosa

5. **No Handler Changes**: The generated embeddings work immediately with your existing handler - no modifications needed!

## Troubleshooting

### "librosa not found"
```bash
pip install librosa soundfile
```

### "Voice name already exists"
The generator will warn but allow it. The new voice will overwrite the name mapping.

### "Embedding not found in handler"
Make sure `embeddings_dir` points to the same directory the handler uses. Check the handler's `voice_embeddings` path.

## Compatibility

✅ Works with existing handler without modifications  
✅ Compatible with `optimized_voice_library.py`  
✅ Compatible with `local_voice_library.py`  
✅ Follows exact file format expected by handler  
✅ Updates all required metadata files  

