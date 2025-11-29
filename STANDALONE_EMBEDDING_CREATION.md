# Standalone Embedding Creation for External Apps

## ✅ YES - It Works Separately!

The `generate_voice_embedding.py` script **does NOT require ChatterboxTTS** and can be used by external applications.

## What Are "Embeddings" in Your System?

**Important**: Your "embeddings" are actually **audio arrays** stored in `.pkl.gz` files, not ML embeddings!

When you use `voice_id` or `voice_name` in your API:
1. Handler loads `.pkl.gz` file
2. Extracts the audio array from it
3. Passes audio array to ChatterboxTTS
4. ChatterboxTTS extracts embeddings internally

## Requirements for External App

**Only these dependencies needed:**
```bash
pip install librosa soundfile numpy
```

**NO ChatterboxTTS required!**

## How It Works

### Step 1: External App Creates Embeddings

```python
# In your external app
from generate_voice_embedding import create_voice_embedding

# Create embedding (saves to voice_embeddings/ directory)
result = create_voice_embedding.from_file(
    audio_file="my_voice.wav",
    voice_name="MyVoice",
    embeddings_dir="./voice_embeddings",  # Must match your API's directory
    gender="male"
)

print(f"Voice ID: {result['voice_id']}")
print(f"Saved to: {result['embedding_file']}")
```

This creates:
- `voice_embeddings/embeddings/{voice_id}.pkl.gz` - Audio array (compressed)
- `voice_embeddings/metadata/{voice_id}.json` - Metadata
- Updates `voice_catalog.json` and `name_mapping.json`

### Step 2: External App Calls API with voice_id

```python
import requests

# Use the voice_id from step 1
api_request = {
    "text": "Hello world!",
    "voice_id": result['voice_id']  # e.g., "1fc6c6ab"
}

# OR use voice_name
api_request = {
    "text": "Hello world!",
    "voice_name": "MyVoice"
}

response = requests.post("https://your-api-endpoint.com/generate", json=api_request)
```

### Step 3: Your Handler Uses the Embedding

Your handler (already modified):
1. Receives `voice_id` or `voice_name`
2. Loads `.pkl.gz` file from `voice_embeddings/embeddings/`
3. Extracts audio array
4. Passes to ChatterboxTTS (which extracts embeddings internally)

## File Structure

```
external_app/
├── generate_voice_embedding.py  # Copy this file
├── requirements.txt
└── your_app.py

voice_embeddings/                # Shared directory (or sync via API)
├── embeddings/
│   ├── 1fc6c6ab.pkl.gz         # Audio array (compressed)
│   └── de16e46c.pkl.gz
├── metadata/
│   ├── 1fc6c6ab.json
│   └── de16e46c.json
├── voice_catalog.json
└── name_mapping.json
```

## Complete Example

```python
"""
External app creating embeddings and using them with API
"""

from generate_voice_embedding import create_voice_embedding
import requests

# Step 1: Create embedding (standalone, no ChatTTS needed!)
result = create_voice_embedding.from_file(
    audio_file="user_voice.wav",
    voice_name="UserVoice",
    embeddings_dir="./voice_embeddings",
    gender="female"
)

print(f"✅ Created voice: {result['voice_id']}")

# Step 2: Upload/sync embeddings to API server
# (You'll need to implement this - could be via file upload API endpoint)

# Step 3: Use voice with API
api_request = {
    "text": "Hello! This is my cloned voice.",
    "voice_id": result['voice_id']  # Use the created voice
}

response = requests.post(
    "https://your-api-endpoint.com/generate",
    json=api_request
)

if response.status_code == 200:
    audio_data = response.json()['audio']
    print("✅ TTS generated successfully!")
else:
    print(f"❌ Error: {response.status_code}")
```

## Key Points

✅ **No ChatterboxTTS needed** in external app  
✅ **Only librosa + soundfile** required  
✅ **Same format** as your handler expects  
✅ **Works with voice_id/voice_name** in API calls  
✅ **Standalone** - no access to your codebase needed  

## Format Details

The `.pkl.gz` files contain:
```python
{
    'audio': np.ndarray,      # Audio samples at 24kHz
    'sample_rate': 24000,
    'voice_name': 'MyVoice',
    'created_at': '2024-01-01T12:00:00'
}
```

This is exactly what `optimized_voice_library.py` expects (lines 119-122):
```python
if isinstance(embedding_data, dict) and 'audio' in embedding_data:
    audio_array = embedding_data['audio']  # ✅ Matches!
```

## Verification

To verify it works:
1. Create embedding with `generate_voice_embedding.py`
2. Check that `.pkl.gz` file exists in `voice_embeddings/embeddings/`
3. Check that `voice_catalog.json` has the new voice
4. Call API with `voice_id` or `voice_name`
5. Should work! ✅

