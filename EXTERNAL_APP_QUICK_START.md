# External App Quick Start Guide

## What You Need to Do (Step-by-Step)

### 1. Create Voice Embedding

```python
from generate_voice_embedding import create_voice_embedding

result = create_voice_embedding.from_file(
    audio_file="user_voice.wav",
    voice_name="UserVoice",
    embeddings_dir="./temp_embeddings"
)

voice_id = result['voice_id']  # Save this!
```

### 2. Upload to Firebase Storage

```python
from firebase_admin import storage

bucket = storage.bucket()
firebase_path = f"voices/embeddings/{voice_id}.pkl.gz"

blob = bucket.blob(firebase_path)
blob.upload_from_filename(f"temp_embeddings/embeddings/{voice_id}.pkl.gz")
blob.make_public()  # Important!
```

### 3. Prepare Metadata

```python
import json

# Read metadata
with open(f"temp_embeddings/metadata/{voice_id}.json", 'r') as f:
    voice_metadata = json.load(f)

# Add Firebase paths (REQUIRED!)
voice_metadata['firebase_storage_path'] = f"voices/embeddings/{voice_id}.pkl.gz"
voice_metadata['firebase_storage_bucket'] = "aitts-d4c6d.firebasestorage.app"

# Save to your database
db.collection('voices').document(voice_id).set(voice_metadata)
```

### 4. Call Handler API

```python
import requests

api_request = {
    "operation": "tts",
    "text": "Hello world!",
    "voice_id": voice_id,
    "voice_metadata": voice_metadata  # Include this!
}

response = requests.post(
    "https://your-runpod-endpoint.com/rpc",
    json=api_request
)
```

---

## Required Fields in voice_metadata

**Minimum (must have):**
```json
{
  "voice_id": "a1b2c3d4",
  "firebase_storage_path": "voices/embeddings/a1b2c3d4.pkl.gz",
  "firebase_storage_bucket": "aitts-d4c6d.firebasestorage.app"
}
```

**Recommended (should have):**
```json
{
  "voice_id": "a1b2c3d4",
  "speaker_name": "UserVoice",
  "embedding_file": "embeddings/a1b2c3d4.pkl.gz",
  "sample_rate": 24000,
  "type": "chattts_embedding",
  "firebase_storage_path": "voices/embeddings/a1b2c3d4.pkl.gz",
  "firebase_storage_bucket": "aitts-d4c6d.firebasestorage.app"
}
```

---

## Complete Example (Copy-Paste Ready)

```python
from generate_voice_embedding import create_voice_embedding
from firebase_admin import storage
from google.cloud import firestore
import json
import requests

# 1. Create embedding
result = create_voice_embedding.from_file("voice.wav", "MyVoice")
voice_id = result['voice_id']

# 2. Upload to Firebase
bucket = storage.bucket()
firebase_path = f"voices/embeddings/{voice_id}.pkl.gz"
blob = bucket.blob(firebase_path)
blob.upload_from_filename(f"temp_embeddings/embeddings/{voice_id}.pkl.gz")
blob.make_public()

# 3. Prepare metadata
with open(f"temp_embeddings/metadata/{voice_id}.json", 'r') as f:
    voice_metadata = json.load(f)
voice_metadata['firebase_storage_path'] = firebase_path
voice_metadata['firebase_storage_bucket'] = "aitts-d4c6d.firebasestorage.app"

# 4. Save to database
db = firestore.Client()
db.collection('voices').document(voice_id).set(voice_metadata)

# 5. Use with handler
response = requests.post(
    "https://your-endpoint.com/rpc",
    json={
        "operation": "tts",
        "text": "Hello!",
        "voice_id": voice_id,
        "voice_metadata": voice_metadata
    }
)
```

---

## That's It! 🎉

The handler will:
1. Receive `voice_metadata` with Firebase paths
2. Add voice to catalog automatically
3. Download embedding from Firebase Storage on first use
4. Cache in memory for fast subsequent requests
5. Generate TTS with your voice!

See `EXTERNAL_APP_INTEGRATION_GUIDE.md` for detailed documentation.

