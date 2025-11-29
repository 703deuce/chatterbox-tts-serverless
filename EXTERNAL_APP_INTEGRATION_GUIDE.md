# External App Integration Guide

## Complete Guide for Integrating Voice Embeddings with RunPod Serverless Handler

This guide shows exactly what you need to do in your external app to create voice embeddings that work with the RunPod Serverless handler.

---

## Overview

**The Problem:**
- RunPod Serverless has an ephemeral filesystem (files don't persist)
- Embedding files need to be stored in Firebase Storage
- Handler downloads embeddings on-demand from Firebase Storage

**The Solution:**
1. External app creates embedding files using `generate_voice_embedding.py`
2. Upload embedding files to Firebase Storage
3. Store voice metadata (including Firebase paths) in your database
4. When calling the handler, include `voice_metadata` in the API request
5. Handler downloads embedding from Firebase Storage on-demand

---

## Step 1: Create Voice Embedding (External App)

### Install Dependencies

```bash
pip install librosa soundfile numpy
```

### Use the Standalone Script

```python
from generate_voice_embedding import create_voice_embedding

# Create voice embedding
result = create_voice_embedding.from_file(
    audio_file="user_voice.wav",
    voice_name="UserVoice",
    embeddings_dir="./temp_embeddings",  # Temporary local directory
    gender="female"
)

voice_id = result['voice_id']  # e.g., "a1b2c3d4"
```

This creates:
- `temp_embeddings/embeddings/{voice_id}.pkl.gz` - The embedding file
- `temp_embeddings/metadata/{voice_id}.json` - Voice metadata
- Updates `temp_embeddings/voice_catalog.json` and `name_mapping.json`

---

## Step 2: Upload to Firebase Storage

### Upload Embedding File

```python
import firebase_admin
from firebase_admin import storage
import os

# Initialize Firebase (if not already done)
if not firebase_admin._apps:
    cred = firebase_admin.credentials.Certificate("path/to/serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'aitts-d4c6d.firebasestorage.app'
    })

bucket = storage.bucket()

# Upload embedding file
embedding_file_path = f"temp_embeddings/embeddings/{voice_id}.pkl.gz"
firebase_storage_path = f"voices/embeddings/{voice_id}.pkl.gz"

blob = bucket.blob(firebase_storage_path)
blob.upload_from_filename(embedding_file_path)
blob.make_public()  # Make it publicly accessible

print(f"✅ Uploaded embedding to: {firebase_storage_path}")
```

### Upload Metadata File (Optional)

```python
# Upload metadata file
metadata_file_path = f"temp_embeddings/metadata/{voice_id}.json"
firebase_metadata_path = f"voices/metadata/{voice_id}.json"

blob = bucket.blob(firebase_metadata_path)
blob.upload_from_filename(metadata_file_path)
```

---

## Step 3: Prepare Voice Metadata for Handler

### Read and Enhance Metadata

```python
import json

# Read the metadata file created by generate_voice_embedding.py
with open(f"temp_embeddings/metadata/{voice_id}.json", 'r') as f:
    voice_metadata = json.load(f)

# Add Firebase Storage paths (REQUIRED for RunPod Serverless)
voice_metadata['firebase_storage_path'] = f"voices/embeddings/{voice_id}.pkl.gz"
voice_metadata['firebase_storage_bucket'] = "aitts-d4c6d.firebasestorage.app"

# Save enhanced metadata to your database
# Example with Firestore:
from google.cloud import firestore

db = firestore.Client()
doc_ref = db.collection('voices').document(voice_id)
doc_ref.set(voice_metadata)

print(f"✅ Saved voice metadata to database: {voice_id}")
```

### Complete Metadata Structure

```json
{
  "voice_id": "a1b2c3d4",
  "speaker_name": "UserVoice",
  "gender": "female",
  "original_file": "user_voice.wav",
  "embedding_file": "embeddings/a1b2c3d4.pkl.gz",
  "firebase_storage_path": "voices/embeddings/a1b2c3d4.pkl.gz",
  "firebase_storage_bucket": "aitts-d4c6d.firebasestorage.app",
  "duration": 23.5,
  "sample_rate": 24000,
  "created_at": "2024-01-15T10:30:00.123456",
  "type": "chattts_embedding"
}
```

---

## Step 4: Call Handler API with Voice Metadata

### Option A: Using voice_id with voice_metadata

```python
import requests

# Get voice metadata from your database
voice_metadata = db.collection('voices').document(voice_id).get().to_dict()

# Call handler API
api_request = {
    "operation": "tts",
    "text": "Hello! This is my cloned voice.",
    "voice_id": voice_id,
    "voice_metadata": voice_metadata  # NEW: Include metadata so handler knows Firebase path
}

response = requests.post(
    "https://your-runpod-endpoint.com/rpc",
    json=api_request,
    headers={"Authorization": f"Bearer {api_key}"}
)

result = response.json()
audio_base64 = result['audio']
```

### Option B: Using voice_name with voice_metadata

```python
# If you know the voice_name, you still need to provide metadata
api_request = {
    "operation": "tts",
    "text": "Hello! This is my cloned voice.",
    "voice_name": "UserVoice",
    "voice_metadata": voice_metadata  # Handler uses this to find Firebase path
}
```

### Option C: Just voice_metadata (Handler adds to catalog)

```python
# Handler will add voice to catalog automatically
api_request = {
    "operation": "tts",
    "text": "Hello! This is my cloned voice.",
    "voice_metadata": voice_metadata  # Handler extracts voice_id from metadata
}
```

---

## Complete Example: End-to-End Flow

```python
"""
Complete example: Create voice embedding and use with handler
"""

from generate_voice_embedding import create_voice_embedding
import firebase_admin
from firebase_admin import storage
from google.cloud import firestore
import requests
import os
import json

# Step 1: Create embedding
print("Step 1: Creating voice embedding...")
result = create_voice_embedding.from_file(
    audio_file="user_voice.wav",
    voice_name="UserVoice",
    embeddings_dir="./temp_embeddings",
    gender="female"
)

voice_id = result['voice_id']
print(f"✅ Created voice: {voice_id}")

# Step 2: Upload to Firebase Storage
print("Step 2: Uploading to Firebase Storage...")
bucket = storage.bucket()

# Upload embedding
embedding_file = f"temp_embeddings/embeddings/{voice_id}.pkl.gz"
firebase_path = f"voices/embeddings/{voice_id}.pkl.gz"

blob = bucket.blob(firebase_path)
blob.upload_from_filename(embedding_file)
blob.make_public()
print(f"✅ Uploaded to: {firebase_path}")

# Step 3: Prepare metadata
print("Step 3: Preparing metadata...")
with open(f"temp_embeddings/metadata/{voice_id}.json", 'r') as f:
    voice_metadata = json.load(f)

# Add Firebase paths
voice_metadata['firebase_storage_path'] = firebase_path
voice_metadata['firebase_storage_bucket'] = "aitts-d4c6d.firebasestorage.app"

# Save to database
db = firestore.Client()
db.collection('voices').document(voice_id).set(voice_metadata)
print(f"✅ Saved metadata to database")

# Step 4: Use with handler
print("Step 4: Calling handler API...")
api_request = {
    "operation": "tts",
    "text": "Hello! This is my cloned voice speaking.",
    "voice_id": voice_id,
    "voice_metadata": voice_metadata
}

response = requests.post(
    "https://your-runpod-endpoint.com/rpc",
    json=api_request,
    headers={"Authorization": f"Bearer {api_key}"}
)

if response.status_code == 200:
    result = response.json()
    print("✅ TTS generated successfully!")
    # Save audio
    audio_base64 = result['audio']
    # ... decode and save audio ...
else:
    print(f"❌ Error: {response.status_code} - {response.text}")

# Cleanup temp files
import shutil
shutil.rmtree("temp_embeddings")
print("✅ Cleaned up temporary files")
```

---

## API Request Format

### Minimal Request (with voice_metadata)

```json
{
  "operation": "tts",
  "text": "Hello world!",
  "voice_id": "a1b2c3d4",
  "voice_metadata": {
    "voice_id": "a1b2c3d4",
    "speaker_name": "UserVoice",
    "firebase_storage_path": "voices/embeddings/a1b2c3d4.pkl.gz",
    "firebase_storage_bucket": "aitts-d4c6d.firebasestorage.app",
    "embedding_file": "embeddings/a1b2c3d4.pkl.gz",
    "sample_rate": 24000,
    "type": "chattts_embedding"
  }
}
```

### Full Request (all fields)

```json
{
  "operation": "tts",
  "text": "Hello world!",
  "voice_id": "a1b2c3d4",
  "voice_metadata": {
    "voice_id": "a1b2c3d4",
    "speaker_name": "UserVoice",
    "gender": "female",
    "original_file": "user_voice.wav",
    "embedding_file": "embeddings/a1b2c3d4.pkl.gz",
    "firebase_storage_path": "voices/embeddings/a1b2c3d4.pkl.gz",
    "firebase_storage_bucket": "aitts-d4c6d.firebasestorage.app",
    "duration": 23.5,
    "sample_rate": 24000,
    "created_at": "2024-01-15T10:30:00.123456",
    "type": "chattts_embedding",
    "description": "User's cloned voice"
  }
}
```

---

## Firebase Storage Structure

```
firebase-storage/
└── voices/
    ├── embeddings/
    │   ├── a1b2c3d4.pkl.gz
    │   ├── e5f6g7h8.pkl.gz
    │   └── ...
    └── metadata/  (optional)
        ├── a1b2c3d4.json
        └── ...
```

---

## Required Fields in voice_metadata

**Minimum required:**
- `voice_id` - Unique voice identifier
- `firebase_storage_path` - Path in Firebase Storage (e.g., `"voices/embeddings/{voice_id}.pkl.gz"`)
- `firebase_storage_bucket` - Firebase bucket name (e.g., `"aitts-d4c6d.firebasestorage.app"`)

**Recommended:**
- `speaker_name` - Voice name for lookup
- `embedding_file` - Relative path (for reference)
- `sample_rate` - Audio sample rate (usually 24000)
- `type` - Voice type (e.g., `"chattts_embedding"`)

**Optional:**
- `gender` - Voice gender
- `duration` - Audio duration in seconds
- `original_file` - Original audio filename
- `description` - Voice description
- `created_at` - Creation timestamp

---

## How It Works

1. **External App:**
   - Creates embedding using `generate_voice_embedding.py`
   - Uploads `.pkl.gz` file to Firebase Storage
   - Stores metadata (with Firebase paths) in database

2. **Handler (RunPod Serverless):**
   - Receives API request with `voice_id` and `voice_metadata`
   - Adds voice to catalog if not already present
   - Checks local filesystem (for built-in voices)
   - If not found, downloads from Firebase Storage using `firebase_storage_path`
   - Caches in memory for subsequent requests
   - Uses audio array for TTS generation

3. **Benefits:**
   - ✅ Works with ephemeral filesystem
   - ✅ No need to rebuild Docker image
   - ✅ Voices can be added dynamically
   - ✅ Fast (in-memory cache after first load)

---

## Error Handling

### Voice Not Found

If handler can't find voice:

```json
{
  "error": "Voice ID 'a1b2c3d4' not found. Ensure voice_metadata is provided or voice exists in catalog."
}
```

**Solution:** Make sure `voice_metadata` is included in API request with `firebase_storage_path`.

### Firebase Download Failed

If Firebase download fails:

```json
{
  "error": "Failed to download embedding from Firebase: a1b2c3d4"
}
```

**Solution:** 
- Check Firebase Storage path is correct
- Ensure file is publicly accessible
- Verify bucket name is correct

---

## Testing

### Test Voice Creation

```python
# Test creating embedding
result = create_voice_embedding.from_file("test.wav", "TestVoice")
assert result['success'] == True
assert 'voice_id' in result
print("✅ Voice creation works!")
```

### Test Firebase Upload

```python
# Test Firebase upload
blob = bucket.blob(f"voices/embeddings/{voice_id}.pkl.gz")
blob.upload_from_filename(f"temp_embeddings/embeddings/{voice_id}.pkl.gz")
blob.make_public()

# Verify it's accessible
url = blob.public_url
response = requests.head(url)
assert response.status_code == 200
print("✅ Firebase upload works!")
```

### Test Handler API

```python
# Test handler API
response = requests.post(
    "https://your-endpoint.com/rpc",
    json={
        "operation": "tts",
        "text": "Test",
        "voice_id": voice_id,
        "voice_metadata": voice_metadata
    }
)
assert response.status_code == 200
assert 'audio' in response.json()
print("✅ Handler API works!")
```

---

## Summary Checklist

- [ ] Install dependencies: `librosa`, `soundfile`, `numpy`
- [ ] Copy `generate_voice_embedding.py` to your app
- [ ] Create embedding using `create_voice_embedding.from_file()`
- [ ] Upload `.pkl.gz` file to Firebase Storage
- [ ] Add `firebase_storage_path` and `firebase_storage_bucket` to metadata
- [ ] Store metadata in your database
- [ ] Include `voice_metadata` in API requests to handler
- [ ] Handler automatically downloads from Firebase on first use
- [ ] Subsequent requests use in-memory cache (fast!)

---

## Quick Reference

**Create embedding:**
```python
from generate_voice_embedding import create_voice_embedding
result = create_voice_embedding.from_file("voice.wav", "VoiceName")
```

**Upload to Firebase:**
```python
blob = bucket.blob(f"voices/embeddings/{voice_id}.pkl.gz")
blob.upload_from_filename("embeddings/{voice_id}.pkl.gz")
```

**Call handler:**
```python
{
  "voice_id": voice_id,
  "voice_metadata": {
    "voice_id": voice_id,
    "firebase_storage_path": "voices/embeddings/{voice_id}.pkl.gz",
    "firebase_storage_bucket": "aitts-d4c6d.firebasestorage.app"
  }
}
```

That's it! Your external app and RunPod Serverless handler now work together seamlessly! 🎉

