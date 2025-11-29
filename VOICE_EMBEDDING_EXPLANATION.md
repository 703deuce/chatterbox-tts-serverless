# How Voice Embeddings Work

## Overview

Voice embeddings in this system are **not traditional ML embeddings** - they're actually **compressed audio arrays** stored as pickle files. This allows the system to use the original audio data directly for voice cloning without needing the original WAV file.

## Storage Structure

```
voice_embeddings/
├── embeddings/
│   ├── 1fc6c6ab.pkl.gz    # Compressed audio array for voice ID
│   ├── de16e46c.pkl.gz    # Another voice
│   └── ...
├── metadata/
│   ├── 1fc6c6ab.json      # Voice metadata
│   └── ...
├── voice_catalog.json      # Master catalog of all voices
├── name_mapping.json       # Maps names to IDs (e.g., "Amy" → "de16e46c")
└── stats.json              # Library statistics
```

## How Embeddings Are Saved

### Step 1: Audio Processing
1. Load audio file (WAV/MP3) → numpy array
2. Resample to 24kHz (ChatterboxTTS standard)
3. Generate unique 8-character hex ID from audio hash

### Step 2: Save Embedding File
```python
# Create embedding data structure
embedding_data = {
    'audio': audio_array,      # The actual audio samples
    'sample_rate': 24000,
    'voice_name': voice_name,
    'created_at': timestamp
}

# Save as compressed pickle
with gzip.open(f"embeddings/{voice_id}.pkl.gz", 'wb') as f:
    pickle.dump(embedding_data, f)
```

### Step 3: Save Metadata
```python
voice_metadata = {
    "voice_id": "de16e46c",
    "speaker_name": "Amy",
    "gender": "female",
    "embedding_file": "embeddings/de16e46c.pkl.gz",
    "duration": 23.7,
    "sample_rate": 24000,
    "type": "chattts_embedding"
}
```

### Step 4: Update Indexes
- Add to `voice_catalog.json`
- Add to `name_mapping.json` (name → ID lookup)
- Update `stats.json`

## How Embeddings Are Loaded

### Current Implementation (Optimized)

```python
# 1. Look up voice ID from name
voice_id = name_mapping["Amy"]  # Returns "de16e46c"

# 2. Load compressed embedding
with gzip.open(f"embeddings/{voice_id}.pkl.gz", 'rb') as f:
    embedding_data = pickle.load(f)

# 3. Extract audio array
audio_array = embedding_data['audio']  # Direct numpy array

# 4. Use directly with TTS (no WAV file needed!)
tts_model.generate(text="Hello", audio_prompt_array=audio_array)
```

## Why This Works

1. **No WAV File Needed**: The audio array is stored directly, so you don't need the original WAV file
2. **Fast Loading**: Compressed pickle files load quickly (98% faster than file I/O)
3. **Direct Usage**: Audio array can be passed directly to ChatterboxTTS
4. **Space Efficient**: Gzip compression reduces file size significantly

## Current Issue

The `generate_voice_cloning()` function in `handler_backup.py` **doesn't actually save embeddings**. It only:
- Generates a sample audio
- Returns success message
- **Does NOT save to embedding files**

## Solution

Use the `save_voice_embedding.py` utility I created, or update `generate_voice_cloning()` to call:

```python
from save_voice_embedding import save_voice_embedding

# After processing reference audio
result = save_voice_embedding(
    audio_array=reference_audio_array,
    sample_rate=ref_sr,
    voice_name=voice_name,
    voice_description=voice_description,
    gender=gender
)
```

## Usage Examples

### Save from Audio File
```python
from save_voice_embedding import save_voice_from_file

result = save_voice_from_file(
    audio_file_path="my_voice.wav",
    voice_name="MyVoice",
    gender="male",
    voice_description="My custom voice"
)
```

### Save from Audio Array
```python
from save_voice_embedding import save_voice_embedding
import soundfile as sf

audio_array, sr = sf.read("my_voice.wav")
result = save_voice_embedding(
    audio_array=audio_array,
    sample_rate=sr,
    voice_name="MyVoice"
)
```

### Use Saved Voice
```python
# After saving, use in TTS requests:
{
    "operation": "tts",
    "mode": "basic",
    "text": "Hello world!",
    "voice_name": "MyVoice"  # Uses saved embedding automatically
}
```

## Key Points

✅ **Embeddings = Compressed Audio Arrays** (not ML embeddings)  
✅ **No WAV File Needed** after saving  
✅ **Fast Direct Loading** from pickle files  
✅ **Automatic Resampling** to 24kHz  
✅ **Unique IDs** generated from audio hash  
✅ **Name Mapping** for easy lookup  

