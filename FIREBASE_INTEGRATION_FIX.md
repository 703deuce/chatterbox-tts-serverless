# Firebase Integration Fix for RunPod Serverless

## The Problem

The handler couldn't find voices that were created externally because:
1. Voices weren't in the local catalog (ephemeral filesystem)
2. Handler needs `voice_metadata` to add voice to catalog dynamically
3. Handler then downloads embedding from Firebase Storage

## The Solution

**Yes, the handler CAN download embeddings from Firebase!** Here's how:

### How It Works Now

1. **External app sends `voice_metadata`** in API request
2. **Handler adds voice to catalog** automatically using `add_voice_to_catalog()`
3. **Handler downloads embedding** from Firebase Storage using `firebase_storage_path`
4. **Handler caches in memory** for subsequent requests

### Required Format

Your API request MUST include `voice_metadata`:

```json
{
  "operation": "tts",
  "text": "Hello world!",
  "voice_id": "19884951",
  "voice_metadata": {
    "voice_id": "19884951",
    "speaker_name": "UserVoice",
    "firebase_storage_path": "voices/embeddings/19884951.pkl.gz",
    "firebase_storage_bucket": "aitts-d4c6d.firebasestorage.app",
    "embedding_file": "embeddings/19884951.pkl.gz",
    "sample_rate": 24000,
    "type": "chattts_embedding"
  }
}
```

### Alternative: Direct Download URL

You can also provide a direct download URL:

```json
{
  "voice_metadata": {
    "voice_id": "19884951",
    "firebase_download_url": "https://firebasestorage.googleapis.com/v0/b/aitts-d4c6d.firebasestorage.app/o/voices%2Fembeddings%2F19884951.pkl.gz?alt=media"
  }
}
```

## Key Points

✅ **No need to add to catalog first** - Handler does it automatically when `voice_metadata` is provided  
✅ **Handler downloads from Firebase** - Uses `firebase_storage_path` or `firebase_download_url`  
✅ **Works with ephemeral filesystem** - Downloads on-demand, caches in memory  
✅ **No local files needed** - Everything comes from Firebase Storage  

## What Was Fixed

1. **Handler now adds voice to catalog first** before trying to load it
2. **Extracts voice_id from metadata** if not provided directly
3. **Better error handling** for Firebase downloads
4. **Supports both storage_path and download_url** formats

## Testing

Make sure your `voice_metadata` includes:
- ✅ `voice_id`
- ✅ `firebase_storage_path` OR `firebase_download_url`
- ✅ `firebase_storage_bucket` (if using storage_path)

The handler will:
1. Add voice to catalog
2. Download from Firebase
3. Use for TTS generation

