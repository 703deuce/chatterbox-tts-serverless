# Handler Debugging Guide - Voice Metadata Issue

## The Problem

The handler is receiving `voice_metadata` but still can't find the voice. The error is:
```
Voice ID '3ddef3d8' not found in optimized embeddings
```

## What We've Done

1. ✅ Updated `handler.py` to support `voice_metadata`
2. ✅ Added `add_voice_to_catalog()` method to `optimized_voice_library.py`
3. ✅ Added Firebase Storage download support
4. ✅ Added detailed DEBUG logging

## Next Steps

### 1. Deploy Updated Handler to RunPod

**IMPORTANT:** The RunPod serverless endpoint needs to be updated with the new code!

The handler code has been updated in GitHub, but RunPod needs to:
1. Pull the latest code from GitHub
2. Rebuild the Docker image
3. Deploy the new version

### 2. Check RunPod Logs

After deploying, the new DEBUG logs will show:
- Whether `voice_metadata` is being received
- What keys are in `voice_metadata`
- Whether the voice is being added to catalog
- Whether Firebase download is being attempted

Look for these log messages:
```
DEBUG: voice_metadata present: True/False
DEBUG: voice_metadata keys: [...]
DEBUG: voice_metadata voice_id: ...
DEBUG: voice_metadata firebase_storage_path: ...
Adding voice metadata to catalog dynamically
Voice {voice_id} added to catalog from metadata, now loading...
```

### 3. Verify voice_metadata Format

Make sure your external app is sending:

```json
{
  "operation": "tts",
  "mode": "streaming",
  "text": "...",
  "voice_id": "3ddef3d8",
  "voice_metadata": {
    "voice_id": "3ddef3d8",
    "speaker_name": "Ashley_Customer_Service",
    "firebase_storage_path": "voices/embeddings/3ddef3d8.pkl.gz",
    "firebase_storage_bucket": "aitts-d4c6d.firebasestorage.app",
    "embedding_file": "embeddings/3ddef3d8.pkl.gz",
    "sample_rate": 24000,
    "type": "chattts_embedding"
  }
}
```

### 4. Common Issues

**Issue 1: Handler not updated on RunPod**
- Solution: Rebuild and redeploy RunPod serverless endpoint

**Issue 2: voice_metadata not in correct format**
- Solution: Check DEBUG logs to see what's being received

**Issue 3: Firebase Storage path incorrect**
- Solution: Verify the file exists at the path in Firebase Storage
- Check that the file is publicly accessible

**Issue 4: voice_metadata not being passed through**
- Solution: Check that `voice_metadata` is in the `input` object, not at the root level

## Expected Flow (After Deployment)

1. Handler receives request with `voice_metadata`
2. DEBUG log: "voice_metadata present: True"
3. Handler adds voice to catalog: "Adding voice metadata to catalog dynamically"
4. Handler downloads from Firebase: "Downloading embedding from Firebase: ..."
5. Handler caches in memory: "Loaded voice {voice_id}: ..."
6. Handler uses for TTS: "Using voice from optimized embeddings: {voice_id} (DIRECT)"

## Testing

After deploying the updated handler:

1. Send a TTS request with `voice_metadata`
2. Check RunPod logs for DEBUG messages
3. Verify the voice is added to catalog
4. Verify Firebase download succeeds
5. Verify TTS generation works

## If Still Not Working

If the handler still can't find the voice after deployment:

1. **Check DEBUG logs** - They will show exactly what's happening
2. **Verify Firebase file exists** - Test the download URL directly
3. **Check voice_metadata format** - Ensure all required fields are present
4. **Verify handler is using updated code** - Check RunPod deployment logs

The DEBUG logs will tell us exactly where the process is failing!

