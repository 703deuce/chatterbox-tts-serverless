# Firebase Storage Integration for Voice Transfer

## Overview

The voice transfer endpoint now supports Firebase Storage paths in addition to URLs and base64 encoding. This allows you to reference audio files stored in Firebase Storage without having to download and re-upload them.

## New Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `input_storage_path` | String | Firebase Storage path for input audio | `"users/tts/user123/input.wav"` |
| `target_storage_path` | String | Firebase Storage path for target audio | `"users/tts/user123/target.wav"` |
| `storage_bucket` | String | Firebase storage bucket name | `"aitts-d4c6d.firebasestorage.app"` |

## Usage Examples

### 1. Firebase Path to Voice Embedding (Fastest)

```python
import requests

def transfer_firebase_to_voice(api_key, input_path, target_voice):
    """Transfer audio from Firebase Storage to voice embedding"""
    
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
            "input_storage_path": input_path,  # Firebase path
            "storage_bucket": "aitts-d4c6d.firebasestorage.app",
            "voice_name": target_voice,
            "no_watermark": True
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    return response.json()['id']

# Usage
job_id = transfer_firebase_to_voice(
    api_key="YOUR_API_KEY",
    input_path="users/tts/user123/recording.wav",
    target_voice="Amy"
)
```

### 2. Firebase Path to Firebase Path (Audio Mode)

```python
def transfer_firebase_to_firebase(api_key, input_path, target_path):
    """Transfer between two Firebase Storage audio files"""
    
    url = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "audio",
            "input_audio": "",  # Not used
            "input_storage_path": input_path,
            "target_audio": "",  # Not used
            "target_storage_path": target_path,
            "storage_bucket": "aitts-d4c6d.firebasestorage.app",
            "no_watermark": True
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    return response.json()['id']

# Usage
job_id = transfer_firebase_to_firebase(
    api_key="YOUR_API_KEY",
    input_path="users/tts/user123/source.wav",
    target_path="users/tts/user123/target_voice.wav"
)
```

### 3. Mixed Mode (Firebase + Regular URL)

```python
def transfer_firebase_to_url(api_key, input_path, target_url):
    """Transfer Firebase audio to URL audio"""
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "audio",
            "input_audio": "",
            "input_storage_path": input_path,  # Firebase path
            "target_audio": target_url,         # Regular URL
            "target_is_url": True,
            "storage_bucket": "aitts-d4c6d.firebasestorage.app",
            "no_watermark": True
        }
    }
    
    # Submit job...
```

## Firebase Storage Path Format

Your Firebase Storage paths should follow this format:

```
{fileType}/{userId}/{filename}
```

### Examples:
- `users/tts/user123/tts_1234567890_audio.wav`
- `transcription_uploads/stt/transcription_123.wav`
- `voice_samples/user456/sample.wav`

## Complete Integration Example

```python
import requests
import time
import json

class FirebaseVoiceTransfer:
    def __init__(self, api_key, storage_bucket="aitts-d4c6d.firebasestorage.app"):
        self.api_key = api_key
        self.storage_bucket = storage_bucket
        self.base_url = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def transfer_to_voice(self, input_path, target_voice):
        """Transfer Firebase audio to voice embedding"""
        
        payload = {
            "input": {
                "operation": "voice_transfer",
                "transfer_mode": "embedding",
                "input_audio": "",
                "input_storage_path": input_path,
                "storage_bucket": self.storage_bucket,
                "voice_name": target_voice,
                "no_watermark": True
            }
        }
        
        response = requests.post(f"{self.base_url}/run", json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()['id']
    
    def transfer_audio_to_audio(self, input_path, target_path):
        """Transfer between Firebase audio files"""
        
        payload = {
            "input": {
                "operation": "voice_transfer",
                "transfer_mode": "audio",
                "input_audio": "",
                "input_storage_path": input_path,
                "target_audio": "",
                "target_storage_path": target_path,
                "storage_bucket": self.storage_bucket,
                "no_watermark": True
            }
        }
        
        response = requests.post(f"{self.base_url}/run", json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()['id']
    
    def wait_for_completion(self, job_id, timeout=300):
        """Wait for job completion"""
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = requests.get(f"{self.base_url}/status/{job_id}", headers=self.headers)
            result = response.json()
            
            if result['status'] == 'COMPLETED':
                return result['output']
            elif result['status'] == 'FAILED':
                raise Exception(f"Job failed: {result.get('error')}")
            
            time.sleep(2)
        
        raise Exception("Job timed out")
    
    def save_result(self, result, output_file):
        """Save transferred audio to file"""
        import base64
        
        audio_data = base64.b64decode(result['audio'])
        with open(output_file, 'wb') as f:
            f.write(audio_data)
        
        print(f"✅ Audio saved to: {output_file}")
        return output_file

# Usage Example
def main():
    # Initialize transfer client
    client = FirebaseVoiceTransfer("YOUR_API_KEY")
    
    try:
        # Transfer Firebase audio to Amy voice
        job_id = client.transfer_to_voice(
            input_path="users/tts/user123/my_recording.wav",
            target_voice="Amy"
        )
        
        print(f"🚀 Job submitted: {job_id}")
        
        # Wait for completion
        result = client.wait_for_completion(job_id)
        
        # Save result
        output_file = client.save_result(result, "transferred_to_amy.wav")
        
        # Print details
        transfer_info = result.get('transfer_info', {})
        print(f"🎉 Transfer completed!")
        print(f"   Input source: {transfer_info.get('input_source')}")
        print(f"   Target voice: {transfer_info.get('target_voice')}")
        print(f"   Duration: {result.get('duration'):.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

## Response Format

The response includes information about the source types used:

```json
{
  "audio": "base64_encoded_audio",
  "sample_rate": 24000,
  "duration": 5.32,
  "format": "wav",
  "model": "s3gen",
  "operation": "voice_transfer",
  "transfer_info": {
    "transfer_mode": "embedding",
    "target_voice": "Amy",
    "source": "voice_library",
    "input_source": "firebase_path"
  },
  "input_duration": 4.8
}
```

### Transfer Info Fields

- `input_source`: `"firebase_path"`, `"url"`, or `"base64"`
- `target_source_type`: `"firebase_path"`, `"url"`, or `"base64"` (for audio mode)

## Benefits

✅ **No file downloads required** - Direct access to Firebase Storage  
✅ **Faster processing** - No base64 encoding/decoding  
✅ **Secure** - Uses Firebase Storage's built-in access controls  
✅ **Scalable** - Works with any Firebase Storage bucket  
✅ **Backward compatible** - Existing URL/base64 methods still work  

## Error Handling

Common errors and solutions:

### Firebase Access Denied
```json
{
  "error": "Failed to download/process audio from URL: 403 Forbidden"
}
```
**Solution**: Ensure your Firebase Storage files are publicly accessible or the bucket has proper permissions.

### Invalid Storage Path
```json
{
  "error": "Failed to download/process audio from URL: 404 Not Found"
}
```
**Solution**: Check that the storage path exists and the file is uploaded to Firebase Storage.

### Missing Parameters
```json
{
  "error": "input_audio or input_storage_path is required"
}
```
**Solution**: Provide either `input_audio` (for URL/base64) or `input_storage_path` (for Firebase).

## Migration from URLs

If you're currently using URLs, migrating to Firebase paths is simple:

### Before (URL method):
```python
payload = {
    "input": {
        "operation": "voice_transfer",
        "transfer_mode": "embedding",
        "input_audio": "https://firebasestorage.googleapis.com/v0/b/aitts-d4c6d.firebasestorage.app/o/users%2Ftts%2Fuser123%2Faudio.wav?alt=media",
        "input_is_url": True,
        "voice_name": "Amy"
    }
}
```

### After (Firebase path method):
```python
payload = {
    "input": {
        "operation": "voice_transfer",
        "transfer_mode": "embedding",
        "input_audio": "",  # Not needed
        "input_storage_path": "users/tts/user123/audio.wav",  # Much simpler!
        "storage_bucket": "aitts-d4c6d.firebasestorage.app",
        "voice_name": "Amy"
    }
}
```

---

This Firebase Storage integration makes voice transfer much easier when working with files already stored in Firebase Storage!
