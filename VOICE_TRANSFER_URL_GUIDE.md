# Voice Transfer with File URLs - Fast & Simple

## Overview

This guide shows how to use voice transfer with **file URLs** instead of base64 encoding. This is much faster because you don't need to encode/decode large audio files!

## Benefits of Using URLs

✅ **No base64 encoding** - Just provide a URL to your audio file  
✅ **Much faster upload** - No need to send large encoded data  
✅ **Simpler code** - Just pass URLs instead of encoding files  
✅ **Works with any hosted audio** - Dropbox, Google Drive, your server, etc.

## How to Use

### 1. Upload Your Audio Files

First, upload your audio files to a web server, cloud storage, or file hosting service:

- **Dropbox**: Upload file → Get share link
- **Google Drive**: Upload file → Get share link  
- **AWS S3**: Upload file → Get public URL
- **Your server**: Upload to web directory → Get URL

### 2. Use the API with URLs

#### Option A: Transfer to Voice Embedding (Fastest)

```python
import requests

def transfer_with_url(api_key, input_url, target_voice):
    """Transfer audio from URL to voice embedding"""
    
    url = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "embedding",
            "input_audio": input_url,  # Just the URL!
            "input_is_url": True,      # Tell API this is a URL
            "voice_name": target_voice,
            "no_watermark": True
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    return response.json()['id']

# Usage
job_id = transfer_with_url(
    api_key="YOUR_API_KEY",
    input_url="https://example.com/my_audio.wav",
    target_voice="Amy"
)
print(f"Job submitted: {job_id}")
```

#### Option B: Transfer Between Two Audio Files

```python
def transfer_audio_to_audio_with_urls(api_key, input_url, target_url):
    """Transfer between two audio URLs"""
    
    url = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "audio",
            "input_audio": input_url,      # Source audio URL
            "input_is_url": True,
            "target_audio": target_url,    # Target audio URL
            "target_is_url": True,
            "no_watermark": True
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    return response.json()['id']

# Usage
job_id = transfer_audio_to_audio_with_urls(
    api_key="YOUR_API_KEY",
    input_url="https://example.com/source.wav",
    target_url="https://example.com/target.wav"
)
print(f"Job submitted: {job_id}")
```

### 3. Check Job Status (Same as Before)

```python
def check_job_status(api_key, job_id):
    """Check if job is complete"""
    
    url = f"https://api.runpod.ai/v2/c2wmx1ln5ccp6c/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.get(url, headers=headers)
    return response.json()

# Poll until complete
import time

while True:
    result = check_job_status(api_key, job_id)
    
    if result['status'] == 'COMPLETED':
        # Save the result
        audio_b64 = result['output']['audio']
        with open('transferred_audio.wav', 'wb') as f:
            f.write(base64.b64decode(audio_b64))
        print("✅ Transfer complete!")
        break
    elif result['status'] == 'FAILED':
        print(f"❌ Failed: {result.get('error')}")
        break
    else:
        print(f"⏳ Status: {result['status']}")
        time.sleep(2)
```

## cURL Examples

### Transfer to Voice Embedding

```bash
curl -X POST "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "operation": "voice_transfer",
      "transfer_mode": "embedding",
      "input_audio": "https://example.com/audio.wav",
      "input_is_url": true,
      "voice_name": "Amy",
      "no_watermark": true
    }
  }'
```

### Transfer Between Audio Files

```bash
curl -X POST "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "operation": "voice_transfer",
      "transfer_mode": "audio",
      "input_audio": "https://example.com/source.wav",
      "input_is_url": true,
      "target_audio": "https://example.com/target.wav",
      "target_is_url": true,
      "no_watermark": true
    }
  }'
```

## URL Requirements

Your audio URLs must be:

✅ **Publicly accessible** - No authentication required  
✅ **Direct download links** - Clicking the URL downloads the file  
✅ **Audio format** - WAV, MP3, M4A, FLAC, OGG  
✅ **Reasonable size** - Under 50MB recommended  

## Example URL Sources

### Dropbox
1. Upload file to Dropbox
2. Right-click → Share → Copy link
3. Replace `?dl=0` with `?dl=1` for direct download

### Google Drive
1. Upload file to Google Drive
2. Right-click → Share → Copy link
3. Replace `/view` with `/uc?export=download`

### AWS S3
1. Upload to S3 bucket
2. Make file public
3. Use the public URL

## Complete Example

```python
import requests
import base64
import time

class FastVoiceTransfer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def transfer_from_url(self, input_url, target_voice="Amy"):
        """Fast transfer using URL"""
        
        payload = {
            "input": {
                "operation": "voice_transfer",
                "transfer_mode": "embedding",
                "input_audio": input_url,
                "input_is_url": True,
                "voice_name": target_voice,
                "no_watermark": True
            }
        }
        
        response = requests.post(f"{self.base_url}/run", json=payload, headers=self.headers)
        return response.json()['id']
    
    def wait_for_completion(self, job_id, timeout=300):
        """Wait for job to complete"""
        
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
        """Save the transferred audio"""
        audio_data = base64.b64decode(result['audio'])
        with open(output_file, 'wb') as f:
            f.write(audio_data)
        print(f"✅ Audio saved to: {output_file}")

# Usage
api_key = "YOUR_API_KEY"
transfer = FastVoiceTransfer(api_key)

try:
    # Submit job (instant - no encoding needed!)
    job_id = transfer.transfer_from_url(
        input_url="https://example.com/my_recording.wav",
        target_voice="Amy"
    )
    print(f"🚀 Job submitted: {job_id}")
    
    # Wait for completion
    result = transfer.wait_for_completion(job_id)
    
    # Save result
    transfer.save_result(result, "transferred_audio.wav")
    
    print(f"🎉 Transfer complete! Duration: {result['duration']:.2f}s")
    
except Exception as e:
    print(f"❌ Error: {e}")
```

## Speed Comparison

| Method | Upload Time | Processing Time | Total Time |
|--------|-------------|-----------------|------------|
| **Base64** | 10-30 seconds | 30-90 seconds | 40-120 seconds |
| **URL** | 1-2 seconds | 30-90 seconds | 31-92 seconds |

**Result**: URL method is **10-30x faster** for upload!

## Troubleshooting

### URL Not Accessible
- Make sure the URL is publicly accessible
- Test by opening the URL in a browser
- Check if the file requires authentication

### Invalid URL Format
- Use full URLs starting with `http://` or `https://`
- Ensure the URL points directly to an audio file

### File Too Large
- Compress your audio files before uploading
- Use shorter audio clips (5-30 seconds recommended)

---

This URL-based approach makes voice transfer much faster and easier to use!
