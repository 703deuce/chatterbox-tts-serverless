# Voice Transfer Endpoint Usage Guide

## Overview

This guide provides detailed instructions for integrating the Voice Transfer endpoint into your applications. The Voice Transfer API allows you to transfer the voice characteristics from one audio file to another, with two powerful modes:

1. **Embedding Mode**: Transfer any audio to sound like one of the 37 pre-trained voice embeddings
2. **Audio Mode**: Transfer any audio to sound like any target audio file

## API Endpoint Details

**Base URL**: `https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run`  
**Authentication**: Bearer token required  
**Content-Type**: `application/json`

## Authentication

All requests require a Bearer token in the Authorization header:

```bash
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

## Voice Transfer Modes

### Mode 1: WAV to Voice Embedding
Transfer any input audio to sound like one of the 37 available voice embeddings.

**Required Parameters:**
- `operation`: `"voice_transfer"`
- `transfer_mode`: `"embedding"`
- `input_audio`: Base64-encoded input audio
- `voice_name`: Target voice name from the library

**Optional Parameters:**
- `voice_id`: Alternative to voice_name (8-character hex ID)
- `no_watermark`: Skip watermarking (default: false)

### Mode 2: WAV to WAV
Transfer any input audio to sound like any target audio file.

**Required Parameters:**
- `operation`: `"voice_transfer"`
- `transfer_mode`: `"audio"`
- `input_audio`: Base64-encoded input audio
- `target_audio`: Base64-encoded target audio

**Optional Parameters:**
- `no_watermark`: Skip watermarking (default: false)

## Available Voices

### Male Voices (21)
Aaron, Adam, Adrian, Alan, Albert, Alex, Alexander, Andrew, Anthony, Arthur, Austin, Benjamin, Blake, Brandon, Brian, Bruce, Bryan, Carl, Charles, Christian, Christopher

### Female Voices (16)
Abigail, Amanda, Amy, Angela, Anna, Ashley, Barbara, Betty, Beverly, Brenda, Brittany, Carol, Carolyn, Catherine, Christina, Christine

## Complete Integration Examples

### Python Integration

#### Basic Voice Transfer to Embedding

```python
import requests
import base64
import time
import json

class VoiceTransferAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def encode_audio_file(self, file_path):
        """Encode audio file to base64"""
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def submit_voice_transfer_job(self, input_audio_path, target_voice_name, no_watermark=False):
        """Submit a voice transfer job (embedding mode)"""
        
        # Encode input audio
        input_audio_b64 = self.encode_audio_file(input_audio_path)
        
        payload = {
            "input": {
                "operation": "voice_transfer",
                "transfer_mode": "embedding",
                "input_audio": input_audio_b64,
                "voice_name": target_voice_name,
                "no_watermark": no_watermark
            }
        }
        
        # Submit job
        response = requests.post(f"{self.base_url}/run", json=payload, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Job submission failed: {response.status_code} - {response.text}")
        
        result = response.json()
        return result['id']  # Return job ID
    
    def submit_audio_to_audio_job(self, input_audio_path, target_audio_path, no_watermark=False):
        """Submit a voice transfer job (audio mode)"""
        
        # Encode both audio files
        input_audio_b64 = self.encode_audio_file(input_audio_path)
        target_audio_b64 = self.encode_audio_file(target_audio_path)
        
        payload = {
            "input": {
                "operation": "voice_transfer",
                "transfer_mode": "audio",
                "input_audio": input_audio_b64,
                "target_audio": target_audio_b64,
                "no_watermark": no_watermark
            }
        }
        
        # Submit job
        response = requests.post(f"{self.base_url}/run", json=payload, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Job submission failed: {response.status_code} - {response.text}")
        
        result = response.json()
        return result['id']  # Return job ID
    
    def poll_job_status(self, job_id, max_wait_time=300, poll_interval=2):
        """Poll job status until completion or timeout"""
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Get job status
            status_url = f"{self.base_url}/status/{job_id}"
            response = requests.get(status_url, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"Status check failed: {response.status_code} - {response.text}")
            
            result = response.json()
            status = result.get('status')
            
            print(f"Job {job_id}: {status}")
            
            if status == 'COMPLETED':
                return result['output']
            elif status == 'FAILED':
                error_msg = result.get('error', 'Unknown error')
                raise Exception(f"Job failed: {error_msg}")
            elif status == 'IN_QUEUE':
                print("Job is queued, waiting...")
            elif status == 'IN_PROGRESS':
                print("Job is in progress...")
            
            time.sleep(poll_interval)
        
        raise Exception(f"Job timed out after {max_wait_time} seconds")
    
    def save_audio_response(self, audio_b64, output_path):
        """Save base64 audio response to file"""
        audio_data = base64.b64decode(audio_b64)
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        print(f"Audio saved to: {output_path}")

# Usage Example
def main():
    # Initialize API client
    api_key = "YOUR_API_KEY_HERE"
    voice_transfer = VoiceTransferAPI(api_key)
    
    try:
        # Example 1: Transfer to voice embedding
        print("=== Voice Transfer to Embedding ===")
        job_id = voice_transfer.submit_voice_transfer_job(
            input_audio_path="my_recording.wav",
            target_voice_name="Amy",
            no_watermark=True
        )
        
        print(f"Job submitted with ID: {job_id}")
        
        # Wait for completion
        result = voice_transfer.poll_job_status(job_id)
        
        # Save the result
        output_file = f"transferred_to_amy.wav"
        voice_transfer.save_audio_response(result['audio'], output_file)
        
        # Print transfer info
        transfer_info = result.get('transfer_info', {})
        print(f"Transfer completed!")
        print(f"  Mode: {transfer_info.get('transfer_mode')}")
        print(f"  Target Voice: {transfer_info.get('target_voice')}")
        print(f"  Duration: {result.get('duration')} seconds")
        print(f"  Sample Rate: {result.get('sample_rate')} Hz")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

#### Audio-to-Audio Transfer Example

```python
def audio_to_audio_transfer_example():
    """Example of transferring audio to sound like another audio file"""
    
    api_key = "YOUR_API_KEY_HERE"
    voice_transfer = VoiceTransferAPI(api_key)
    
    try:
        print("=== Audio to Audio Transfer ===")
        job_id = voice_transfer.submit_audio_to_audio_job(
            input_audio_path="source_recording.wav",
            target_audio_path="target_voice.wav",
            no_watermark=True
        )
        
        print(f"Job submitted with ID: {job_id}")
        
        # Wait for completion
        result = voice_transfer.poll_job_status(job_id)
        
        # Save the result
        output_file = "transferred_audio.wav"
        voice_transfer.save_audio_response(result['audio'], output_file)
        
        # Print transfer info
        transfer_info = result.get('transfer_info', {})
        print(f"Transfer completed!")
        print(f"  Mode: {transfer_info.get('transfer_mode')}")
        print(f"  Target Duration: {transfer_info.get('target_duration')} seconds")
        print(f"  Output Duration: {result.get('duration')} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
```

### JavaScript/Node.js Integration

```javascript
const axios = require('axios');
const fs = require('fs');

class VoiceTransferAPI {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.baseUrl = 'https://api.runpod.ai/v2/c2wmx1ln5ccp6c';
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }

    encodeAudioFile(filePath) {
        const audioBuffer = fs.readFileSync(filePath);
        return audioBuffer.toString('base64');
    }

    async submitVoiceTransferJob(inputAudioPath, targetVoiceName, noWatermark = false) {
        const inputAudioB64 = this.encodeAudioFile(inputAudioPath);
        
        const payload = {
            input: {
                operation: 'voice_transfer',
                transfer_mode: 'embedding',
                input_audio: inputAudioB64,
                voice_name: targetVoiceName,
                no_watermark: noWatermark
            }
        };

        try {
            const response = await axios.post(`${this.baseUrl}/run`, payload, { headers: this.headers });
            return response.data.id;
        } catch (error) {
            throw new Error(`Job submission failed: ${error.response?.status} - ${error.response?.data}`);
        }
    }

    async submitAudioToAudioJob(inputAudioPath, targetAudioPath, noWatermark = false) {
        const inputAudioB64 = this.encodeAudioFile(inputAudioPath);
        const targetAudioB64 = this.encodeAudioFile(targetAudioPath);
        
        const payload = {
            input: {
                operation: 'voice_transfer',
                transfer_mode: 'audio',
                input_audio: inputAudioB64,
                target_audio: targetAudioB64,
                no_watermark: noWatermark
            }
        };

        try {
            const response = await axios.post(`${this.baseUrl}/run`, payload, { headers: this.headers });
            return response.data.id;
        } catch (error) {
            throw new Error(`Job submission failed: ${error.response?.status} - ${error.response?.data}`);
        }
    }

    async pollJobStatus(jobId, maxWaitTime = 300000, pollInterval = 2000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWaitTime) {
            try {
                const response = await axios.get(`${this.baseUrl}/status/${jobId}`, { headers: this.headers });
                const result = response.data;
                const status = result.status;
                
                console.log(`Job ${jobId}: ${status}`);
                
                if (status === 'COMPLETED') {
                    return result.output;
                } else if (status === 'FAILED') {
                    throw new Error(`Job failed: ${result.error || 'Unknown error'}`);
                } else if (status === 'IN_QUEUE') {
                    console.log('Job is queued, waiting...');
                } else if (status === 'IN_PROGRESS') {
                    console.log('Job is in progress...');
                }
                
                await new Promise(resolve => setTimeout(resolve, pollInterval));
            } catch (error) {
                throw new Error(`Status check failed: ${error.message}`);
            }
        }
        
        throw new Error(`Job timed out after ${maxWaitTime / 1000} seconds`);
    }

    saveAudioResponse(audioB64, outputPath) {
        const audioBuffer = Buffer.from(audioB64, 'base64');
        fs.writeFileSync(outputPath, audioBuffer);
        console.log(`Audio saved to: ${outputPath}`);
    }
}

// Usage Example
async function main() {
    const apiKey = 'YOUR_API_KEY_HERE';
    const voiceTransfer = new VoiceTransferAPI(apiKey);
    
    try {
        console.log('=== Voice Transfer to Embedding ===');
        const jobId = await voiceTransfer.submitVoiceTransferJob(
            'my_recording.wav',
            'Amy',
            true
        );
        
        console.log(`Job submitted with ID: ${jobId}`);
        
        const result = await voiceTransfer.pollJobStatus(jobId);
        
        const outputFile = 'transferred_to_amy.wav';
        voiceTransfer.saveAudioResponse(result.audio, outputFile);
        
        console.log('Transfer completed!');
        console.log(`  Mode: ${result.transfer_info.transfer_mode}`);
        console.log(`  Target Voice: ${result.transfer_info.target_voice}`);
        console.log(`  Duration: ${result.duration} seconds`);
        
    } catch (error) {
        console.error(`Error: ${error.message}`);
    }
}

main();
```

### cURL Examples

#### Submit Voice Transfer Job (Embedding Mode)

```bash
curl -X POST "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "operation": "voice_transfer",
      "transfer_mode": "embedding",
      "input_audio": "BASE64_ENCODED_AUDIO_HERE",
      "voice_name": "Amy",
      "no_watermark": true
    }
  }'
```

#### Submit Voice Transfer Job (Audio Mode)

```bash
curl -X POST "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "operation": "voice_transfer",
      "transfer_mode": "audio",
      "input_audio": "BASE64_ENCODED_INPUT_AUDIO",
      "target_audio": "BASE64_ENCODED_TARGET_AUDIO",
      "no_watermark": true
    }
  }'
```

#### Check Job Status

```bash
curl -X GET "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/status/JOB_ID_HERE" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Response Format

### Successful Response (Embedding Mode)

```json
{
  "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
  "sample_rate": 24000,
  "duration": 5.32,
  "format": "wav",
  "model": "s3gen",
  "operation": "voice_transfer",
  "optimization": "direct_audio_array",
  "transfer_info": {
    "transfer_mode": "embedding",
    "target_voice": "Amy",
    "source": "voice_library",
    "voice_loading": "optimized_direct_access"
  },
  "input_duration": 4.8,
  "processing_time": "30-90 seconds typical"
}
```

### Successful Response (Audio Mode)

```json
{
  "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
  "sample_rate": 24000,
  "duration": 6.15,
  "format": "wav",
  "model": "s3gen",
  "operation": "voice_transfer",
  "transfer_info": {
    "transfer_mode": "audio",
    "target_source": "user_provided_audio",
    "target_duration": 8.2
  },
  "input_duration": 5.1,
  "processing_time": "30-90 seconds typical"
}
```

### Job Status Response

```json
{
  "id": "job_id_here",
  "status": "COMPLETED",
  "output": {
    // Response data as shown above
  }
}
```

### Error Response

```json
{
  "id": "job_id_here",
  "status": "FAILED",
  "error": "Error description here"
}
```

## Response Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `audio` | String | Base64-encoded transferred audio data (WAV format) |
| `sample_rate` | Integer | Output sample rate (always 24000 Hz) |
| `duration` | Float | Duration of transferred audio in seconds |
| `format` | String | Audio format ("wav") |
| `model` | String | AI model used ("s3gen") |
| `operation` | String | Operation performed ("voice_transfer") |
| `optimization` | String | Optimization method used (if applicable) |
| `transfer_info` | Object | Transfer details and metadata |
| `input_duration` | Float | Duration of input audio in seconds |
| `processing_time` | String | Typical processing time range |

### Transfer Info Fields

**For Embedding Mode:**
- `transfer_mode`: "embedding"
- `target_voice`: Name of target voice from library
- `source`: "voice_library"
- `voice_loading`: Optimization method used

**For Audio Mode:**
- `transfer_mode`: "audio"
- `target_source`: "user_provided_audio"
- `target_duration`: Duration of target audio in seconds

## Error Handling

### Common Error Scenarios

1. **Invalid API Key**
   ```json
   {
     "error": "Invalid API key"
   }
   ```

2. **Invalid Parameters**
   ```json
   {
     "error": "Missing required parameter: voice_name"
   }
   ```

3. **Invalid Audio Format**
   ```json
   {
     "error": "Invalid audio format or corrupted audio data"
   }
   ```

4. **Audio Too Long**
   ```json
   {
     "error": "Input audio exceeds maximum duration limit"
   }
   ```

5. **Voice Not Found**
   ```json
   {
     "error": "Voice 'InvalidVoice' not found in library"
   }
   ```

### Error Handling Best Practices

```python
def handle_voice_transfer_with_retry(api_client, input_audio, target_voice, max_retries=3):
    """Handle voice transfer with retry logic and error handling"""
    
    for attempt in range(max_retries):
        try:
            # Submit job
            job_id = api_client.submit_voice_transfer_job(input_audio, target_voice)
            
            # Poll for completion
            result = api_client.poll_job_status(job_id)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            if "Invalid API key" in error_msg:
                raise Exception("Authentication failed - check your API key")
            elif "Voice not found" in error_msg:
                raise Exception(f"Voice '{target_voice}' not available")
            elif "Invalid audio format" in error_msg:
                raise Exception("Audio file is corrupted or in unsupported format")
            elif "exceeds maximum duration" in error_msg:
                raise Exception("Audio file is too long (max 60 seconds recommended)")
            elif attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {error_msg}")
                print("Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise Exception(f"All retry attempts failed: {error_msg}")
```

## Performance Guidelines

### Processing Times
- **Typical processing**: 30-90 seconds
- **Short audio (5-15 seconds)**: 30-60 seconds
- **Long audio (30-60 seconds)**: 60-120 seconds
- **Queue time**: 0-30 seconds (depending on server load)

### Audio Requirements
- **Input formats**: WAV, MP3, M4A, FLAC, OGG
- **Output format**: Always 24kHz WAV
- **Recommended input duration**: 5-60 seconds
- **Maximum input duration**: 60 seconds
- **Target audio duration**: 5-30 seconds (for audio mode)

### Optimization Tips
1. **Use appropriate audio lengths**: 10-30 seconds for best results
2. **Ensure good audio quality**: Clear speech, minimal background noise
3. **Use voice embeddings when possible**: Faster than audio-to-audio transfer
4. **Implement proper retry logic**: Handle temporary failures gracefully
5. **Cache results**: Avoid re-processing identical requests

## Rate Limiting and Quotas

- **Concurrent jobs**: Limited based on your API plan
- **Request frequency**: No strict rate limiting, but be reasonable
- **Job timeout**: 300 seconds (5 minutes) maximum
- **Queue management**: Jobs are processed in FIFO order

## Best Practices

### 1. Input Audio Preparation
```python
def prepare_audio_for_transfer(audio_path):
    """Prepare audio file for optimal transfer results"""
    
    # Check file size and duration
    import wave
    with wave.open(audio_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        duration = frames / sample_rate
    
    if duration > 60:
        print("Warning: Audio is longer than 60 seconds, may affect quality")
    
    if duration < 2:
        print("Warning: Audio is very short, may not transfer well")
    
    return {
        'duration': duration,
        'sample_rate': sample_rate,
        'frames': frames
    }
```

### 2. Voice Selection
```python
def get_available_voices(api_client):
    """Get list of available voices for transfer"""
    try:
        # This would use the list_local_voices endpoint
        voices = api_client.list_voices()
        return voices
    except Exception as e:
        print(f"Could not fetch voice list: {e}")
        return []
```

### 3. Batch Processing
```python
def batch_voice_transfer(api_client, input_files, target_voice):
    """Process multiple files with the same target voice"""
    
    results = []
    for input_file in input_files:
        try:
            job_id = api_client.submit_voice_transfer_job(input_file, target_voice)
            result = api_client.poll_job_status(job_id)
            results.append({
                'input_file': input_file,
                'output': result,
                'success': True
            })
        except Exception as e:
            results.append({
                'input_file': input_file,
                'error': str(e),
                'success': False
            })
    
    return results
```

## Troubleshooting

### Common Issues and Solutions

1. **Job Stuck in Queue**
   - Check server status
   - Wait for queue to clear
   - Consider upgrading API plan

2. **Poor Transfer Quality**
   - Ensure input audio is clear and high quality
   - Try different target voices
   - Check audio duration (5-30 seconds optimal)

3. **Authentication Errors**
   - Verify API key is correct
   - Check API key permissions
   - Ensure proper Authorization header format

4. **Timeout Errors**
   - Increase polling timeout
   - Check network connectivity
   - Verify job ID is correct

5. **Audio Format Issues**
   - Convert audio to WAV format
   - Ensure audio is not corrupted
   - Check file permissions

## Support and Resources

- **API Documentation**: See main TTS_API_DOCUMENTATION.md
- **Voice Library**: Use `list_local_voices` endpoint to see available voices
- **Testing**: Use the provided test scripts for validation
- **Performance**: Monitor processing times and optimize accordingly

---

This guide provides everything you need to integrate voice transfer functionality into your applications. The API is designed to be reliable, fast, and easy to use with proper error handling and comprehensive response data.
