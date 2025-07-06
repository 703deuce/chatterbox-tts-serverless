# Chatterbox TTS Runpod Serverless API

A comprehensive serverless API implementation for Chatterbox TTS with full parameter control, running on Runpod's GPU infrastructure.

## Features

- **Text-to-Speech (TTS)**: Generate high-quality speech from text with full parameter control
- **Voice Conversion (VC)**: Convert voice characteristics between audio samples  
- **Voice Cloning**: Clone voices using reference audio
- **Advanced Controls**: Emotion, speed, expressiveness, and audio processing
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs
- **Serverless Deployment**: Auto-scaling on Runpod infrastructure
- **Base64 Audio**: Easy integration with web applications

## Files

- `handler.py`: Main serverless handler with comprehensive parameter support
- `Dockerfile`: Container configuration for GPU deployment
- `requirements.txt`: Python dependencies
- `test_api.py`: Testing script with parameter examples
- `build_and_deploy.ps1`: PowerShell build script for Windows
- `build_and_deploy.sh`: Bash build script for Linux/Mac
- `README.md`: This comprehensive documentation

## Deployment Instructions

### 1. Build and Push Docker Image

For Windows (PowerShell):
```powershell
.\build_and_deploy.ps1 your-dockerhub-username
```

For Linux/Mac:
```bash
./build_and_deploy.sh your-dockerhub-username
```

### 2. Deploy on Runpod

1. Go to [Runpod Serverless](https://runpod.io/serverless)
2. Click "Create Endpoint"
3. Enter your Docker image: `your-username/chatterbox-tts-serverless`
4. Select GPU type (recommended: RTX 4090 or A100)
5. Configure settings:
   - Container Disk: 20GB minimum
   - Memory: 16GB minimum
   - Timeout: 300 seconds
6. Deploy the endpoint

### 3. Get Your API Endpoint

After deployment, you'll receive an API endpoint URL like:
```
https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
```

## Comprehensive Parameter Reference

### Core Synthesis Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `exaggeration` | float | 0.25-2.0 | 0.5 | Controls expressiveness/emotion intensity |
| `cfg_weight` | float | 0.2-1.0 | 0.5 | Prompt adherence vs. expressiveness |
| `temperature` | float | 0.05-5.0 | 0.8 | Speech speed and randomness |
| `speed_factor` | float | 0.1-3.0 | 1.0 | Overall speech speed multiplier |
| `seed` | int | any | null | Random seed for reproducibility |

### Voice Parameters

| Parameter | Type | Options | Default | Description |
|-----------|------|---------|---------|-------------|
| `voice_mode` | string | "predefined", "clone" | "predefined" | Voice source type |
| `predefined_voice_id` | string | voice ID | null | Built-in voice identifier |
| `reference_audio` | string | base64 | null | Reference audio for cloning |
| `max_reference_duration_sec` | int | 1-60 | 30 | Max reference audio duration |

### Audio Output Parameters

| Parameter | Type | Options | Default | Description |
|-----------|------|---------|---------|-------------|
| `output_format` | string | "wav", "opus", "mp3" | "wav" | Output audio format |
| `sample_rate` | int | 16000, 22050, 24000, 44100, 48000 | 24000 | Output sample rate |
| `audio_normalization` | string | "peak", "rms" | null | Audio normalization method |

### Text Processing Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `split_text` | bool | true/false | false | Auto-chunk long text |
| `chunk_size` | int | 10-500 | 120 | Characters per chunk |
| `candidates_per_chunk` | int | 1-5 | 1 | Generation candidates per chunk |
| `retries` | int | 1-5 | 1 | Retry attempts for failures |
| `parallel_workers` | int | 1-8 | 1 | Parallel processing workers |

### Language and Post-Processing

| Parameter | Type | Options | Default | Description |
|-----------|------|---------|---------|-------------|
| `language` | string | "en", "es", "fr", etc. | "en" | Language code |
| `sound_word_remove_replace` | string | text | null | Post-process sound/word removal |
| `auto_editor_margin` | float | 0.0-1.0 | null | Audio smoothing margin |

## API Usage

### Authentication

Add your Runpod API key to the Authorization header:
```
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

### Basic Text-to-Speech

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "tts",
      "text": "Hello, this is a test of the Chatterbox TTS system!"
    }
  }'
```

### Advanced TTS with Full Parameters

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "tts",
      "text": "This is an expressive speech with custom parameters!",
      "exaggeration": 1.2,
      "cfg_weight": 0.3,
      "temperature": 1.5,
      "speed_factor": 0.9,
      "seed": 42,
      "voice_mode": "predefined",
      "output_format": "wav",
      "sample_rate": 44100,
      "audio_normalization": "peak",
      "language": "en",
      "split_text": true,
      "chunk_size": 100
    }
  }'
```

### Voice Cloning with Reference Audio

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "tts",
      "text": "This will sound like the reference voice!",
      "voice_mode": "clone",
      "reference_audio": "base64_encoded_reference_audio",
      "max_reference_duration_sec": 15,
      "exaggeration": 0.8,
      "cfg_weight": 0.4,
      "temperature": 0.9
    }
  }'
```

### Voice Conversion with Advanced Options

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "vc",
      "source_audio": "base64_encoded_source_audio",
      "target_audio": "base64_encoded_target_audio",
      "max_source_duration_sec": 30,
      "max_target_duration_sec": 20,
      "output_format": "wav",
      "sample_rate": 48000,
      "audio_normalization": "rms"
    }
  }'
```

## Python Client Example

```python
import requests
import base64
import json

# Configuration
RUNPOD_API_KEY = "your_runpod_api_key"
ENDPOINT_ID = "your_endpoint_id"
API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

def advanced_tts(
    text,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    speed_factor=1.0,
    seed=None,
    voice_mode="predefined",
    reference_audio_path=None,
    output_format="wav",
    sample_rate=24000,
    audio_normalization=None,
    language="en"
):
    """Generate TTS with advanced parameters"""
    
    payload = {
        "input": {
            "task": "tts",
            "text": text,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "speed_factor": speed_factor,
            "voice_mode": voice_mode,
            "output_format": output_format,
            "sample_rate": sample_rate,
            "language": language
        }
    }
    
    # Add optional parameters
    if seed is not None:
        payload["input"]["seed"] = seed
    if audio_normalization:
        payload["input"]["audio_normalization"] = audio_normalization
    
    # Handle voice cloning
    if voice_mode == "clone" and reference_audio_path:
        with open(reference_audio_path, "rb") as f:
            reference_b64 = base64.b64encode(f.read()).decode()
        payload["input"]["reference_audio"] = reference_b64
    
    response = requests.post(API_URL, json=payload, headers=headers)
    response.raise_for_status()
    
    result = response.json()
    
    # Save audio
    audio_data = base64.b64decode(result["audio"])
    output_filename = f"output_{seed or 'random'}.{output_format}"
    with open(output_filename, "wb") as f:
        f.write(audio_data)
    
    print(f"Generated: {output_filename}")
    print(f"Duration: {result['duration']:.2f} seconds")
    print(f"Sample rate: {result['sample_rate']} Hz")
    print(f"Parameters used: {json.dumps(result['parameters'], indent=2)}")
    
    return result

def voice_cloning_example():
    """Example of voice cloning with custom parameters"""
    
    result = advanced_tts(
        text="This is a test of voice cloning with custom expressiveness!",
        exaggeration=1.5,          # High expressiveness
        cfg_weight=0.3,            # Low adherence for more natural flow
        temperature=1.2,           # Moderate randomness
        speed_factor=0.8,          # Slightly slower speech
        seed=123,                  # Reproducible generation
        voice_mode="clone",
        reference_audio_path="reference_voice.wav",
        output_format="wav",
        sample_rate=44100,
        audio_normalization="peak",
        language="en"
    )
    
    return result

def batch_generation_example():
    """Example of generating multiple variations"""
    
    texts = [
        "This is a calm and neutral voice.",
        "This is an excited and expressive voice!",
        "This is a slow and dramatic voice..."
    ]
    
    configs = [
        {"exaggeration": 0.3, "cfg_weight": 0.7, "temperature": 0.5},  # Calm
        {"exaggeration": 1.8, "cfg_weight": 0.2, "temperature": 1.5},  # Excited
        {"exaggeration": 1.0, "cfg_weight": 0.4, "temperature": 0.3}   # Dramatic
    ]
    
    for i, (text, config) in enumerate(zip(texts, configs)):
        result = advanced_tts(
            text=text,
            seed=i,  # Different seed for each
            **config
        )
        print(f"Generated variation {i+1}")

# Example usage
if __name__ == "__main__":
    # Basic TTS
    basic_result = advanced_tts("Hello, this is a basic test!")
    
    # Advanced TTS with custom parameters
    advanced_result = advanced_tts(
        text="This is an advanced test with custom parameters!",
        exaggeration=1.2,
        cfg_weight=0.3,
        temperature=1.1,
        speed_factor=0.9,
        seed=42,
        audio_normalization="peak"
    )
    
    # Voice cloning (if you have reference audio)
    # cloned_result = voice_cloning_example()
    
    # Batch generation
    # batch_generation_example()
```

## Parameter Usage Tips

### For General Use
- **Default settings** (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most content
- **Lower `cfg_weight`** (~0.3) if reference speaker speaks fast
- **Higher `sample_rate`** (44100/48000) for better quality

### For Expressive/Dramatic Speech
- **Lower `cfg_weight`** (~0.3) for more natural flow
- **Higher `exaggeration`** (0.7-2.0) for more emotion
- **Moderate `temperature`** (0.8-1.2) for controlled randomness

### For Voice Cloning
- **Reference audio**: 5-30 seconds of clear speech
- **Lower `cfg_weight`** (0.2-0.4) for better voice matching
- **Moderate `exaggeration`** (0.5-1.0) depending on reference style

### For Production Use
- **Set `seed`** for reproducible results
- **Use `audio_normalization`** for consistent loudness
- **Enable `split_text`** for long content
- **Set appropriate `sample_rate`** for your use case

## Error Handling

The API returns errors in the following format:
```json
{
  "error": "Error message description"
}
```

Common errors and solutions:
- `Text cannot be empty`: Provide non-empty text
- `Text too long`: Split text or reduce length (max 5000 chars)
- `Parameter out of range`: Check parameter ranges in documentation
- `Reference audio required`: Provide reference audio for voice cloning
- `Invalid audio format`: Use supported formats (wav, opus, mp3)

## Performance Optimization

### Cold Start Optimization
- First request: 30-60 seconds (model loading)
- Subsequent requests: 1-5 seconds
- Keep endpoints warm with periodic requests

### Quality vs Speed
- **Higher quality**: `candidates_per_chunk > 1`, higher `sample_rate`
- **Faster generation**: Lower `temperature`, `speed_factor > 1.0`
- **Balanced**: Default parameters with selective optimization

### Memory Management
- **Long text**: Enable `split_text` with appropriate `chunk_size`
- **Batch processing**: Use `parallel_workers` for multiple chunks
- **Reference audio**: Limit `max_reference_duration_sec`

## Cost Optimization

- **GPU Selection**: RTX 4090 for balanced cost/performance
- **Timeout Settings**: Set appropriate timeouts (300-600 seconds)
- **Batch Requests**: Process multiple texts in single session
- **Spot Instances**: Use for development/testing

## Troubleshooting

### Common Issues
1. **Container startup failures**: Check GPU availability and VRAM
2. **Model loading errors**: Ensure sufficient disk space (20GB+)
3. **Audio quality issues**: Verify input formats and parameters
4. **Timeout errors**: Increase timeout or optimize parameters

### Debug Tips
- Check logs in Runpod dashboard
- Test with minimal parameters first
- Verify base64 encoding/decoding
- Use test script for systematic debugging

## Support

- **Chatterbox TTS**: [GitHub Repository](https://github.com/resemble-ai/chatterbox)
- **Runpod Docs**: [Serverless Documentation](https://docs.runpod.io/serverless)
- **Issues**: Create issues in this repository for API-specific problems
- **Discord**: Join the Chatterbox Discord for community support 