# Chatterbox TTS Serverless API (Streaming)

A serverless API for the [Chatterbox TTS Streaming](https://github.com/davidbrowne17/chatterbox-streaming) text-to-speech model, optimized for RunPod GPU infrastructure with real-time streaming capabilities.

## Features

- **Real-time streaming**: Leverages the streaming version of ChatterboxTTS for faster response times
- **GPU-accelerated**: Optimized for NVIDIA GPUs with CUDA support
- **Zero-shot voice cloning**: Use reference audio for voice conversion
- **Emotion control**: Adjust exaggeration and CFG parameters
- **Comprehensive API**: Supports all ChatterboxTTS parameters
- **Production-ready**: Handles errors, validation, and edge cases

## Streaming Performance

The streaming version provides:
- Real-time factor of 0.499 (target < 1) on RTX 4090
- Latency to first chunk: ~0.472s
- Significantly improved UX for real-time applications

## API Parameters

### Core Parameters
- `text` (required): Text to convert to speech
- `exaggeration` (0.0-1.0): Emotion exaggeration level (default: 0.5)
- `cfg` (0.0-1.0): CFG weight for generation control (default: 0.5) 
- `temperature` (0.1-2.0): Temperature for generation (default: 0.7)
- `seed` (int): Random seed for reproducibility

### Voice Parameters
- `voice_mode`: "predefined" or "clone"
- `reference_audio`: Base64 encoded reference audio for voice cloning
- `max_reference_duration_sec`: Maximum reference audio duration (1-60s)

### Audio Parameters
- `output_format`: "wav", "opus", or "mp3"
- `sample_rate`: 16000, 22050, 24000, 44100, or 48000
- `audio_normalization`: "peak" or "rms"

## Usage

### Basic Text-to-Speech
```python
payload = {
    "input": {
        "operation": "tts",
        "text": "Hello! This is ChatterboxTTS with streaming capabilities.",
        "exaggeration": 0.5,
        "cfg": 0.5,
        "temperature": 0.7
    }
}
```

### Voice Cloning
```python
payload = {
    "input": {
        "operation": "tts",
        "text": "This is a cloned voice!",
        "voice_mode": "clone",
        "reference_audio": "base64_encoded_audio_data",
        "exaggeration": 0.7,
        "cfg": 0.3
    }
}
```

## RunPod Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t chatterbox-streaming .
   ```

2. **Deploy to RunPod**:
   - Create a new serverless endpoint
   - Use the built Docker image
   - Configure GPU requirements (recommended: RTX 4090 or better)

3. **Test the endpoint**:
   ```bash
   python sync_test.py
   ```

## Technical Details

### Dependencies
- **ChatterboxTTS Streaming**: Real-time TTS with streaming support
- **PyTorch**: GPU-accelerated tensor operations
- **RunPod SDK**: Serverless infrastructure integration
- **Audio Processing**: soundfile, librosa for audio handling

### Performance Optimizations
- Streaming generation for reduced latency
- GPU memory management
- Efficient audio processing pipeline
- Optimized Docker image with CUDA support

## Error Handling

The API includes comprehensive error handling for:
- Invalid parameters and ranges
- Audio format issues
- GPU memory constraints
- Network timeouts
- Voice cloning failures

## Testing

Run the test script to verify functionality:
```bash
python sync_test.py
```

The test script will:
1. Generate speech from text
2. Save the audio output
3. Verify streaming performance
4. Test parameter validation

## Links

- **Original ChatterboxTTS**: https://github.com/ResembleAI/chatterbox
- **Streaming Version**: https://github.com/davidbrowne17/chatterbox-streaming  
- **RunPod Documentation**: https://docs.runpod.io/
- **Hugging Face Model**: https://huggingface.co/ResembleAI/chatterbox

## License

MIT License - see the original ChatterboxTTS repository for details. 