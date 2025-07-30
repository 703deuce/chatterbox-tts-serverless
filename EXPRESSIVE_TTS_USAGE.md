# Enhanced Expressive TTS with F5-TTS Integration

## Overview

The enhanced handler now supports expressive tags that seamlessly combine Chatterbox TTS for normal text and F5-TTS for expressive segments. This allows for natural, studio-quality speech synthesis with emotional expression.

## Features

✅ **Expressive Tag Support** - Use `{whisper}`, `{yell}`, `{happy}`, etc. for emotional speech  
✅ **Seamless Stitching** - Audio segments are combined with crossfades for natural flow  
✅ **Parallel Processing** - F5-TTS and Chatterbox segments processed simultaneously  
✅ **Voice Consistency** - Same speaker embedding used across all segments  
✅ **Automatic Fallback** - Falls back to Chatterbox if F5-TTS unavailable  

## Supported Expressive Tags

### Native F5-TTS Tags (Built-in Support)

| Tag | Effect | Use Case | Native F5-TTS |
|-----|--------|----------|---------------|
| `{whisper}` | Quiet, soft speech | Secrets, intimate moments | ✅ |
| `{shout}` | Loud, emphasized speech | Excitement, urgency | ✅ |
| `{angry}` | Aggressive tone | Confrontation, frustration | ✅ |
| `{sad}` | Melancholic tone | Emotional, sorrowful content | ✅ |
| `{happy}` | Joyful tone | Celebrations, good news | ✅ |

### Extended Tags (Map to Native Tags)

| Tag | Maps To | Effect | Use Case |
|-----|---------|--------|----------|
| `{yell}` | `{shout}` | Loud, emphasized speech | Alternative to shout |
| `{excited}` | `{happy}` | Energetic tone | Enthusiasm, anticipation |
| `{calm}` | `{whisper}` | Peaceful tone | Meditation, relaxation |
| `{nervous}` | `{sad}` | Anxious tone | Uncertainty, worry |
| `{confident}` | `{happy}` | Assured tone | Leadership, presentations |

### Custom Tags (via Reference Audio)

You can add **unlimited custom expressions** by providing reference audio samples:

```json
{
  "input": {
    "operation": "expressive_tts",
    "text": "This is {robotic}computer-generated{/robotic} speech",
    "voice_name": "Amy",
    "custom_tag_audio": {
      "robotic": "base64_encoded_robot_voice_sample"
    }
  }
}
```

## Usage Examples

### Native F5-TTS Tags

```json
{
  "input": {
    "operation": "expressive_tts",
    "text": "Welcome to our store! {happy}We have amazing deals today!{/happy} {whisper}But this offer is just between us.{/whisper}",
    "voice_name": "Amy",
    "sample_rate": 24000
  }
}
```

### Extended Tags (Auto-Mapped)

```json
{
  "input": {
    "operation": "expressive_tts", 
    "text": "I was walking down the street when {excited}suddenly I heard a loud noise!{/excited} {nervous}I didn't know what to do{/nervous} but then {calm}I realized it was just construction work.{/calm}",
    "voice_id": "abc123ef",
    "crossfade_ms": 150
  }
}
```

### Custom Reference Audio for New Expressions

```json
{
  "input": {
    "operation": "expressive_tts",
    "text": "Welcome to the {robotic}automated system{/robotic}. Please {elderly}speak clearly into the microphone{/elderly}.",
    "voice_name": "Benjamin",
    "custom_tag_audio": {
      "robotic": "UklGRigAAABXQVZFZm10IBAAAAABAAEA...",
      "elderly": "UklGRjgAAABXQVZFZm10IBAAAAABAAEK..."
    }
  }
}
```

### Mixing Native and Custom Tags

```json
{
  "input": {
    "operation": "expressive_tts",
    "text": "{happy}Hello there!{/happy} {pirate}Ahoy matey, welcome aboard!{/pirate} {sad}But the treasure is gone.{/sad}",
    "voice_name": "Christopher",
    "custom_tag_audio": {
      "pirate": "base64_encoded_pirate_voice_sample"
    }
  }
}
```

## API Parameters

### Standard Parameters
- `operation`: Set to `"expressive_tts"` or `"tts"` (auto-detects expressive tags)
- `text`: Text with optional expressive tags
- `voice_name` / `voice_id`: Speaker voice from library  
- `reference_audio`: Custom voice (base64 encoded audio)
- `sample_rate`: Output sample rate (default: 24000)
- `audio_normalization`: "peak", "rms", or null

### Expressive-Specific Parameters
- `crossfade_ms`: Crossfade duration between segments (default: 100ms)
- `max_parallel_segments`: Limit concurrent processing (default: 4)
- `custom_tag_audio`: Dict of custom reference audio for non-native tags
  ```json
  "custom_tag_audio": {
    "tag_name": "base64_encoded_audio_sample",
    "another_tag": "base64_encoded_audio_sample"
  }
  ```

### Custom Reference Audio Guidelines
- **Format**: WAV, MP3, or any format supported by soundfile
- **Duration**: 3-10 seconds recommended (automatically trimmed if longer)
- **Quality**: Clear, expressive samples work best
- **Encoding**: Base64 encoded audio data
- **Sample Rate**: Any (automatically resampled to 24kHz)

### F5-TTS Parameters (when available)
- `speed`: Speech rate multiplier per tag
- `emphasis`: Emotional emphasis level
- `volume_scale`: Volume adjustment per tag

## Response Format

```json
{
  "audio": "base64_encoded_wav_data",
  "sample_rate": 24000,
  "text": "original_input_text",
  "mode": "expressive_enhanced", 
  "parameters": {
    "voice_cloning": true,
    "voice_name": "Amy",
    "crossfade_ms": 100
  },
  "processing_stats": {
    "total_segments": 5,
    "successful_segments": 5,
    "failed_segments": 0,
    "segment_breakdown": {
      "chatterbox_segments": 3,
      "f5_segments": 2,
      "tag_distribution": {"excited": 1, "whisper": 1}
    },
    "total_processing_time": 2.34,
    "audio_duration": 8.7,
    "rtf": 0.269,
    "engines_used": ["chatterbox", "f5"]
  }
}
```

## Tag Syntax

### Full Tag Format
```
{tag}text content{/tag}
```
Example: `{whisper}This is whispered{/whisper}`

### Inline Tag Format  
```
{tag:text content}
```
Example: `{yell:THIS IS LOUD}`

### Nested Tags (Not Supported)
❌ `{excited}This is {whisper}nested{/whisper} content{/excited}`  
✅ `{excited}This is excited{/excited} {whisper}this is whispered{/whisper}`

## Installation

### Required Dependencies

Add to `requirements.txt`:
```
pydub>=0.25.0
f5-tts
librosa>=0.10.0
regex>=2023.0.0
```

### Install Dependencies

```bash
pip install pydub f5-tts librosa regex
```

## Deployment

### 1. Update Handler

Replace your current handler with the enhanced version:

```bash
# Backup current handler
cp handler.py handler_backup.py

# Use enhanced handler
cp enhanced_handler.py handler.py
```

### 2. Deploy to RunPod

The enhanced handler is fully compatible with RunPod serverless infrastructure and maintains backward compatibility with existing TTS operations.

## Performance

### Processing Time
- **Parallel Processing**: F5-TTS and Chatterbox segments process simultaneously
- **Typical RTF**: 0.2-0.4x (faster than real-time)
- **Cold Start**: ~15-30 seconds (includes F5-TTS model loading)

### Memory Usage
- **F5-TTS Model**: ~2-4GB VRAM
- **Chatterbox Model**: ~1-2GB VRAM  
- **Total Recommended**: 8GB+ VRAM for optimal performance

### Fallback Behavior
- If F5-TTS unavailable: Expressive tags processed by Chatterbox
- If segment processing fails: Automatic fallback to Chatterbox
- Graceful degradation ensures requests always complete

## Testing

Run the test suite to verify functionality:

```bash
python test_expressive_tts.py
```

## Troubleshooting

### Common Issues

**F5-TTS Not Available**
- Install with: `pip install f5-tts`
- System will automatically use Chatterbox fallback

**Audio Artifacts at Boundaries**
- Increase `crossfade_ms` parameter (try 150-200ms)
- Check that all segments use same sample rate

**Memory Issues**
- Reduce `max_parallel_segments` 
- Use smaller voice embeddings
- Consider CPU processing for F5-TTS

**Slow Processing**
- Enable parallel processing (default)
- Use GPU acceleration
- Limit text length per request

## Examples

### Storytelling with Emotion
```python
text = """
Once upon a time, there was a brave knight. 
{excited}He was ready for adventure!{/excited} 
But when he reached the dark forest, {nervous}he started to feel afraid.{/nervous}
{whisper}Something was watching him from the shadows.{/whisper}
{yell}Suddenly, a dragon appeared!{/yell}
{confident}But our hero was ready for this moment.{/confident}
"""

payload = {
    "input": {
        "operation": "expressive_tts",
        "text": text,
        "voice_name": "Christopher",
        "crossfade_ms": 120
    }
}
```

### Product Announcement
```python
text = """
{excited}Introducing our revolutionary new product!{/excited}
This innovative solution will change everything.
{whisper}Early bird customers get 50% off{/whisper} but 
{yell:act fast - only 24 hours left!}
{confident}Order now and transform your business.{/confident}
"""
```

The enhanced expressive TTS system delivers natural, emotionally rich speech synthesis perfect for storytelling, presentations, customer service, and any application requiring expressive voice content. 