#!/usr/bin/env python3
"""
Example: How an external application would use the voice embedding generator

This demonstrates how another app can generate embeddings that work
with your handler without any modifications.
"""

from generate_voice_embedding import VoiceEmbeddingGenerator, create_voice_embedding

# ============================================================================
# Example 1: Simple file-based generation
# ============================================================================

def example_simple():
    """Simple example - generate from audio file"""
    print("Example 1: Simple file-based generation")
    print("-" * 50)
    
    generator = VoiceEmbeddingGenerator(embeddings_dir="./voice_embeddings")
    
    result = generator.from_file(
        audio_file="example_voice.wav",  # Your audio file
        voice_name="ExampleVoice",
        gender="male",
        voice_description="Example voice for testing"
    )
    
    print(f"✅ Created voice: {result['voice_name']}")
    print(f"   Voice ID: {result['voice_id']}")
    print(f"   Duration: {result['duration']:.2f}s")
    print(f"\nNow you can use this in your handler:")
    print(f'   voice_name: "{result["voice_name"]}"')
    print()


# ============================================================================
# Example 2: Using convenience function
# ============================================================================

def example_convenience():
    """Using the convenience function"""
    print("Example 2: Using convenience function")
    print("-" * 50)
    
    # Automatically detects input type
    result = create_voice_embedding(
        audio_input="example_voice.wav",
        voice_name="ConvenienceVoice",
        gender="female"
    )
    
    print(f"✅ Created: {result['voice_name']} ({result['voice_id']})")
    print()


# ============================================================================
# Example 3: From numpy array (e.g., from audio processing pipeline)
# ============================================================================

def example_from_array():
    """Generate from numpy array"""
    print("Example 3: From numpy array")
    print("-" * 50)
    
    import numpy as np
    import soundfile as sf
    
    # Load audio into numpy array
    audio_array, sample_rate = sf.read("example_voice.wav")
    
    generator = VoiceEmbeddingGenerator()
    result = generator.from_array(
        audio_array=audio_array,
        sample_rate=sample_rate,
        voice_name="ArrayVoice",
        gender="male"
    )
    
    print(f"✅ Created from array: {result['voice_name']}")
    print()


# ============================================================================
# Example 4: From base64 (e.g., from web upload)
# ============================================================================

def example_from_base64():
    """Generate from base64 string (common in web apps)"""
    print("Example 4: From base64 string")
    print("-" * 50)
    
    import base64
    
    # Simulate base64 audio from web upload
    with open("example_voice.wav", "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    generator = VoiceEmbeddingGenerator()
    result = generator.from_base64(
        audio_base64=audio_base64,
        voice_name="Base64Voice",
        gender="female"
    )
    
    print(f"✅ Created from base64: {result['voice_name']}")
    print()


# ============================================================================
# Example 5: Batch processing multiple voices
# ============================================================================

def example_batch():
    """Process multiple voices at once"""
    print("Example 5: Batch processing")
    print("-" * 50)
    
    generator = VoiceEmbeddingGenerator()
    
    voices_to_create = [
        ("voice1.wav", "Voice1", "male"),
        ("voice2.wav", "Voice2", "female"),
        ("voice3.wav", "Voice3", "male"),
    ]
    
    results = []
    for audio_file, name, gender in voices_to_create:
        try:
            result = generator.from_file(
                audio_file=audio_file,
                voice_name=name,
                gender=gender
            )
            results.append(result)
            print(f"✅ {name}: {result['voice_id']}")
        except FileNotFoundError:
            print(f"❌ File not found: {audio_file}")
        except Exception as e:
            print(f"❌ Error creating {name}: {e}")
    
    print(f"\nTotal created: {len(results)} voices")
    print()


# ============================================================================
# Example 6: Integration with TTS API
# ============================================================================

def example_with_api():
    """Generate embedding then use it in TTS API"""
    print("Example 6: Integration with TTS API")
    print("-" * 50)
    
    import requests
    
    # Step 1: Generate embedding
    generator = VoiceEmbeddingGenerator()
    result = generator.from_file(
        audio_file="example_voice.wav",
        voice_name="APIVoice",
        gender="male"
    )
    
    print(f"✅ Embedding created: {result['voice_name']} ({result['voice_id']})")
    
    # Step 2: Use in TTS API (your handler)
    # Note: This is just an example - adjust to your actual API endpoint
    try:
        api_response = requests.post(
            "https://your-api-endpoint/tts",
            json={
                "input": {
                    "operation": "tts",
                    "mode": "basic",
                    "text": "Hello! This is my new voice.",
                    "voice_name": result['voice_name']  # Use the generated voice!
                }
            },
            headers={"Authorization": "Bearer YOUR_API_KEY"}
        )
        
        if api_response.status_code == 200:
            print("✅ TTS API call successful!")
        else:
            print(f"❌ API error: {api_response.status_code}")
    except Exception as e:
        print(f"⚠️  API call failed (this is expected if endpoint not configured): {e}")
    
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Voice Embedding Generator - External Usage Examples")
    print("=" * 60)
    print()
    
    # Run examples (comment out ones that need actual files)
    try:
        example_simple()
    except FileNotFoundError:
        print("⚠️  Example 1 skipped: example_voice.wav not found")
        print()
    
    try:
        example_convenience()
    except FileNotFoundError:
        print("⚠️  Example 2 skipped: example_voice.wav not found")
        print()
    
    # Uncomment to run other examples:
    # example_from_array()
    # example_from_base64()
    # example_batch()
    # example_with_api()
    
    print("=" * 60)
    print("Note: These examples require actual audio files to run.")
    print("Replace 'example_voice.wav' with your actual audio file path.")
    print("=" * 60)

