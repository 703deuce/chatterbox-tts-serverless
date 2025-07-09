#!/usr/bin/env python3
"""
Test script for all three Chatterbox TTS modes:
1. Basic TTS Generation
2. Streaming TTS Generation  
3. Streaming with Voice Cloning

Based on the GitHub documentation examples.
"""

import requests
import json
import base64
import soundfile as sf
import time
import io
import os

# RunPod endpoint configuration - set these as environment variables
ENDPOINT_URL = os.getenv("RUNPOD_ENDPOINT_URL", "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run")
API_KEY = os.getenv("RUNPOD_API_KEY", "YOUR_API_KEY_HERE")

def load_reference_audio(file_path):
    """Load reference audio file and convert to base64"""
    try:
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        return base64.b64encode(audio_data).decode('utf-8')
    except FileNotFoundError:
        print(f"Reference audio file not found: {file_path}")
        return None

def save_audio_from_base64(base64_audio, filename, sample_rate=24000):
    """Save base64 audio to file"""
    try:
        audio_data = base64.b64decode(base64_audio)
        audio_buffer = io.BytesIO(audio_data)
        audio_array, sr = sf.read(audio_buffer)
        sf.write(filename, audio_array, sr)
        print(f"Audio saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

def test_basic_tts():
    """Test Mode 1: Basic TTS Generation (as per GitHub docs)"""
    print("\n=== Testing Basic TTS Generation ===")
    
    payload = {
        "input": {
            "operation": "tts",
            "mode": "basic",
            "text": "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if 'output' in result and 'audio' in result['output']:
                save_audio_from_base64(result['output']['audio'], "test_basic.wav")
                print("✅ Basic TTS generation successful!")
            else:
                print("❌ No audio in response")
        else:
            print(f"❌ Request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_streaming_tts():
    """Test Mode 2: Streaming TTS Generation (as per GitHub docs)"""
    print("\n=== Testing Streaming TTS Generation ===")
    
    payload = {
        "input": {
            "operation": "tts",
            "mode": "streaming",
            "text": "Welcome to the world of streaming text-to-speech! This audio will be generated and played in real-time chunks.",
            "chunk_size": 50,  # Default from GitHub docs
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
            "print_metrics": True
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if 'output' in result and 'audio' in result['output']:
                save_audio_from_base64(result['output']['audio'], "test_streaming.wav")
                
                # Print streaming metrics
                if 'streaming_metrics' in result['output']:
                    metrics = result['output']['streaming_metrics']
                    print(f"📊 Streaming Metrics:")
                    print(f"  - Total chunks: {metrics.get('total_chunks', 'N/A')}")
                    print(f"  - First chunk latency: {metrics.get('first_chunk_latency', 'N/A'):.3f}s")
                    print(f"  - Total generation time: {metrics.get('total_generation_time', 'N/A'):.3f}s")
                    print(f"  - Audio duration: {metrics.get('audio_duration', 'N/A'):.3f}s")
                    print(f"  - RTF: {metrics.get('rtf', 'N/A'):.3f}")
                
                print("✅ Streaming TTS generation successful!")
            else:
                print("❌ No audio in response")
        else:
            print(f"❌ Request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_streaming_voice_cloning():
    """Test Mode 3: Streaming with Voice Cloning (as per GitHub docs)"""
    print("\n=== Testing Streaming with Voice Cloning ===")
    
    # You would need to provide a reference audio file
    # For this test, we'll create a simple sine wave as placeholder
    reference_audio_b64 = None
    
    # Uncomment and modify this line to use a real reference audio file:
    # reference_audio_b64 = load_reference_audio("reference_voice.wav")
    
    if not reference_audio_b64:
        print("⚠️  No reference audio provided. Creating placeholder...")
        # Create a simple sine wave as placeholder (this won't work for actual voice cloning)
        import numpy as np
        sample_rate = 24000
        duration = 3.0
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration))
        sine_wave = np.sin(2 * np.pi * frequency * t) * 0.1
        
        # Save as WAV and convert to base64
        buffer = io.BytesIO()
        sf.write(buffer, sine_wave, sample_rate, format='WAV')
        buffer.seek(0)
        reference_audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        print("📝 Created placeholder reference audio (sine wave)")
    
    payload = {
        "input": {
            "operation": "tts",
            "mode": "streaming_voice_cloning",
            "text": "This streaming synthesis will use a custom voice from the reference audio file.",
            "reference_audio": reference_audio_b64,
            "chunk_size": 25,  # Smaller chunks for lower latency as per docs
            "exaggeration": 0.7,  # Default from GitHub example
            "cfg_weight": 0.3,    # Default from GitHub example
            "temperature": 0.8,
            "print_metrics": True
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"Request payload: {json.dumps({**payload, 'input': {**payload['input'], 'reference_audio': '[BASE64_AUDIO_DATA]'}}, indent=2)}")
    
    try:
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if 'output' in result and 'audio' in result['output']:
                save_audio_from_base64(result['output']['audio'], "test_streaming_voice_cloning.wav")
                
                # Print streaming metrics
                if 'streaming_metrics' in result['output']:
                    metrics = result['output']['streaming_metrics']
                    print(f"📊 Streaming Voice Cloning Metrics:")
                    print(f"  - Total chunks: {metrics.get('total_chunks', 'N/A')}")
                    print(f"  - First chunk latency: {metrics.get('first_chunk_latency', 'N/A'):.3f}s")
                    print(f"  - Total generation time: {metrics.get('total_generation_time', 'N/A'):.3f}s")
                    print(f"  - Audio duration: {metrics.get('audio_duration', 'N/A'):.3f}s")
                    print(f"  - RTF: {metrics.get('rtf', 'N/A'):.3f}")
                
                print("✅ Streaming Voice Cloning generation successful!")
            else:
                print("❌ No audio in response")
        else:
            print(f"❌ Request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all three test modes"""
    print("🎤 Testing Chatterbox TTS - All Three Modes")
    print("=" * 50)
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Please set RUNPOD_API_KEY environment variable")
        print("Example: export RUNPOD_API_KEY=rpa_your_key_here")
        print("         export RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/your_endpoint_id/run")
        return
    
    # Test all three modes
    test_basic_tts()
    test_streaming_tts()
    test_streaming_voice_cloning()
    
    print("\n" + "=" * 50)
    print("🏁 Testing complete!")
    print("\nModes supported:")
    print("✅ basic - Simple generate() call")
    print("✅ streaming - generate_stream() without voice cloning")
    print("✅ streaming_voice_cloning - generate_stream() with audio_prompt_path")

if __name__ == "__main__":
    main() 