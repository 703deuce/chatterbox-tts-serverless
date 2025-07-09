#!/usr/bin/env python3
"""
Quick test for basic TTS mode to verify audio processing fix
"""

import requests
import json
import base64
import soundfile as sf
import io
import os

# RunPod endpoint configuration - set these as environment variables
ENDPOINT_URL = os.getenv("RUNPOD_ENDPOINT_URL", "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run")
API_KEY = os.getenv("RUNPOD_API_KEY", "YOUR_API_KEY_HERE")

def save_audio_from_base64(base64_audio, filename):
    """Save base64 audio to file"""
    try:
        audio_data = base64.b64decode(base64_audio)
        audio_buffer = io.BytesIO(audio_data)
        audio_array, sr = sf.read(audio_buffer)
        sf.write(filename, audio_array, sr)
        print(f"✅ Audio saved to {filename} (sample rate: {sr})")
        return True
    except Exception as e:
        print(f"❌ Error saving audio: {e}")
        return False

def test_basic_mode():
    """Test basic TTS mode"""
    print("🎤 Testing Basic TTS Mode - Audio Processing Fix")
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Please set RUNPOD_API_KEY environment variable")
        return
    
    payload = {
        "input": {
            "operation": "tts",
            "mode": "basic",
            "text": "Testing audio processing fix in basic mode."
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print("📤 Sending request...")
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if 'output' in result and 'audio' in result['output']:
                print("✅ Audio received successfully!")
                print(f"📊 Sample rate: {result['output'].get('sample_rate', 'N/A')}")
                print(f"📝 Mode: {result['output'].get('mode', 'N/A')}")
                
                # Save audio file
                if save_audio_from_base64(result['output']['audio'], "test_basic_fix.wav"):
                    print("✅ Audio processing fix confirmed - no tensor conversion errors!")
                else:
                    print("❌ Audio processing still has issues")
            else:
                print("❌ No audio in response")
                print(f"Full response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_basic_mode() 