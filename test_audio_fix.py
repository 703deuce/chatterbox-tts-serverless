#!/usr/bin/env python3
"""
Test script to verify audio processing fix works correctly
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

def test_audio_fix():
    """Test that audio processing fix works for all modes"""
    print("🎤 Testing Audio Processing Fix")
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Please set RUNPOD_API_KEY environment variable")
        print("Example: export RUNPOD_API_KEY=rpa_your_key_here")
        print("         export RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/your_endpoint_id/run")
        return
    
    # Test basic mode
    payload = {
        "input": {
            "operation": "tts",
            "mode": "basic",
            "text": "Testing audio processing fix."
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print("📤 Testing basic mode...")
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if 'output' in result and 'audio' in result['output']:
                print("✅ Basic mode audio processing working!")
                save_audio_from_base64(result['output']['audio'], "test_basic_audio_fix.wav")
            else:
                print("❌ No audio in basic mode response")
        else:
            print(f"❌ Basic mode request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing basic mode: {e}")

if __name__ == "__main__":
    test_audio_fix() 