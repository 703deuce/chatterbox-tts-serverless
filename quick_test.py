#!/usr/bin/env python3
"""
Quick test for Chatterbox TTS endpoint
"""

import requests
import json
import base64
import os

# Your endpoint URL
ENDPOINT_URL = "https://api.runpod.ai/v2/z9h9q9uxwzu04e/run"

def quick_test():
    # Get API key
    api_key = os.getenv('RUNPOD_API_KEY') or input("Enter your RunPod API key: ").strip()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simple test payload
    payload = {
        "input": {
            "operation": "tts",
            "text": "Hello! This is a quick test of Chatterbox TTS. How does it sound?",
            "exaggeration": 0.5,
            "cfg": 0.5,
            "temperature": 0.7
        }
    }
    
    print(f"🧪 Testing endpoint: {ENDPOINT_URL}")
    print(f"📝 Text: {payload['input']['text']}")
    
    try:
        print("📤 Sending request...")
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=120)
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if 'output' in result and 'audio' in result['output']:
                # Save audio
                audio_data = base64.b64decode(result['output']['audio'])
                output_file = "quick_test_output.wav"
                
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                
                print(f"✅ SUCCESS! Audio saved to: {output_file}")
                print(f"🎵 Sample rate: {result['output'].get('sample_rate', 'Unknown')} Hz")
                print(f"📏 File size: {len(audio_data)} bytes")
                
                # Show response details
                if 'parameters' in result['output']:
                    print(f"⚙️  Parameters used: {result['output']['parameters']}")
                
            else:
                print("❌ FAILED: No audio in response")
                print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Request took too long (>120s)")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    quick_test() 