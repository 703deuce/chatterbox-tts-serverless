#!/usr/bin/env python3
"""
Test script for Chatterbox TTS Runpod Serverless Endpoint
"""

import requests
import json
import base64
import time
import os
from typing import Dict, Any

# Runpod endpoint configuration
ENDPOINT_URL = "https://api.runpod.ai/v2/z9h9q9uxwzu04e/run"
ENDPOINT_ID = "z9h9q9uxwzu04e"

def get_api_key():
    """Get API key from environment variable or user input"""
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        api_key = input("Enter your RunPod API key: ").strip()
    return api_key

def test_basic_tts(api_key: str):
    """Test basic text-to-speech functionality"""
    print("🧪 Testing Basic TTS...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "operation": "tts",
            "text": "Hello! This is a test of the Chatterbox TTS API. How does it sound?",
            "exaggeration": 0.5,
            "cfg": 0.5,
            "temperature": 0.7
        }
    }
    
    print(f"📤 Sending request to: {ENDPOINT_URL}")
    print(f"📝 Text: {payload['input']['text']}")
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Request successful!")
            
            if 'output' in result and 'audio' in result['output']:
                # Save the audio file
                audio_data = base64.b64decode(result['output']['audio'])
                output_file = "test_basic_tts.wav"
                
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                
                print(f"🎵 Audio saved to: {output_file}")
                print(f"📊 Sample Rate: {result['output'].get('sample_rate', 'N/A')} Hz")
                print(f"📏 Audio Duration: ~{len(audio_data) / (result['output'].get('sample_rate', 24000) * 2):.2f} seconds")
                return True
            else:
                print("❌ No audio in response")
                print(f"Response: {json.dumps(result, indent=2)}")
                return False
                
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_voice_cloning(api_key: str):
    """Test voice cloning functionality"""
    print("\n🎭 Testing Voice Cloning...")
    
    # Create a simple reference audio (you'd normally use a real audio file)
    print("⚠️  Note: This test uses a placeholder for reference audio.")
    print("   For real voice cloning, you'd provide actual audio data.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "operation": "tts",
            "text": "This is a test of voice cloning with Chatterbox TTS.",
            "voice_mode": "predefined",  # Using predefined instead of clone for this test
            "exaggeration": 0.7,
            "cfg": 0.3,
            "temperature": 0.8
        }
    }
    
    print(f"📝 Text: {payload['input']['text']}")
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Voice cloning test successful!")
            
            if 'output' in result and 'audio' in result['output']:
                # Save the audio file
                audio_data = base64.b64decode(result['output']['audio'])
                output_file = "test_voice_clone.wav"
                
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                
                print(f"🎵 Audio saved to: {output_file}")
                return True
            else:
                print("❌ No audio in response")
                return False
                
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_parameter_variations(api_key: str):
    """Test different parameter combinations"""
    print("\n⚙️  Testing Parameter Variations...")
    
    test_cases = [
        {
            "name": "High Exaggeration",
            "params": {"exaggeration": 0.9, "cfg": 0.3, "temperature": 0.6},
            "text": "I'm absolutely thrilled to test this!"
        },
        {
            "name": "Low Exaggeration", 
            "params": {"exaggeration": 0.2, "cfg": 0.7, "temperature": 0.9},
            "text": "This is a calm and measured test."
        },
        {
            "name": "Balanced Settings",
            "params": {"exaggeration": 0.5, "cfg": 0.5, "temperature": 0.7},
            "text": "Testing balanced parameter settings."
        }
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"\n  📋 Test {i+1}: {test_case['name']}")
        
        payload = {
            "input": {
                "operation": "tts",
                "text": test_case['text'],
                **test_case['params']
            }
        }
        
        try:
            response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'output' in result and 'audio' in result['output']:
                    audio_data = base64.b64decode(result['output']['audio'])
                    output_file = f"test_params_{i+1}_{test_case['name'].lower().replace(' ', '_')}.wav"
                    
                    with open(output_file, 'wb') as f:
                        f.write(audio_data)
                    
                    print(f"     ✅ Success! Audio saved to: {output_file}")
                    success_count += 1
                else:
                    print(f"     ❌ No audio in response")
            else:
                print(f"     ❌ Failed with status: {response.status_code}")
                
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    print(f"\n📊 Parameter test results: {success_count}/{len(test_cases)} successful")
    return success_count == len(test_cases)

def main():
    """Main test function"""
    print("🚀 Chatterbox TTS Runpod Endpoint Test")
    print("=" * 50)
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key provided")
        return
    
    print(f"🔗 Testing endpoint: {ENDPOINT_URL}")
    print(f"🔑 Using API key: {api_key[:8]}...")
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_basic_tts(api_key):
        tests_passed += 1
    
    if test_voice_cloning(api_key):
        tests_passed += 1
    
    if test_parameter_variations(api_key):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📈 Test Summary: {tests_passed}/{total_tests} test suites passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your endpoint is working perfectly!")
    elif tests_passed > 0:
        print("⚠️  Some tests passed. Check the errors above.")
    else:
        print("❌ All tests failed. Check your endpoint configuration and API key.")
    
    print("\n💡 Generated audio files:")
    for file in ["test_basic_tts.wav", "test_voice_clone.wav"]:
        if os.path.exists(file):
            print(f"   🎵 {file}")

if __name__ == "__main__":
    main() 