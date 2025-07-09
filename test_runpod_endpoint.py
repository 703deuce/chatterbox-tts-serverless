#!/usr/bin/env python3
"""
Test script for RunPod Chatterbox TTS Streaming endpoint
Tests the corrected streaming API with proper parameters
"""

import requests
import json
import time
import base64
import io
import wave
import os

# RunPod endpoint
ENDPOINT_URL = "https://api.runpod.ai/v2/i9dmbiobkkmekx/run"

def get_api_key():
    """Get RunPod API key from environment or user input"""
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("⚠️ RunPod API key not found in environment variable 'RUNPOD_API_KEY'")
        api_key = input("Please enter your RunPod API key: ").strip()
    return api_key

def test_basic_streaming():
    """Test basic streaming functionality"""
    print("🧪 Testing basic streaming...")
    
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key provided. Cannot test endpoint.")
        return False
    
    payload = {
        "input": {
            "text": "Hello, this is a test of the Chatterbox streaming TTS system. The quick brown fox jumps over the lazy dog.",
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
            "chunk_size": 25,
            "context_window": 50,
            "fade_duration": 0.02,
            "print_metrics": True
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        print(f"📡 Sending request to: {ENDPOINT_URL}")
        print(f"📝 Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers, timeout=120)
        print(f"🔄 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response: {json.dumps(result, indent=2)}")
            
            # Check if we have audio data
            if 'output' in result and 'audio_data' in result['output']:
                print("🎵 Audio data received!")
                audio_data = result['output']['audio_data']
                print(f"📊 Audio data length: {len(audio_data)} characters")
                
                # Try to decode and save audio
                try:
                    audio_bytes = base64.b64decode(audio_data)
                    with open("test_output.wav", "wb") as f:
                        f.write(audio_bytes)
                    print("💾 Saved audio to test_output.wav")
                except Exception as e:
                    print(f"⚠️ Could not decode audio: {e}")
            
            # Check for streaming metrics
            if 'output' in result and 'metrics' in result['output']:
                metrics = result['output']['metrics']
                print(f"📈 Streaming metrics: {metrics}")
                
            return True
            
        elif response.status_code == 401:
            print("❌ Authentication failed. Please check your API key.")
            return False
        elif response.status_code == 400:
            print("❌ Bad request. Check the payload format.")
            print(f"Response: {response.text}")
            return False
        elif response.status_code == 500:
            print("❌ Server error. The endpoint may be having issues.")
            print(f"Response: {response.text}")
            return False
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - this could indicate a cold start")
        return False
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

def test_parameter_variations():
    """Test different parameter combinations"""
    print("\n🧪 Testing parameter variations...")
    
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key provided. Cannot test parameter variations.")
        return
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    test_cases = [
        {
            "name": "High exaggeration",
            "params": {
                "text": "This is a test with high exaggeration.",
                "exaggeration": 0.9,
                "cfg_weight": 0.5,
                "temperature": 0.8
            }
        },
        {
            "name": "Low temperature",
            "params": {
                "text": "This is a test with low temperature for consistency.",
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "temperature": 0.3
            }
        },
        {
            "name": "Large chunks",
            "params": {
                "text": "This is a test with larger chunk size for fewer but bigger chunks.",
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "temperature": 0.8,
                "chunk_size": 50
            }
        },
        {
            "name": "Small chunks",
            "params": {
                "text": "This is a test with smaller chunk size for more frequent updates.",
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "temperature": 0.8,
                "chunk_size": 10
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        
        payload = {
            "input": test_case['params']
        }
        
        try:
            response = requests.post(ENDPOINT_URL, json=payload, headers=headers, timeout=90)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {test_case['name']}: Success")
                
                # Check for metrics
                if 'output' in result and 'metrics' in result['output']:
                    metrics = result['output']['metrics']
                    if 'total_chunks' in metrics:
                        print(f"📊 Total chunks: {metrics['total_chunks']}")
                    if 'generation_time' in metrics:
                        print(f"⏱️ Generation time: {metrics['generation_time']:.3f}s")
                
            else:
                print(f"❌ {test_case['name']}: Failed ({response.status_code})")
                print(f"Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"💥 {test_case['name']}: Error - {e}")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🧪 Testing edge cases...")
    
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key provided. Cannot test edge cases.")
        return
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    edge_cases = [
        {
            "name": "Empty text",
            "params": {"text": ""}
        },
        {
            "name": "Very long text",
            "params": {
                "text": "This is a very long text that goes on and on and on. " * 20,
                "chunk_size": 25
            }
        },
        {
            "name": "Invalid parameter",
            "params": {
                "text": "Test with invalid parameter",
                "invalid_param": "should_be_ignored"
            }
        },
        {
            "name": "Extreme temperature",
            "params": {
                "text": "Test with extreme temperature",
                "temperature": 1.5
            }
        }
    ]
    
    for test_case in edge_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        
        payload = {
            "input": test_case['params']
        }
        
        try:
            response = requests.post(ENDPOINT_URL, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {test_case['name']}: Handled gracefully")
            else:
                print(f"⚠️ {test_case['name']}: Status {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"💥 {test_case['name']}: Error - {e}")

def main():
    """Run all tests"""
    print("🚀 Starting RunPod Chatterbox TTS Streaming Tests")
    print("=" * 50)
    
    # Test basic functionality
    success = test_basic_streaming()
    
    if success:
        print("\n✅ Basic test passed! Running additional tests...")
        
        # Test parameter variations
        test_parameter_variations()
        
        # Test edge cases
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed!")
        print("Check the logs above for any issues.")
        
    else:
        print("\n❌ Basic test failed. Please check the endpoint and try again.")
        print("Common issues:")
        print("- Invalid API key")
        print("- Endpoint not deployed yet")
        print("- Cold start taking longer than expected")
        print("- Import or dependency issues")

if __name__ == "__main__":
    main() 