#!/usr/bin/env python3
"""
Test the optimization components via RunPod API calls
This tests if the optimized components are working on RunPod
"""

import requests
import json
import time

# RunPod configuration
RUNPOD_API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"  # Your API key
RUNPOD_ENDPOINT_URL = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"

def test_optimization_via_api():
    """Test if optimized components are available via API"""
    
    print("🧪 Testing Optimization Components via RunPod API")
    print("=" * 55)
    
    # Test 1: Check if optimized voice library works
    print("\n📚 Test 1: Optimized Voice Library Test")
    test_payload = {
        "input": {
            "operation": "test_optimization",
            "test_type": "voice_library"
        }
    }
    
    try:
        response = requests.post(
            RUNPOD_ENDPOINT_URL,
            headers={
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type": "application/json"
            },
            json=test_payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Voice library test API call successful")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error calling API: {e}")
    
    # Test 2: Test basic TTS with optimization indicator
    print(f"\n🎤 Test 2: Basic TTS (check for optimization indicators)")
    tts_payload = {
        "input": {
            "operation": "tts",
            "mode": "basic",
            "text": "Testing optimized voice loading performance",
            "voice_name": "Amy"
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            RUNPOD_ENDPOINT_URL,
            headers={
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type": "application/json"
            },
            json=tts_payload
        )
        api_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ TTS API call successful in {api_time:.3f}s")
            
            # Check for optimization indicators in response
            if 'output' in result:
                output = result['output']
                params = output.get('parameters', {})
                optimization = params.get('optimization', 'not_found')
                
                if optimization == 'direct_audio_array':
                    print(f"🚀 OPTIMIZATION DETECTED: {optimization}")
                else:
                    print(f"📊 Optimization status: {optimization}")
                    
                print(f"Voice cloning enabled: {params.get('voice_cloning', False)}")
                print(f"Mode: {output.get('mode', 'unknown')}")
            else:
                print(f"Response structure: {list(result.keys())}")
                
        else:
            print(f"❌ TTS API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error calling TTS API: {e}")

def poll_job_result(job_id: str, max_wait: int = 60):
    """Poll for job completion"""
    status_url = f"https://api.runpod.ai/v2/c2wmx1ln5ccp6c/status/{job_id}"
    
    for i in range(max_wait):
        try:
            response = requests.get(
                status_url,
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
            )
            
            if response.status_code == 200:
                result = response.json()
                status = result.get('status', 'unknown')
                
                if status == 'COMPLETED':
                    return result.get('output')
                elif status == 'FAILED':
                    print(f"❌ Job failed: {result.get('error', 'Unknown error')}")
                    return None
                else:
                    print(f"⏳ Job status: {status} (waiting...)")
                    
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error polling job: {e}")
            break
    
    print(f"❌ Job timed out after {max_wait} seconds")
    return None

if __name__ == "__main__":
    test_optimization_via_api() 