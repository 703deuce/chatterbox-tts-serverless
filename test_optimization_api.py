#!/usr/bin/env python3
"""
Simple test to check if optimization components work via existing API
"""

import requests
import json
import time

# RunPod configuration
RUNPOD_API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
RUNPOD_ENDPOINT_URL = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"

def test_voice_loading_performance():
    """Test voice loading performance via existing list_local_voices operation"""
    
    print("🧪 Testing Voice Loading Performance")
    print("=" * 40)
    
    # Test multiple voice list calls to see loading performance
    payload = {
        "input": {
            "operation": "list_local_voices"
        }
    }
    
    times = []
    for i in range(3):
        print(f"\nTest {i+1}/3...")
        
        start_time = time.time()
        try:
            response = requests.post(
                RUNPOD_ENDPOINT_URL,
                headers={
                    "Authorization": f"Bearer {RUNPOD_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Wait for completion if it's async
                if 'id' in result and 'status' in result:
                    job_id = result['id']
                    print(f"Job ID: {job_id}, polling for result...")
                    
                    # Poll for result
                    output = poll_for_result(job_id)
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    if output:
                        voice_count = len(output.get('local_voices', []))
                        print(f"✅ Found {voice_count} voices in {total_time:.3f}s")
                        times.append(total_time)
                        
                        # Check if optimization indicators are present
                        if 'optimization' in output:
                            print(f"🚀 Optimization detected: {output['optimization']}")
                        
                    else:
                        print(f"❌ Job failed or timed out")
                        
                else:
                    # Synchronous response
                    end_time = time.time()
                    total_time = end_time - start_time
                    print(f"✅ Synchronous response in {total_time:.3f}s")
                    times.append(total_time)
                    
            else:
                print(f"❌ API call failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📊 Average response time: {avg_time:.3f}s")
        print(f"Response times: {[f'{t:.3f}s' for t in times]}")
    
    # Now test TTS with voice to see if optimization is being used
    print(f"\n🎤 Testing TTS with voice (Amy)...")
    tts_payload = {
        "input": {
            "operation": "tts",
            "mode": "basic",
            "text": "Testing optimized voice loading",
            "voice_name": "Amy"
        }
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            RUNPOD_ENDPOINT_URL,
            headers={
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type": "application/json"
            },
            json=tts_payload
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if 'id' in result:
                job_id = result['id']
                print(f"TTS Job ID: {job_id}, polling...")
                
                output = poll_for_result(job_id, max_wait=120)
                end_time = time.time()
                total_time = end_time - start_time
                
                if output:
                    print(f"✅ TTS completed in {total_time:.3f}s")
                    
                    # Check for optimization indicators
                    params = output.get('parameters', {})
                    if 'optimization' in params:
                        print(f"🚀 TTS Optimization: {params['optimization']}")
                    else:
                        print(f"📊 Parameters: {params}")
                        
                else:
                    print(f"❌ TTS job failed or timed out")
                    
    except Exception as e:
        print(f"❌ TTS test error: {e}")

def poll_for_result(job_id: str, max_wait: int = 60):
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
                elif status in ['IN_PROGRESS', 'IN_QUEUE']:
                    print(f"⏳ Status: {status}")
                    
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Polling error: {e}")
            break
    
    return None

if __name__ == "__main__":
    test_voice_loading_performance() 