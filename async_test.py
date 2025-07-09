#!/usr/bin/env python3
"""
Async-aware test for Chatterbox TTS endpoint
"""

import requests
import json
import base64
import time
import os

# Your endpoint URLs
RUN_ENDPOINT = "https://api.runpod.ai/v2/z9h9q9uxwzu04e/run"
STATUS_ENDPOINT = "https://api.runpod.ai/v2/z9h9q9uxwzu04e/status"
RUNSYNC_ENDPOINT = "https://api.runpod.ai/v2/z9h9q9uxwzu04e/runsync"

def test_async():
    # Get API key
    api_key = os.getenv('RUNPOD_API_KEY') or input("Enter your RunPod API key: ").strip()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test payload
    payload = {
        "input": {
            "operation": "tts",
            "text": "Hello! This is an async test of Chatterbox TTS.",
            "exaggeration": 0.5,
            "cfg": 0.5,
            "temperature": 0.7
        }
    }
    
    print("🧪 Testing ASYNC endpoint...")
    print(f"📝 Text: {payload['input']['text']}")
    
    try:
        # Step 1: Submit job
        print("📤 Submitting job...")
        response = requests.post(RUN_ENDPOINT, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Submit failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        result = response.json()
        job_id = result.get('id')
        
        if not job_id:
            print(f"❌ No job ID in response: {result}")
            return False
        
        print(f"✅ Job submitted! ID: {job_id}")
        print(f"🔄 Status: {result.get('status', 'Unknown')}")
        
        # Step 2: Poll for results
        print("⏳ Polling for results...")
        max_attempts = 30  # 5 minutes max
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            time.sleep(10)  # Wait 10 seconds between polls
            
            # Check status
            status_response = requests.get(f"{STATUS_ENDPOINT}/{job_id}", headers=headers, timeout=30)
            
            if status_response.status_code != 200:
                print(f"❌ Status check failed: HTTP {status_response.status_code}")
                continue
            
            status_result = status_response.json()
            status = status_result.get('status', 'Unknown')
            
            print(f"📊 Attempt {attempt}: Status = {status}")
            
            if status == 'COMPLETED':
                output = status_result.get('output')
                if output and 'audio' in output:
                    # Save audio
                    audio_data = base64.b64decode(output['audio'])
                    output_file = "async_test_output.wav"
                    
                    with open(output_file, 'wb') as f:
                        f.write(audio_data)
                    
                    print(f"🎉 SUCCESS! Audio saved to: {output_file}")
                    print(f"🎵 Sample rate: {output.get('sample_rate', 'Unknown')} Hz")
                    print(f"📏 File size: {len(audio_data)} bytes")
                    return True
                else:
                    print(f"❌ Job completed but no audio: {output}")
                    return False
                    
            elif status == 'FAILED':
                error = status_result.get('error', 'Unknown error')
                print(f"❌ Job failed: {error}")
                return False
                
            elif status in ['IN_QUEUE', 'IN_PROGRESS']:
                continue  # Keep polling
            else:
                print(f"❓ Unknown status: {status}")
        
        print("⏰ Timeout: Job didn't complete within 5 minutes")
        return False
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_sync():
    """Try the synchronous endpoint (may timeout for cold starts)"""
    api_key = os.getenv('RUNPOD_API_KEY') or input("Enter your RunPod API key: ").strip()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "operation": "tts",
            "text": "Quick sync test!",
            "exaggeration": 0.5,
            "cfg": 0.5,
            "temperature": 0.7
        }
    }
    
    print("\n🧪 Testing SYNC endpoint (may timeout)...")
    
    try:
        response = requests.post(RUNSYNC_ENDPOINT, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            if 'output' in result and 'audio' in result['output']:
                audio_data = base64.b64decode(result['output']['audio'])
                output_file = "sync_test_output.wav"
                
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                
                print(f"🎉 SYNC SUCCESS! Audio saved to: {output_file}")
                return True
        
        print(f"❌ SYNC failed: HTTP {response.status_code}")
        return False
        
    except requests.exceptions.Timeout:
        print("⏰ SYNC timeout (expected for cold start)")
        return False
    except Exception as e:
        print(f"❌ SYNC error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Chatterbox TTS Async Test")
    print("=" * 40)
    
    # Try async first (recommended)
    success = test_async()
    
    if not success:
        print("\n🔄 Trying sync endpoint...")
        test_sync() 