#!/usr/bin/env python3
"""
Debug script to check what's happening with the basic TTS test
"""

import requests
import json
import time

# Configuration
API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
ENDPOINT_URL = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def debug_basic_tts():
    """Debug the basic TTS call"""
    print("🔍 Debugging Basic TTS")
    
    payload = {
        "input": {
            "operation": "tts",
            "text": "Hello world test",
            "voice_name": "amy"
        }
    }
    
    print(f"📤 Sending payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=30)
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data.get('id')
            print(f"✅ Job ID: {job_id}")
            
            # Poll for result
            status_url = f"https://api.runpod.ai/v2/c2wmx1ln5ccp6c/status/{job_id}"
            
            for i in range(30):  # Wait up to 60 seconds
                time.sleep(2)
                status_response = requests.get(status_url, headers=headers)
                
                if status_response.status_code == 200:
                    result = status_response.json()
                    status = result.get('status')
                    print(f"🔄 Poll {i+1}: Status = {status}")
                    
                    if status == 'COMPLETED':
                        output = result.get('output', {})
                        print(f"📋 Full output keys: {list(output.keys())}")
                        print(f"📋 Full output: {json.dumps(output, indent=2)}")
                        break
                    elif status == 'FAILED':
                        print(f"❌ Job failed: {result}")
                        break
                else:
                    print(f"❌ Status check failed: {status_response.status_code}")
                    break
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"📋 Response: {response.text}")
    
    except Exception as e:
        print(f"💥 Exception: {e}")

if __name__ == "__main__":
    debug_basic_tts() 