#!/usr/bin/env python3
"""
Quick test to see if enhanced handler auto-detects expressive tags
"""

import requests
import json
import base64
import time

API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
API_URL = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"

def poll_job(job_id):
    """Poll for job completion"""
    status_url = f"https://api.runpod.ai/v2/c2wmx1ln5ccp6c/status/{job_id}"
    
    for i in range(120):  # 2 minutes max
        response = requests.get(
            status_url,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            result = response.json()
            status = result.get('status', 'unknown')
            
            if status == 'COMPLETED':
                return result.get('output')
            elif status == 'FAILED':
                print(f"❌ Job failed: {result.get('error', 'Unknown error')}")
                return None
            elif i % 10 == 0:
                print(f"⏳ Status: {status} ({i+1}s)")
                
        time.sleep(1)
    
    print("❌ Timeout")
    return None

def test_expressive_with_tts_operation():
    """Test expressive tags using standard 'tts' operation"""
    print("🧪 Testing Expressive Tags with Standard TTS Operation")
    
    payload = {
        "input": {
            "operation": "tts",  # Use standard operation
            "text": "Hello! {happy}This should be happy{/happy} and {whisper}this should be quiet{/whisper}",
            "voice_name": "Amy",
            "sample_rate": 24000
        }
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        if 'id' in result:
            print(f"Job ID: {result['id']}")
            output = poll_job(result['id'])
            
            if output:
                print(f"✅ Success! Mode: {output.get('mode', 'unknown')}")
                
                # Check if expressive processing was detected
                if 'expressive' in output.get('mode', '').lower():
                    print("🎉 Enhanced expressive handler is working!")
                else:
                    print("📋 Standard handler used (expressive tags not detected)")
                
                # Save audio
                if 'audio' in output:
                    audio_data = base64.b64decode(output['audio'])
                    with open('test_auto_detection.wav', 'wb') as f:
                        f.write(audio_data)
                    print(f"Audio saved: test_auto_detection.wav ({len(audio_data)} bytes)")
                
                return True
            else:
                print("❌ Job failed")
                return False
    else:
        print(f"❌ Request failed: {response.status_code}")
        return False

if __name__ == "__main__":
    test_expressive_with_tts_operation() 