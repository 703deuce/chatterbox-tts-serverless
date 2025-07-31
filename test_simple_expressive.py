#!/usr/bin/env python3
"""
Simple test to debug expressive TTS issues
"""

import requests
import json
import time

API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
API_URL = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"

def poll_job(job_id, timeout=120):
    """Poll for job completion"""
    status_url = f"https://api.runpod.ai/v2/c2wmx1ln5ccp6c/status/{job_id}"
    
    for i in range(timeout):
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
                error = result.get('error', 'Unknown error')
                print(f"❌ Job failed: {error}")
                return {'error': error}
            elif i % 10 == 0:
                print(f"   Status: {status} ({i+1}s)")
                
        time.sleep(1)
    
    return None

def test_simple_expressive():
    """Test with very simple expressive text"""
    print("🧪 Testing Simple Expressive TTS")
    
    payload = {
        "input": {
            "operation": "expressive_tts",
            "text": "Hello {happy}world{/happy}",  # Very simple test
            "voice_name": "Amy"
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
                if 'error' in output:
                    print(f"❌ Error: {output['error']}")
                else:
                    print(f"✅ Success! Keys: {list(output.keys())}")
                    if 'processing_stats' in output:
                        print(f"Stats: {output['processing_stats']}")
                return output
    
    print("❌ Request failed")
    return None

if __name__ == "__main__":
    test_simple_expressive() 