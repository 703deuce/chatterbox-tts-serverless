#!/usr/bin/env python3

import requests
import json
import os

# Use the SYNC endpoint for immediate response
SYNC_ENDPOINT = "https://api.runpod.ai/v2/i9dmbiobkkmekx/runsync"

def test_sync():
    # Get API key
    api_key = os.getenv('RUNPOD_API_KEY') or input("Enter your RunPod API key: ").strip()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simple test payload with correct streaming API parameter names
    payload = {
        "input": {
            "operation": "tts",
            "text": "Hello! This is a sync test with the streaming ChatterboxTTS API. The real-time streaming delivers audio chunks progressively for better performance.",
            "exaggeration": 0.5,
            "cfg_weight": 0.7,
            "temperature": 0.8,
            "chunk_size": 25,
            "context_window": 50,
            "fade_duration": 0.02,
            "print_metrics": True
        }
    }
    
    print(f"🧪 Testing SYNC endpoint: {SYNC_ENDPOINT}")
    print(f"📝 Text: {payload['input']['text']}")
    print("⏳ Sending request...")
    
    try:
        response = requests.post(
            SYNC_ENDPOINT, 
            headers=headers, 
            json=payload,
            timeout=120  # 2 minute timeout
        )
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Response received!")
            
            if result.get("status") == "COMPLETED" and result.get("output", {}).get("audio"):
                print(f"🎵 Audio generated successfully!")
                print(f"📏 Sample rate: {result['output'].get('sample_rate', 'unknown')}")
                print(f"📝 Text: {result['output'].get('text', 'unknown')}")
                print(f"⚙️ Parameters: {result['output'].get('parameters', {})}")
                
                # Save audio file
                audio_b64 = result["output"]["audio"]
                import base64
                audio_data = base64.b64decode(audio_b64)
                with open("test_output.wav", "wb") as f:
                    f.write(audio_data)
                print("💾 Audio saved as test_output.wav")
            else:
                print("❌ No audio in response")
                print(json.dumps(result, indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - workers may be stuck or scaling")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_sync() 