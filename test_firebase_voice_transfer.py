#!/usr/bin/env python3
"""
Test script for Firebase Storage integration with voice transfer endpoint
"""

import requests
import base64
import time
import json

# Configuration
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
BASE_URL = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c"
FIREBASE_BUCKET = "aitts-d4c6d.firebasestorage.app"

def test_firebase_voice_transfer():
    """Test voice transfer using Firebase Storage paths"""
    
    print("🔥 Testing Firebase Storage Voice Transfer")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Test 1: Firebase path to voice embedding
    print("\n📋 Test 1: Firebase Path → Voice Embedding")
    payload_1 = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "embedding",
            "input_audio": "",  # Not used when using Firebase path
            "input_storage_path": "users/tts/user123/input_audio.wav",  # Example path
            "storage_bucket": FIREBASE_BUCKET,
            "voice_name": "Amy",
            "no_watermark": True
        }
    }
    
    print(f"📤 Submitting job with payload:")
    print(json.dumps(payload_1, indent=2))
    
    try:
        response = requests.post(f"{BASE_URL}/run", json=payload_1, headers=headers)
        if response.status_code == 200:
            job_id = response.json()['id']
            print(f"✅ Job submitted successfully: {job_id}")
            
            # Poll for completion
            result = poll_job_completion(job_id, headers)
            if result:
                print(f"🎉 Transfer completed!")
                print(f"   Input source: {result.get('transfer_info', {}).get('input_source')}")
                print(f"   Target voice: {result.get('transfer_info', {}).get('target_voice')}")
                print(f"   Duration: {result.get('duration', 0):.2f}s")
        else:
            print(f"❌ Job submission failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Firebase path to Firebase path (audio mode)
    print("\n📋 Test 2: Firebase Path → Firebase Path (Audio Mode)")
    payload_2 = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "audio",
            "input_audio": "",  # Not used when using Firebase paths
            "input_storage_path": "users/tts/user123/source.wav",
            "target_audio": "",  # Not used when using Firebase paths
            "target_storage_path": "users/tts/user123/target.wav",
            "storage_bucket": FIREBASE_BUCKET,
            "no_watermark": True
        }
    }
    
    print(f"📤 Submitting job with payload:")
    print(json.dumps(payload_2, indent=2))
    
    try:
        response = requests.post(f"{BASE_URL}/run", json=payload_2, headers=headers)
        if response.status_code == 200:
            job_id = response.json()['id']
            print(f"✅ Job submitted successfully: {job_id}")
            
            # Poll for completion
            result = poll_job_completion(job_id, headers)
            if result:
                print(f"🎉 Transfer completed!")
                print(f"   Input source: {result.get('transfer_info', {}).get('input_source')}")
                print(f"   Target source: {result.get('transfer_info', {}).get('target_source_type')}")
                print(f"   Duration: {result.get('duration', 0):.2f}s")
        else:
            print(f"❌ Job submission failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_url_voice_transfer():
    """Test voice transfer using regular URLs (existing functionality)"""
    
    print("\n🌐 Testing URL Voice Transfer (Existing)")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "embedding",
            "input_audio": "https://example.com/test_audio.wav",
            "input_is_url": True,
            "voice_name": "Amy",
            "no_watermark": True
        }
    }
    
    print(f"📤 Submitting URL job with payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(f"{BASE_URL}/run", json=payload, headers=headers)
        if response.status_code == 200:
            job_id = response.json()['id']
            print(f"✅ URL job submitted successfully: {job_id}")
            
            # Poll for completion
            result = poll_job_completion(job_id, headers)
            if result:
                print(f"🎉 URL transfer completed!")
                print(f"   Input source: {result.get('transfer_info', {}).get('input_source')}")
        else:
            print(f"❌ URL job submission failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

def poll_job_completion(job_id, headers, timeout=120):
    """Poll job status until completion"""
    
    print(f"⏳ Polling job {job_id}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            status_response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers)
            result = status_response.json()
            status = result.get('status')
            
            print(f"   Status: {status}")
            
            if status == 'COMPLETED':
                return result.get('output')
            elif status == 'FAILED':
                print(f"❌ Job failed: {result.get('error')}")
                return None
            
            time.sleep(3)  # Poll every 3 seconds
            
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            break
    
    print(f"⏰ Job timed out after {timeout} seconds")
    return None

def test_firebase_url_construction():
    """Test Firebase URL construction"""
    
    print("\n🔧 Testing Firebase URL Construction")
    print("=" * 50)
    
    from urllib.parse import quote
    
    test_paths = [
        "users/tts/user123/audio.wav",
        "transcription_uploads/stt/file with spaces.wav",
        "users/tts/user456/special-chars@#$.wav"
    ]
    
    for path in test_paths:
        encoded_path = quote(path, safe='/')
        firebase_url = f"https://firebasestorage.googleapis.com/v0/b/{FIREBASE_BUCKET}/o/{encoded_path}?alt=media"
        print(f"📁 Path: {path}")
        print(f"🔗 URL:  {firebase_url}")
        print()

def main():
    """Run all tests"""
    
    print("🚀 Firebase Storage Voice Transfer Integration Test")
    print("=" * 60)
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Please set your API key in the script before running tests!")
        return
    
    # Test URL construction first
    test_firebase_url_construction()
    
    # Test Firebase Storage integration
    test_firebase_voice_transfer()
    
    # Test existing URL functionality
    test_url_voice_transfer()
    
    print("\n✅ All tests completed!")
    print("\n📝 Summary of new parameters:")
    print("   - input_storage_path: Firebase path for input audio")
    print("   - target_storage_path: Firebase path for target audio") 
    print("   - storage_bucket: Firebase storage bucket (default: aitts-d4c6d.firebasestorage.app)")
    print("   - Existing parameters still work: input_audio, input_is_url, etc.")

if __name__ == "__main__":
    main()
