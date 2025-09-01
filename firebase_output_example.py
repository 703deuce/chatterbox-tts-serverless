#!/usr/bin/env python3
"""
Example showing how to use Firebase Storage for output instead of base64
"""

import requests
import time
import json
from datetime import datetime

# Configuration
API_KEY = "YOUR_API_KEY_HERE"
BASE_URL = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c"
FIREBASE_BUCKET = "aitts-d4c6d.firebasestorage.app"

def generate_output_path(user_id, operation_type="voice_transfer"):
    """Generate a unique output path for Firebase Storage"""
    timestamp = int(datetime.now().timestamp())
    return f"outputs/{operation_type}/{user_id}/transfer_{timestamp}.wav"

def voice_transfer_with_firebase_output(api_key, input_path, target_voice, user_id):
    """
    Perform voice transfer and get Firebase download URL instead of base64
    """
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Generate unique output path
    output_path = generate_output_path(user_id)
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "embedding",
            "input_audio": "",  # Not used when using Firebase path
            "input_storage_path": input_path,
            "storage_bucket": FIREBASE_BUCKET,
            "voice_name": target_voice,
            "output_storage_path": output_path,  # NEW: Where to save output
            "return_download_url": True,         # NEW: Return URL instead of base64
            "no_watermark": True
        }
    }
    
    print(f"🚀 Submitting voice transfer job...")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Target Voice: {target_voice}")
    
    # Submit job
    response = requests.post(f"{BASE_URL}/run", json=payload, headers=headers)
    response.raise_for_status()
    
    job_id = response.json()['id']
    print(f"✅ Job submitted: {job_id}")
    
    # Poll for completion
    result = poll_job_completion(job_id, headers)
    
    if result:
        if 'download_url' in result:
            print(f"🎉 Transfer completed!")
            print(f"   Download URL: {result['download_url']}")
            print(f"   Storage Path: {result['output_storage_path']}")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Output Method: {result['output_method']}")
            return result['download_url']
        else:
            print(f"⚠️  Got base64 response instead of download URL")
            return None
    
    return None

def audio_to_audio_transfer_with_firebase_output(api_key, input_path, target_path, user_id):
    """
    Perform audio-to-audio transfer with Firebase output
    """
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Generate unique output path
    output_path = generate_output_path(user_id, "audio_transfer")
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "transfer_mode": "audio",
            "input_audio": "",
            "input_storage_path": input_path,
            "target_audio": "",
            "target_storage_path": target_path,
            "storage_bucket": FIREBASE_BUCKET,
            "output_storage_path": output_path,  # NEW: Firebase output path
            "return_download_url": True,         # NEW: Get URL not base64
            "no_watermark": True
        }
    }
    
    print(f"🚀 Submitting audio-to-audio transfer job...")
    print(f"   Input: {input_path}")
    print(f"   Target: {target_path}")
    print(f"   Output: {output_path}")
    
    # Submit job
    response = requests.post(f"{BASE_URL}/run", json=payload, headers=headers)
    response.raise_for_status()
    
    job_id = response.json()['id']
    print(f"✅ Job submitted: {job_id}")
    
    # Poll for completion
    result = poll_job_completion(job_id, headers)
    
    if result and 'download_url' in result:
        print(f"🎉 Audio transfer completed!")
        print(f"   Download URL: {result['download_url']}")
        return result['download_url']
    
    return None

def poll_job_completion(job_id, headers, timeout=300):
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
            
            time.sleep(3)
            
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            break
    
    print(f"⏰ Job timed out after {timeout} seconds")
    return None

def test_firebase_output():
    """Test the Firebase output functionality"""
    
    print("🔥 Testing Firebase Storage Output")
    print("=" * 50)
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Please set your API key in the script!")
        return
    
    user_id = "user123"  # Example user ID
    
    # Test 1: Voice transfer with Firebase output
    print("\n📋 Test 1: Voice Transfer → Firebase Output")
    download_url = voice_transfer_with_firebase_output(
        api_key=API_KEY,
        input_path="users/tts/user123/my_recording.wav",
        target_voice="Amy",
        user_id=user_id
    )
    
    if download_url:
        print(f"✅ Success! Download URL: {download_url}")
        
        # Your other app can now download the file directly:
        print(f"\n📥 Your app can download the file like this:")
        print(f"   fetch('{download_url}')")
        print(f"   .then(response => response.blob())")
        print(f"   .then(audioBlob => {{")
        print(f"     // Use the audio blob directly")
        print(f"     const audioUrl = URL.createObjectURL(audioBlob);")
        print(f"     audioElement.src = audioUrl;")
        print(f"   }});")
    
    # Test 2: Audio-to-audio transfer with Firebase output
    print("\n📋 Test 2: Audio-to-Audio Transfer → Firebase Output")
    download_url_2 = audio_to_audio_transfer_with_firebase_output(
        api_key=API_KEY,
        input_path="users/tts/user123/source.wav",
        target_path="users/tts/user123/target.wav",
        user_id=user_id
    )
    
    if download_url_2:
        print(f"✅ Success! Download URL: {download_url_2}")

def compare_response_formats():
    """Compare the old vs new response formats"""
    
    print("\n📊 Response Format Comparison")
    print("=" * 50)
    
    print("\n🔸 OLD (Base64 Response):")
    old_response = {
        "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
        "sample_rate": 24000,
        "duration": 5.32,
        "format": "wav",
        "operation": "voice_transfer",
        "output_method": "base64"
    }
    print(json.dumps(old_response, indent=2))
    
    print("\n🔸 NEW (Firebase Storage Response):")
    new_response = {
        "download_url": "https://firebasestorage.googleapis.com/v0/b/aitts-d4c6d.firebasestorage.app/o/outputs%2Fvoice_transfer%2Fuser123%2Ftransfer_1234567890.wav?alt=media",
        "output_storage_path": "outputs/voice_transfer/user123/transfer_1234567890.wav",
        "sample_rate": 24000,
        "duration": 5.32,
        "format": "wav",
        "operation": "voice_transfer",
        "output_method": "firebase_storage"
    }
    print(json.dumps(new_response, indent=2))
    
    print(f"\n✅ Benefits of new format:")
    print(f"   - No large base64 strings in JSON response")
    print(f"   - Direct download URLs for your frontend")
    print(f"   - Files stored in organized Firebase Storage structure")
    print(f"   - Faster response times (no base64 encoding)")

def main():
    """Run the test"""
    
    print("🚀 Firebase Storage Output Test")
    print("=" * 60)
    
    # Show response format comparison
    compare_response_formats()
    
    # Test the functionality
    test_firebase_output()
    
    print(f"\n📝 New Parameters Summary:")
    print(f"   - output_storage_path: Firebase path where output will be saved")
    print(f"   - return_download_url: Set to true to get download URL instead of base64")
    print(f"   - If upload fails, automatically falls back to base64")

if __name__ == "__main__":
    main()
