#!/usr/bin/env python3
"""
Simple test to generate audio using RunPod Chatterbox TTS and download it
"""

import requests
import json
import time
import base64
import os

# RunPod endpoint
ENDPOINT_URL = "https://api.runpod.ai/v2/bw6fy86ys8iwj2/run"
STATUS_URL = "https://api.runpod.ai/v2/bw6fy86ys8iwj2/status"

def get_api_key():
    """Get RunPod API key from environment or user input"""
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("⚠️ RunPod API key not found in environment variable 'RUNPOD_API_KEY'")
        api_key = input("Please enter your RunPod API key: ").strip()
    return api_key

def submit_job(api_key, text):
    """Submit TTS job to RunPod"""
    print(f"🎤 Submitting TTS job: '{text}'")
    
    payload = {
        "input": {
            "text": text,
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
            "chunk_size": 25,
            "print_metrics": True
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('id')
            print(f"✅ Job submitted successfully! Job ID: {job_id}")
            return job_id
        else:
            print(f"❌ Failed to submit job: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"💥 Error submitting job: {e}")
        return None

def poll_for_result(api_key, job_id):
    """Poll for job completion and get results"""
    print(f"⏳ Polling for job completion...")
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    max_attempts = 30  # 5 minutes max
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{STATUS_URL}/{job_id}", headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                status = result.get('status')
                
                if status == 'COMPLETED':
                    print("✅ Job completed successfully!")
                    return result.get('output')
                elif status == 'FAILED':
                    print("❌ Job failed!")
                    print(f"Error: {result.get('error', 'Unknown error')}")
                    return None
                elif status in ['IN_QUEUE', 'IN_PROGRESS']:
                    print(f"⏳ Status: {status} (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(10)  # Wait 10 seconds before next poll
                else:
                    print(f"⚠️ Unknown status: {status}")
                    time.sleep(10)
                    
            else:
                print(f"❌ Failed to get status: {response.status_code}")
                time.sleep(10)
                
        except Exception as e:
            print(f"💥 Error polling status: {e}")
            time.sleep(10)
            
        attempt += 1
    
    print("⏰ Timeout waiting for job completion")
    return None

def download_audio(output_data, filename="generated_audio.wav"):
    """Download audio from base64 data"""
    try:
        if not output_data:
            print("❌ No output data received")
            return False
            
        # Check if audio_data exists in output
        if 'audio_data' not in output_data:
            print("❌ No audio_data in output")
            print(f"Available keys: {list(output_data.keys())}")
            return False
            
        audio_data = output_data['audio_data']
        print(f"📊 Audio data length: {len(audio_data)} characters")
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data)
        
        # Save to file
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        
        print(f"💾 Audio saved to: {filename}")
        
        # Show metrics if available
        if 'metrics' in output_data:
            metrics = output_data['metrics']
            print(f"📈 Metrics: {metrics}")
            
        return True
        
    except Exception as e:
        print(f"💥 Error downloading audio: {e}")
        return False

def main():
    """Main function"""
    print("🎵 Simple Chatterbox TTS Audio Generation Test")
    print("=" * 50)
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    # Text to generate
    text = "Hello! This is a test of the Chatterbox streaming TTS system. It sounds pretty good, doesn't it?"
    
    # Submit job
    job_id = submit_job(api_key, text)
    if not job_id:
        print("❌ Failed to submit job. Exiting.")
        return
    
    # Poll for results
    output_data = poll_for_result(api_key, job_id)
    if not output_data:
        print("❌ Failed to get results. Exiting.")
        return
    
    # Download audio
    success = download_audio(output_data)
    
    if success:
        print("\n🎉 Success! Audio file generated and downloaded.")
        print("You can now play the 'generated_audio.wav' file.")
    else:
        print("\n❌ Failed to download audio.")

if __name__ == "__main__":
    main() 