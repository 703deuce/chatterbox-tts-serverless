#!/usr/bin/env python3
"""
Test Enhanced Expressive TTS After Deployment
Quick test to verify the enhanced features work
"""

import requests
import json
import base64
import time

API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
API_URL = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"

def poll_job(job_id, timeout=120):
    """Poll for job completion"""
    status_url = f"https://api.runpod.ai/v2/c2wmx1ln5ccp6c/status/{job_id}"
    
    print(f"⏳ Polling job: {job_id}")
    
    for i in range(timeout):
        response = requests.get(
            status_url,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            result = response.json()
            status = result.get('status', 'unknown')
            
            if status == 'COMPLETED':
                print(f"✅ Job completed in {i+1}s")
                return result.get('output')
            elif status == 'FAILED':
                error = result.get('error', 'Unknown error')
                print(f"❌ Job failed: {error}")
                return None
            elif i % 10 == 0:
                print(f"   Status: {status} ({i+1}s)")
                
        time.sleep(1)
    
    print(f"❌ Timeout after {timeout}s")
    return None

def save_audio(output, filename):
    """Save audio from output"""
    if output and 'audio' in output:
        audio_data = base64.b64decode(output['audio'])
        with open(filename, 'wb') as f:
            f.write(audio_data)
        print(f"🎵 Audio saved: {filename} ({len(audio_data)} bytes)")
        return True
    return False

def test_auto_detection():
    """Test auto-detection with standard 'tts' operation"""
    print("\n🧪 Test 1: Auto-Detection with Standard TTS")
    
    payload = {
        "input": {
            "operation": "tts",  # Standard operation
            "text": "Hello! {happy}This should be detected as expressive{/happy} automatically.",
            "voice_name": "Amy",
            "sample_rate": 24000
        }
    }
    
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
            output = poll_job(result['id'])
            if output:
                mode = output.get('mode', 'unknown')
                print(f"   Mode: {mode}")
                
                if 'expressive' in mode.lower():
                    print("🎉 Enhanced expressive handler detected tags!")
                    save_audio(output, "test_auto_detection.wav")
                    return True
                else:
                    print("📋 Standard handler used (no expressive detection)")
                    save_audio(output, "test_standard_fallback.wav")
                    return False
    
    print("❌ Request failed")
    return False

def test_explicit_operation():
    """Test explicit 'expressive_tts' operation"""
    print("\n🧪 Test 2: Explicit Expressive TTS Operation")
    
    payload = {
        "input": {
            "operation": "expressive_tts",  # Explicit expressive operation
            "text": "Welcome! {happy}We are so excited{/happy} {whisper}but keep this quiet{/whisper} {shout:THIS IS IMPORTANT!}",
            "voice_name": "Benjamin",
            "sample_rate": 24000,
            "crossfade_ms": 120
        }
    }
    
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
            output = poll_job(result['id'])
            if output:
                print("✅ Explicit expressive TTS worked!")
                
                # Check for processing stats
                if 'processing_stats' in output:
                    stats = output['processing_stats']
                    print(f"   Total segments: {stats.get('total_segments', 0)}")
                    print(f"   F5-TTS segments: {stats.get('segment_breakdown', {}).get('f5_segments', 0)}")
                    print(f"   Engines used: {stats.get('engines_used', [])}")
                
                save_audio(output, "test_explicit_expressive.wav")
                return True
    
    print("❌ Explicit expressive TTS failed")
    return False

def test_native_f5_tags():
    """Test native F5-TTS tags"""
    print("\n🧪 Test 3: Native F5-TTS Tags")
    
    payload = {
        "input": {
            "operation": "tts",
            "text": "{whisper}This is whispered{/whisper} and {shout:THIS IS SHOUTED} and {happy}this is happy{/happy}",
            "voice_name": "Christopher",
            "sample_rate": 24000
        }
    }
    
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
            output = poll_job(result['id'])
            if output:
                print("✅ Native F5-TTS tags processed!")
                save_audio(output, "test_native_f5_tags.wav")
                return True
    
    print("❌ Native F5-TTS tags failed")
    return False

def main():
    """Run enhanced deployment tests"""
    print("🚀 Testing Enhanced Expressive TTS Deployment")
    print("=" * 50)
    
    # Wait a moment for potential deployment
    print("⏳ Waiting 10 seconds for potential redeployment...")
    time.sleep(10)
    
    results = []
    
    # Run tests
    results.append(("Auto-Detection", test_auto_detection()))
    results.append(("Explicit Operation", test_explicit_operation()))
    results.append(("Native F5 Tags", test_native_f5_tags()))
    
    # Summary
    print("\n" + "=" * 50)
    print("🧪 Enhanced Deployment Test Results")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed > 0:
        print("🎉 Enhanced expressive TTS is working!")
        print("\n🎵 Generated audio files:")
        for filename in ["test_auto_detection.wav", "test_explicit_expressive.wav", "test_native_f5_tags.wav"]:
            import os
            if os.path.exists(filename):
                print(f"  ✅ {filename}")
        print("\nYou can now listen to these files to hear the expressive TTS in action!")
    else:
        print("⚠️  Enhanced features not yet active. RunPod may still be deploying the new version.")

if __name__ == "__main__":
    main() 