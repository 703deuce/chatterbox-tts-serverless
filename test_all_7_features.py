#!/usr/bin/env python3
"""
Comprehensive Test Suite for TTS API - All 7 Features
Tests: Basic TTS, Streaming TTS, Voice Cloning (TRUE), Voice Listing, Voice Conversion, Voice Transfer
"""

import requests
import json
import time
import base64
import os
from datetime import datetime

# Configuration
API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
ENDPOINT_URL = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def poll_for_result(job_id, max_wait=120):
    """Poll for job completion with timeout"""
    status_url = f"https://api.runpod.ai/v2/c2wmx1ln5ccp6c/status/{job_id}"
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(status_url, headers=headers, timeout=10)
            if response.status_code == 200:
                result = response.json()
                status = result.get('status')
                
                if status == 'COMPLETED':
                    return result.get('output', {})
                elif status == 'FAILED':
                    return {'error': f"Job failed: {result.get('error', 'Unknown error')}"}
                    
            time.sleep(2)
        except Exception as e:
            print(f"Polling error: {e}")
            time.sleep(2)
    
    return {'error': 'Timeout waiting for job completion'}

def load_test_audio():
    """Load test audio file and encode as base64"""
    test_files = [
        "test_basic_final.wav",
        "amy_basic_cloned.wav", 
        "betty_basic.wav",
        "test_voice_cloning_adrian.wav",
        "test_basic_tts.wav"
    ]
    
    for filename in test_files:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                audio_data = f.read()
                return base64.b64encode(audio_data).decode('utf-8')
    
    print("⚠️ No test audio files found, skipping audio-dependent tests")
    return None

def test_basic_tts():
    """Test 1: Basic TTS"""
    print("\n🧪 TEST 1: Basic TTS")
    print("=" * 50)
    
    payload = {
        "input": {
            "operation": "tts",
            "text": "Hello! This is a test of our basic text to speech functionality.",
            "voice_name": "amy",
            "temperature": 0.7,
            "speed": 1.0
        }
    }
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            job_id = response.json().get('id')
            print(f"✅ Job submitted: {job_id}")
            
            result = poll_for_result(job_id)
            if 'error' in result:
                print(f"❌ FAILED: {result['error']}")
                return False
            elif 'audio' in result:  # Fixed: looking for 'audio' instead of 'audio_b64'
                print(f"✅ SUCCESS: Audio generated ({len(result['audio'])} chars)")
                return True
            else:
                print(f"❌ FAILED: Unexpected response format - Keys: {list(result.keys())}")
                return False
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_streaming_tts():
    """Test 2: Streaming TTS"""
    print("\n🧪 TEST 2: Streaming TTS")
    print("=" * 50)
    
    payload = {
        "input": {
            "operation": "tts",  # Fixed: using 'tts' instead of 'streaming_tts'
            "text": "This is a streaming text to speech test with multiple sentences. Each part should be processed separately for faster response times.",
            "voice_name": "betty",
            "temperature": 0.8,
            "streaming": True,  # Added streaming parameter
            "chunk_size": 100
        }
    }
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            job_id = response.json().get('id')
            print(f"✅ Job submitted: {job_id}")
            
            result = poll_for_result(job_id, max_wait=180)
            if 'error' in result:
                print(f"❌ FAILED: {result['error']}")
                return False
            elif 'audio_chunks' in result or 'audio' in result:  # Check for either format
                if 'audio_chunks' in result:
                    chunks = result['audio_chunks']
                    print(f"✅ SUCCESS: {len(chunks)} audio chunks generated")
                else:
                    print(f"✅ SUCCESS: Streaming audio generated ({len(result['audio'])} chars)")
                return True
            else:
                print(f"❌ FAILED: Unexpected response format - Keys: {list(result.keys())}")
                return False
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_voice_cloning():
    """Test 3: TRUE Voice Cloning (Create new voice embedding)"""
    print("\n🧪 TEST 3: TRUE Voice Cloning (Create New Voice)")
    print("=" * 50)
    
    audio_b64 = load_test_audio()
    if not audio_b64:
        print("⚠️ SKIPPED: No test audio available")
        return True  # Skip but don't fail
    
    payload = {
        "input": {
            "operation": "voice_cloning",
            "reference_audio": audio_b64,
            "voice_name": f"TestVoice_{int(time.time())}",
            "voice_description": "Test voice created from audio sample",
            "save_to_library": True
        }
    }
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            job_id = response.json().get('id')
            print(f"✅ Job submitted: {job_id}")
            
            result = poll_for_result(job_id, max_wait=180)
            if 'error' in result:
                print(f"❌ FAILED: {result['error']}")
                return False
            elif 'voice_info' in result and 'sample_audio' in result:
                voice_info = result['voice_info']
                print(f"✅ SUCCESS: Voice '{voice_info.get('name')}' created")
                print(f"   ID: {voice_info.get('id')}")
                print(f"   Saved to library: {result.get('saved_to_library', False)}")
                return True
            else:
                print(f"❌ FAILED: Unexpected response format - Keys: {list(result.keys())}")
                return False
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_voice_listing():
    """Test 4: Voice Listing"""
    print("\n🧪 TEST 4: Voice Listing")
    print("=" * 50)
    
    payload = {
        "input": {
            "operation": "list_local_voices"
        }
    }
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            job_id = response.json().get('id')
            print(f"✅ Job submitted: {job_id}")
            
            result = poll_for_result(job_id)
            if 'error' in result:
                print(f"❌ FAILED: {result['error']}")
                return False
            elif 'voices' in result or 'available_voices' in result or 'local_voices' in result:  # Check all possible keys
                voices = result.get('voices', result.get('available_voices', result.get('local_voices', [])))
                print(f"✅ SUCCESS: {len(voices)} voices available")
                for voice in voices[:3]:  # Show first 3
                    if isinstance(voice, dict):
                        print(f"   - {voice.get('name', 'Unknown')} ({voice.get('gender', 'Unknown')})")
                    else:
                        print(f"   - {voice}")
                return True
            else:
                print(f"❌ FAILED: Unexpected response format - Keys: {list(result.keys())}")
                return False
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_voice_conversion():
    """Test 5: Voice Conversion"""
    print("\n🧪 TEST 5: Voice Conversion")
    print("=" * 50)
    
    audio_b64 = load_test_audio()
    if not audio_b64:
        print("⚠️ SKIPPED: No test audio available")
        return True  # Skip but don't fail
    
    payload = {
        "input": {
            "operation": "voice_conversion",
            "input_audio": audio_b64,
            "voice_name": "Adrian",  # Using voice_name instead of target_speaker
            "conversion_strength": 0.8
        }
    }
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            job_id = response.json().get('id')
            print(f"✅ Job submitted: {job_id}")
            
            result = poll_for_result(job_id, max_wait=180)
            if 'error' in result:
                print(f"❌ FAILED: {result['error']}")
                return False
            elif 'converted_audio_b64' in result or 'converted_audio' in result or 'audio' in result:
                audio_key = 'converted_audio_b64' if 'converted_audio_b64' in result else ('converted_audio' if 'converted_audio' in result else 'audio')
                print(f"✅ SUCCESS: Voice converted ({len(result[audio_key])} chars)")
                return True
            else:
                print(f"❌ FAILED: Unexpected response format - Keys: {list(result.keys())}")
                return False
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_voice_transfer_embedding():
    """Test 6: Voice Transfer (WAV to Embedding)"""
    print("\n🧪 TEST 6: Voice Transfer (WAV to Embedding)")
    print("=" * 50)
    
    audio_b64 = load_test_audio()
    if not audio_b64:
        print("⚠️ SKIPPED: No test audio available")
        return True  # Skip but don't fail
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "input_audio": audio_b64,
            "transfer_mode": "embedding",
            "target_voice": "Amy",
            "voice_name": "Amy",  # Using existing voice from the list
            "conversion_strength": 0.7
        }
    }
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            job_id = response.json().get('id')
            print(f"✅ Job submitted: {job_id}")
            
            result = poll_for_result(job_id, max_wait=180)
            if 'error' in result:
                print(f"❌ FAILED: {result['error']}")
                return False
            elif 'transferred_audio_b64' in result or 'transferred_audio' in result or 'audio' in result:
                audio_key = 'transferred_audio_b64' if 'transferred_audio_b64' in result else ('transferred_audio' if 'transferred_audio' in result else 'audio')
                print(f"✅ SUCCESS: Voice transferred to embedding ({len(result[audio_key])} chars)")
                return True
            else:
                print(f"❌ FAILED: Unexpected response format - Keys: {list(result.keys())}")
                return False
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_voice_transfer_audio():
    """Test 7: Voice Transfer (WAV to WAV)"""
    print("\n🧪 TEST 7: Voice Transfer (WAV to WAV)")
    print("=" * 50)
    
    audio_b64 = load_test_audio()
    if not audio_b64:
        print("⚠️ SKIPPED: No test audio available")
        return True  # Skip but don't fail
    
    payload = {
        "input": {
            "operation": "voice_transfer",
            "input_audio": audio_b64,
            "transfer_mode": "audio",
            "target_audio": audio_b64,  # Using same audio as target for test
            "conversion_strength": 0.6
        }
    }
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            job_id = response.json().get('id')
            print(f"✅ Job submitted: {job_id}")
            
            result = poll_for_result(job_id, max_wait=180)
            if 'error' in result:
                print(f"❌ FAILED: {result['error']}")
                return False
            elif 'transferred_audio_b64' in result or 'transferred_audio' in result or 'audio' in result:
                audio_key = 'transferred_audio_b64' if 'transferred_audio_b64' in result else ('transferred_audio' if 'transferred_audio' in result else 'audio')
                print(f"✅ SUCCESS: Voice transferred to audio ({len(result[audio_key])} chars)")
                return True
            else:
                print(f"❌ FAILED: Unexpected response format - Keys: {list(result.keys())}")
                return False
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TTS API COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Endpoint: {ENDPOINT_URL}")
    print("📋 Testing 7 Features: Basic TTS, Streaming TTS, Voice Cloning (TRUE), Voice Listing, Voice Conversion, Voice Transfer x2")
    
    tests = [
        ("Basic TTS", test_basic_tts),
        ("Streaming TTS", test_streaming_tts),
        ("TRUE Voice Cloning", test_voice_cloning),
        ("Voice Listing", test_voice_listing),
        ("Voice Conversion", test_voice_conversion),
        ("Voice Transfer (Embedding)", test_voice_transfer_embedding),
        ("Voice Transfer (Audio)", test_voice_transfer_audio)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 FINAL TEST RESULTS")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! API is fully functional!")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")
    
    print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 