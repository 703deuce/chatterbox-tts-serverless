#!/usr/bin/env python3
"""
Comprehensive test script for Chatterbox TTS Runpod Serverless API
Tests all parameters and features including voice cloning and advanced controls
"""

import requests
import base64
import json
import time
import os
from pathlib import Path

# Configuration - Update these values
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "your_runpod_api_key_here")
ENDPOINT_ID = os.getenv("ENDPOINT_ID", "your_endpoint_id_here")
API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

def test_basic_tts():
    """Test basic TTS functionality"""
    print("🧪 Testing basic TTS...")
    
    payload = {
        "input": {
            "task": "tts",
            "text": "Hello, this is a test of the Chatterbox TTS system!"
        }
    }
    
    start_time = time.time()
    response = requests.post(API_URL, json=payload, headers=headers)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        if "audio" in result:
            audio_data = base64.b64decode(result["audio"])
            with open("test_basic.wav", "wb") as f:
                f.write(audio_data)
            
            print(f"✅ Basic TTS test passed!")
            print(f"   Duration: {end_time - start_time:.2f} seconds")
            print(f"   Sample rate: {result['sample_rate']}")
            print(f"   Audio duration: {result.get('duration', 'N/A'):.2f}s")
            print(f"   Audio saved to: test_basic.wav")
            return True
        else:
            print(f"❌ No audio in response: {result}")
            return False
    else:
        print(f"❌ Request failed: {response.status_code} - {response.text}")
        return False

def test_advanced_parameters():
    """Test TTS with advanced parameters"""
    print("\n🧪 Testing advanced TTS parameters...")
    
    payload = {
        "input": {
            "task": "tts",
            "text": "This is an expressive test with advanced parameters for comprehensive voice synthesis control!",
            "exaggeration": 1.2,
            "cfg_weight": 0.3,
            "temperature": 1.1,
            "speed_factor": 0.9,
            "seed": 42,
            "output_format": "wav",
            "sample_rate": 44100,
            "audio_normalization": "peak",
            "language": "en"
        }
    }
    
    start_time = time.time()
    response = requests.post(API_URL, json=payload, headers=headers)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        if "audio" in result:
            audio_data = base64.b64decode(result["audio"])
            with open("test_advanced.wav", "wb") as f:
                f.write(audio_data)
            
            print(f"✅ Advanced parameters test passed!")
            print(f"   Duration: {end_time - start_time:.2f} seconds")
            print(f"   Parameters used: {json.dumps(result['parameters'], indent=4)}")
            print(f"   Audio saved to: test_advanced.wav")
            return True
        else:
            print(f"❌ No audio in response: {result}")
            return False
    else:
        print(f"❌ Request failed: {response.status_code} - {response.text}")
        return False

def test_voice_cloning():
    """Test voice cloning with reference audio"""
    print("\n🧪 Testing voice cloning...")
    
    # Create a dummy reference audio (in real use, load actual audio file)
    # For testing, we'll use a short silence or skip if no reference audio
    reference_audio_path = "reference_voice.wav"
    
    if not Path(reference_audio_path).exists():
        print("⚠️  No reference audio found, skipping voice cloning test")
        print("   Create 'reference_voice.wav' to test voice cloning")
        return True
    
    # Load reference audio
    with open(reference_audio_path, "rb") as f:
        reference_b64 = base64.b64encode(f.read()).decode()
    
    payload = {
        "input": {
            "task": "tts",
            "text": "This text will be spoken in the cloned voice from the reference audio!",
            "voice_mode": "clone",
            "reference_audio": reference_b64,
            "max_reference_duration_sec": 20,
            "exaggeration": 0.8,
            "cfg_weight": 0.4,
            "temperature": 0.9,
            "seed": 123
        }
    }
    
    start_time = time.time()
    response = requests.post(API_URL, json=payload, headers=headers)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        if "audio" in result:
            audio_data = base64.b64decode(result["audio"])
            with open("test_cloned.wav", "wb") as f:
                f.write(audio_data)
            
            print(f"✅ Voice cloning test passed!")
            print(f"   Duration: {end_time - start_time:.2f} seconds")
            print(f"   Audio saved to: test_cloned.wav")
            return True
        else:
            print(f"❌ No audio in response: {result}")
            return False
    else:
        print(f"❌ Request failed: {response.status_code} - {response.text}")
        return False

def test_text_processing():
    """Test text processing parameters"""
    print("\n🧪 Testing text processing parameters...")
    
    long_text = """
    This is a longer text to test the text processing capabilities of the system.
    The API should be able to handle chunking and processing of longer content.
    This includes multiple sentences with various punctuation marks and expressions.
    The system should maintain consistency across the entire generation process.
    """
    
    payload = {
        "input": {
            "task": "tts",
            "text": long_text.strip(),
            "split_text": True,
            "chunk_size": 80,
            "candidates_per_chunk": 1,
            "retries": 2,
            "exaggeration": 0.7,
            "cfg_weight": 0.5,
            "temperature": 0.8
        }
    }
    
    start_time = time.time()
    response = requests.post(API_URL, json=payload, headers=headers)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        if "audio" in result:
            audio_data = base64.b64decode(result["audio"])
            with open("test_text_processing.wav", "wb") as f:
                f.write(audio_data)
            
            print(f"✅ Text processing test passed!")
            print(f"   Duration: {end_time - start_time:.2f} seconds")
            print(f"   Text length: {len(long_text.strip())} characters")
            print(f"   Audio saved to: test_text_processing.wav")
            return True
        else:
            print(f"❌ No audio in response: {result}")
            return False
    else:
        print(f"❌ Request failed: {response.status_code} - {response.text}")
        return False

def test_audio_formats():
    """Test different audio formats and sample rates"""
    print("\n🧪 Testing audio formats and sample rates...")
    
    test_configs = [
        {"output_format": "wav", "sample_rate": 22050, "audio_normalization": "peak"},
        {"output_format": "wav", "sample_rate": 48000, "audio_normalization": "rms"},
        {"output_format": "wav", "sample_rate": 16000, "audio_normalization": None}
    ]
    
    for i, config in enumerate(test_configs):
        print(f"   Testing config {i+1}: {config}")
        
        payload = {
            "input": {
                "task": "tts",
                "text": f"This is audio format test number {i+1}",
                "seed": i,
                **config
            }
        }
        
        response = requests.post(API_URL, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            if "audio" in result:
                audio_data = base64.b64decode(result["audio"])
                filename = f"test_format_{i+1}.wav"
                with open(filename, "wb") as f:
                    f.write(audio_data)
                
                print(f"   ✅ Config {i+1} passed - {filename}")
            else:
                print(f"   ❌ Config {i+1} failed - No audio in response")
                return False
        else:
            print(f"   ❌ Config {i+1} failed - {response.status_code}")
            return False
    
    print(f"✅ Audio formats test passed!")
    return True

def test_expression_variations():
    """Test various expression and style variations"""
    print("\n🧪 Testing expression variations...")
    
    test_cases = [
        {
            "name": "Calm and Neutral",
            "text": "This is a calm and neutral voice speaking clearly.",
            "exaggeration": 0.3,
            "cfg_weight": 0.7,
            "temperature": 0.5,
            "speed_factor": 1.0
        },
        {
            "name": "Excited and Expressive",
            "text": "This is an excited and expressive voice with lots of energy!",
            "exaggeration": 1.8,
            "cfg_weight": 0.2,
            "temperature": 1.5,
            "speed_factor": 1.2
        },
        {
            "name": "Slow and Dramatic",
            "text": "This is a slow and dramatic voice with deep emotion...",
            "exaggeration": 1.0,
            "cfg_weight": 0.4,
            "temperature": 0.3,
            "speed_factor": 0.7
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"   Testing: {case['name']}")
        
        payload = {
            "input": {
                "task": "tts",
                "text": case["text"],
                "exaggeration": case["exaggeration"],
                "cfg_weight": case["cfg_weight"],
                "temperature": case["temperature"],
                "speed_factor": case["speed_factor"],
                "seed": i
            }
        }
        
        response = requests.post(API_URL, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            if "audio" in result:
                audio_data = base64.b64decode(result["audio"])
                filename = f"test_expression_{i+1}_{case['name'].lower().replace(' ', '_')}.wav"
                with open(filename, "wb") as f:
                    f.write(audio_data)
                
                print(f"   ✅ {case['name']} - {filename}")
            else:
                print(f"   ❌ {case['name']} - No audio in response")
                return False
        else:
            print(f"   ❌ {case['name']} - {response.status_code}")
            return False
    
    print(f"✅ Expression variations test passed!")
    return True

def test_voice_conversion():
    """Test voice conversion functionality"""
    print("\n🧪 Testing voice conversion...")
    
    # Check if test audio files exist
    source_path = "source_audio.wav"
    target_path = "target_audio.wav"
    
    if not Path(source_path).exists() or not Path(target_path).exists():
        print("⚠️  No test audio files found, skipping voice conversion test")
        print("   Create 'source_audio.wav' and 'target_audio.wav' to test voice conversion")
        return True
    
    # Load test audio files
    with open(source_path, "rb") as f:
        source_b64 = base64.b64encode(f.read()).decode()
    
    with open(target_path, "rb") as f:
        target_b64 = base64.b64encode(f.read()).decode()
    
    payload = {
        "input": {
            "task": "vc",
            "source_audio": source_b64,
            "target_audio": target_b64,
            "max_source_duration_sec": 30,
            "max_target_duration_sec": 20,
            "output_format": "wav",
            "sample_rate": 44100,
            "audio_normalization": "peak"
        }
    }
    
    start_time = time.time()
    response = requests.post(API_URL, json=payload, headers=headers)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        if "audio" in result:
            audio_data = base64.b64decode(result["audio"])
            with open("test_voice_conversion.wav", "wb") as f:
                f.write(audio_data)
            
            print(f"✅ Voice conversion test passed!")
            print(f"   Duration: {end_time - start_time:.2f} seconds")
            print(f"   Source duration: {result.get('source_duration', 'N/A'):.2f}s")
            print(f"   Target duration: {result.get('target_duration', 'N/A'):.2f}s")
            print(f"   Audio saved to: test_voice_conversion.wav")
            return True
        else:
            print(f"❌ No audio in response: {result}")
            return False
    else:
        print(f"❌ Request failed: {response.status_code} - {response.text}")
        return False

def test_error_handling():
    """Test error handling and parameter validation"""
    print("\n🧪 Testing error handling...")
    
    error_tests = [
        {
            "name": "Empty text",
            "payload": {"input": {"task": "tts", "text": ""}},
            "expected_error": "Text cannot be empty"
        },
        {
            "name": "Invalid exaggeration",
            "payload": {"input": {"task": "tts", "text": "Test", "exaggeration": 3.0}},
            "expected_error": "Exaggeration must be between"
        },
        {
            "name": "Invalid cfg_weight",
            "payload": {"input": {"task": "tts", "text": "Test", "cfg_weight": 1.5}},
            "expected_error": "CFG weight must be between"
        },
        {
            "name": "Invalid temperature",
            "payload": {"input": {"task": "tts", "text": "Test", "temperature": 10.0}},
            "expected_error": "Temperature must be between"
        },
        {
            "name": "Invalid voice mode",
            "payload": {"input": {"task": "tts", "text": "Test", "voice_mode": "invalid"}},
            "expected_error": "Voice mode must be"
        },
        {
            "name": "Missing reference audio for cloning",
            "payload": {"input": {"task": "tts", "text": "Test", "voice_mode": "clone"}},
            "expected_error": "Reference audio required"
        }
    ]
    
    passed_tests = 0
    
    for test in error_tests:
        print(f"   Testing: {test['name']}")
        
        response = requests.post(API_URL, json=test["payload"], headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            if "error" in result and test["expected_error"] in result["error"]:
                print(f"   ✅ {test['name']} - Error correctly caught")
                passed_tests += 1
            else:
                print(f"   ❌ {test['name']} - Unexpected response: {result}")
        else:
            print(f"   ❌ {test['name']} - Request failed: {response.status_code}")
    
    if passed_tests == len(error_tests):
        print(f"✅ Error handling test passed! ({passed_tests}/{len(error_tests)})")
        return True
    else:
        print(f"❌ Error handling test failed! ({passed_tests}/{len(error_tests)})")
        return False

def main():
    """Run all comprehensive tests"""
    print("🚀 Starting Comprehensive Chatterbox TTS API Tests")
    print(f"API URL: {API_URL}")
    print("-" * 60)
    
    # Check configuration
    if RUNPOD_API_KEY == "your_runpod_api_key_here":
        print("❌ Please set your RUNPOD_API_KEY environment variable or update the script")
        return
    
    if ENDPOINT_ID == "your_endpoint_id_here":
        print("❌ Please set your ENDPOINT_ID environment variable or update the script")
        return
    
    # Test suite
    tests = [
        ("Basic TTS", test_basic_tts),
        ("Advanced Parameters", test_advanced_parameters),
        ("Voice Cloning", test_voice_cloning),
        ("Text Processing", test_text_processing),
        ("Audio Formats", test_audio_formats),
        ("Expression Variations", test_expression_variations),
        ("Voice Conversion", test_voice_conversion),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} completed successfully")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    end_time = time.time()
    
    print("\n" + "="*60)
    print(f"🏁 Test Results Summary")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working correctly with full parameter support.")
    else:
        print("⚠️  Some tests failed. Check the configuration and deployment.")
    
    print("\n📁 Generated test files:")
    test_files = [
        "test_basic.wav",
        "test_advanced.wav", 
        "test_cloned.wav",
        "test_text_processing.wav",
        "test_format_1.wav",
        "test_format_2.wav",
        "test_format_3.wav",
        "test_expression_1_calm_and_neutral.wav",
        "test_expression_2_excited_and_expressive.wav",
        "test_expression_3_slow_and_dramatic.wav",
        "test_voice_conversion.wav"
    ]
    
    for filename in test_files:
        if Path(filename).exists():
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename} (not generated)")

if __name__ == "__main__":
    main() 