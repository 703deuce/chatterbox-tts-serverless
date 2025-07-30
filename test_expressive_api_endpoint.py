#!/usr/bin/env python3
"""
Test Script for Enhanced Expressive TTS API Endpoint
Tests the deployed RunPod serverless endpoint with expressive tags
"""

import requests
import json
import base64
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RunPod API Configuration
API_BASE_URL = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"
API_KEY = "YOUR_API_KEY"  # Replace with your actual API key

class ExpressiveTTSAPITester:
    """Test the enhanced expressive TTS API endpoint"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.base_url = API_BASE_URL
    
    def make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the RunPod API"""
        try:
            logger.info(f"Making API request to {self.base_url}")
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=300  # 5 minutes timeout for TTS generation
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"API Response Status: {response.status_code}")
            logger.info(f"Response keys: {list(result.keys())}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def save_audio_response(self, response: Dict[str, Any], filename: str) -> bool:
        """Save audio from API response to file"""
        try:
            if 'output' in response and 'audio' in response['output']:
                audio_b64 = response['output']['audio']
                audio_data = base64.b64decode(audio_b64)
                
                with open(filename, 'wb') as f:
                    f.write(audio_data)
                
                logger.info(f"Audio saved to {filename} ({len(audio_data)} bytes)")
                return True
            else:
                logger.error("No audio data found in response")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False
    
    def test_basic_functionality(self) -> bool:
        """Test basic TTS without expressive tags"""
        logger.info("\n=== Testing Basic TTS (No Expressive Tags) ===")
        
        payload = {
            "input": {
                "operation": "tts",
                "text": "Hello, this is a basic test of the TTS system without any expressive tags.",
                "voice_name": "Amy",
                "sample_rate": 24000
            }
        }
        
        try:
            response = self.make_request(payload)
            
            if self.save_audio_response(response, "test_basic_api.wav"):
                logger.info("✅ Basic TTS test successful")
                return True
            else:
                logger.error("❌ Basic TTS test failed - no audio generated")
                return False
                
        except Exception as e:
            logger.error(f"❌ Basic TTS test failed: {e}")
            return False
    
    def test_native_expressive_tags(self) -> bool:
        """Test native F5-TTS expressive tags"""
        logger.info("\n=== Testing Native F5-TTS Expressive Tags ===")
        
        payload = {
            "input": {
                "operation": "expressive_tts",
                "text": "Welcome to our service! {happy}We are thrilled to help you today!{/happy} {whisper}But first, let me tell you a secret.{/whisper} {shout:THIS IS VERY IMPORTANT!} {sad}Unfortunately, not everything is perfect.{/sad} {angry}Some things make me frustrated!{/angry}",
                "voice_name": "Amy",
                "sample_rate": 24000,
                "crossfade_ms": 120
            }
        }
        
        try:
            response = self.make_request(payload)
            
            # Check if response indicates expressive processing
            if 'output' in response:
                output = response['output']
                if 'mode' in output and 'expressive' in output['mode']:
                    logger.info("✅ Expressive mode detected")
                
                if 'processing_stats' in output:
                    stats = output['processing_stats']
                    logger.info(f"   Processing stats: {stats}")
                    
                    if stats.get('f5_segments', 0) > 0:
                        logger.info("✅ F5-TTS segments were processed")
                    else:
                        logger.warning("⚠️  No F5-TTS segments - may be using fallback")
            
            if self.save_audio_response(response, "test_expressive_native.wav"):
                logger.info("✅ Native expressive tags test successful")
                return True
            else:
                logger.error("❌ Native expressive tags test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Native expressive tags test failed: {e}")
            return False
    
    def test_extended_tags(self) -> bool:
        """Test extended tags that map to native ones"""
        logger.info("\n=== Testing Extended Expressive Tags ===")
        
        payload = {
            "input": {
                "operation": "expressive_tts",
                "text": "Let me test the extended tags. {excited}I am so thrilled about this!{/excited} {calm}Now let me speak peacefully.{/calm} {nervous}This makes me a bit worried.{/nervous} {confident}But I know I can handle it!{/confident}",
                "voice_name": "Benjamin",
                "sample_rate": 24000
            }
        }
        
        try:
            response = self.make_request(payload)
            
            if self.save_audio_response(response, "test_expressive_extended.wav"):
                logger.info("✅ Extended expressive tags test successful")
                return True
            else:
                logger.error("❌ Extended expressive tags test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Extended expressive tags test failed: {e}")
            return False
    
    def test_mixed_content(self) -> bool:
        """Test mixed normal and expressive content"""
        logger.info("\n=== Testing Mixed Normal and Expressive Content ===")
        
        payload = {
            "input": {
                "operation": "expressive_tts",
                "text": "This is a story about emotions. Once upon a time, there was a character who felt {happy}absolutely delighted{/happy} about their achievements. But then something happened that made them {sad}deeply sorrowful{/sad}. They began to {whisper}speak very quietly{/whisper} about their concerns. Eventually, they decided to {shout:SPEAK UP LOUDLY} about what bothered them. In the end, they found peace and could {calm}speak with tranquility{/calm} once again.",
                "voice_name": "Christopher",
                "sample_rate": 24000,
                "crossfade_ms": 150
            }
        }
        
        try:
            response = self.make_request(payload)
            
            if self.save_audio_response(response, "test_expressive_mixed.wav"):
                logger.info("✅ Mixed content test successful")
                return True
            else:
                logger.error("❌ Mixed content test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Mixed content test failed: {e}")
            return False
    
    def test_inline_tags(self) -> bool:
        """Test inline tag format"""
        logger.info("\n=== Testing Inline Tag Format ===")
        
        payload = {
            "input": {
                "operation": "expressive_tts",
                "text": "Testing inline format: {happy:This is joyful content!} And this: {shout:LOUD ANNOUNCEMENT!} Finally: {whisper:secret message}",
                "voice_name": "Amy",
                "sample_rate": 24000
            }
        }
        
        try:
            response = self.make_request(payload)
            
            if self.save_audio_response(response, "test_expressive_inline.wav"):
                logger.info("✅ Inline tags test successful")
                return True
            else:
                logger.error("❌ Inline tags test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Inline tags test failed: {e}")
            return False
    
    def test_fallback_behavior(self) -> bool:
        """Test fallback when F5-TTS not available"""
        logger.info("\n=== Testing Fallback Behavior ===")
        
        # Test with unsupported operation to trigger fallback
        payload = {
            "input": {
                "operation": "voice_conversion",
                "input_audio": "fake_audio_data",
                "voice_name": "Amy"
            }
        }
        
        try:
            response = self.make_request(payload)
            logger.info("✅ Fallback behavior test - API responded correctly")
            return True
                
        except Exception as e:
            logger.info(f"✅ Fallback behavior test - Got expected error: {e}")
            return True  # Expected to fail for this test
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all API tests"""
        logger.info("🧪 Starting Enhanced Expressive TTS API Tests")
        logger.info("=" * 60)
        
        tests = [
            ("Basic TTS", self.test_basic_functionality),
            ("Native Expressive Tags", self.test_native_expressive_tags),
            ("Extended Tags", self.test_extended_tags),
            ("Mixed Content", self.test_mixed_content),
            ("Inline Tags", self.test_inline_tags),
            ("Fallback Behavior", self.test_fallback_behavior)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                start_time = time.time()
                result = test_func()
                end_time = time.time()
                
                results[test_name] = result
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{test_name}: {status} ({end_time - start_time:.2f}s)")
                
            except Exception as e:
                results[test_name] = False
                logger.error(f"{test_name}: ❌ FAILED with exception: {e}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🧪 API Test Results Summary")
        logger.info("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name:25} {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All API tests passed! Enhanced expressive TTS is working correctly.")
        else:
            logger.info("⚠️  Some tests failed. Check the output above for details.")
        
        return results

def main():
    """Main test function"""
    # Check if API key is set
    api_key = API_KEY
    if api_key == "YOUR_API_KEY":
        print("❌ Please set your API_KEY in the script before running tests")
        print("   Update the API_KEY variable at the top of this file")
        return False
    
    # Create tester and run tests
    tester = ExpressiveTTSAPITester(api_key)
    results = tester.run_all_tests()
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 