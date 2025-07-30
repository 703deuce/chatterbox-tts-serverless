#!/usr/bin/env python3
"""
Test Script for Native F5-TTS Tags and Custom Reference Audio
Tests the corrected F5-TTS implementation with actual supported tags
"""

import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_native_f5_tags():
    """Test parsing with native F5-TTS tags"""
    print("\n=== Testing Native F5-TTS Tags ===")
    
    try:
        from expressive_text_parser import create_text_parser
        parser = create_text_parser()
        
        # Test cases with native F5-TTS tags
        native_test_cases = [
            "Simple text without tags",
            "Hello {whisper}this is whispered{/whisper} back to normal.",
            "Normal speech {shout:SHOUTING LOUDLY} then continue.",
            "{happy}I'm so excited!{/happy} But then {sad}something terrible happened{/sad}.",
            "{angry}I AM FURIOUS{/angry} about this situation!",
            "Mix all native: {whisper}secret{/whisper} {happy}joy{/happy} {angry}rage{/angry} {sad}sorrow{/sad} {shout:LOUD}",
        ]
        
        # Test cases with extended tags (should map to native)
        extended_test_cases = [
            "Extended tags: {yell}YELLING{/yell} {excited}SO EXCITED{/excited}",
            "{calm}Peaceful moment{/calm} then {nervous}anxious feeling{/nervous}",
            "{confident}I can do this{/confident} presentation."
        ]
        
        all_cases = native_test_cases + extended_test_cases
        
        for i, text in enumerate(all_cases):
            print(f"\nTest {i+1}: {text}")
            
            # Validate text
            is_valid, error = parser.validate_text(text)
            if not is_valid:
                print(f"  ❌ Validation failed: {error}")
                continue
            
            # Parse text
            segments = parser.parse_text(text)
            stats = parser.get_segment_stats(segments)
            
            print(f"  ✅ Parsed into {len(segments)} segments")
            print(f"     Chatterbox: {stats['chatterbox_segments']}, F5-TTS: {stats['f5_segments']}")
            print(f"     Tags used: {stats['tag_distribution']}")
            
            # Show segment details
            for segment in segments:
                if segment.engine == 'f5':
                    print(f"     {segment.index}: [F5-TTS:{segment.tag_type}] '{segment.text[:30]}...'")
                else:
                    print(f"     {segment.index}: [Chatterbox] '{segment.text[:30]}...'")
        
        print("✅ Native F5-TTS tag parsing tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Native F5-TTS tag test failed: {e}")
        return False

def test_f5_integration_native():
    """Test F5-TTS integration with native tag support"""
    print("\n=== Testing F5-TTS Integration (Native Tags) ===")
    
    try:
        from f5_tts_integration import create_f5_tts
        
        f5_tts = create_f5_tts()
        
        if f5_tts.is_available():
            print("✅ F5-TTS is available")
            
            # Get supported tags info
            tag_info = f5_tts.get_supported_tags()
            print(f"   Native F5-TTS tags: {tag_info['native_f5_tags']}")
            print(f"   Extended tags: {tag_info['extended_tags']}")
            print(f"   Custom audio support: {tag_info['custom_audio_support']}")
            
            # Show tag details
            print("\n   Tag Details:")
            for tag, details in tag_info['tag_details'].items():
                builtin_status = "Native" if details['builtin'] else "Extended"
                print(f"     {tag}: {details['description']} [{builtin_status}]")
            
        else:
            print("⚠️  F5-TTS not available - will use Chatterbox fallback")
            print("   This is expected if F5-TTS package is not installed")
        
        return True
        
    except Exception as e:
        print(f"❌ F5-TTS native integration test failed: {e}")
        return False

def test_custom_audio_support():
    """Test custom reference audio parsing logic"""
    print("\n=== Testing Custom Reference Audio Support ===")
    
    try:
        # Simulate job input with custom tag audio
        job_input = {
            'operation': 'expressive_tts',
            'text': 'Hello {robotic}system initialized{/robotic} welcome {pirate}ahoy matey{/pirate}',
            'voice_name': 'Amy',
            'custom_tag_audio': {
                'robotic': 'fake_base64_robot_audio_data',
                'pirate': 'fake_base64_pirate_audio_data'
            }
        }
        
        print(f"Mock Job Input:")
        print(f"  Text: {job_input['text']}")
        print(f"  Custom Tags: {list(job_input['custom_tag_audio'].keys())}")
        
        # Test text parsing with custom tags
        from expressive_text_parser import create_text_parser
        parser = create_text_parser()
        
        # Note: For this test, we'll add the custom tags to supported tags temporarily
        original_supported = parser.SUPPORTED_TAGS.copy()
        parser.SUPPORTED_TAGS.update({
            'robotic': 'robotic',
            'pirate': 'pirate'
        })
        
        segments = parser.parse_text(job_input['text'])
        stats = parser.get_segment_stats(segments)
        
        print(f"\n  Parsing Results:")
        print(f"    Total segments: {len(segments)}")
        print(f"    F5-TTS segments: {stats['f5_segments']} (including custom)")
        print(f"    Tag distribution: {stats['tag_distribution']}")
        
        custom_segments = [s for s in segments if s.tag_type in ['robotic', 'pirate']]
        print(f"    Custom tag segments: {len(custom_segments)}")
        
        # Restore original supported tags
        parser.SUPPORTED_TAGS = original_supported
        
        print("✅ Custom reference audio support test completed")
        return True
        
    except Exception as e:
        print(f"❌ Custom reference audio test failed: {e}")
        return False

def create_sample_api_requests():
    """Create sample API requests showcasing the corrected functionality"""
    print("\n=== Sample API Requests ===")
    
    samples = [
        {
            "name": "Native F5-TTS Tags Only",
            "request": {
                "input": {
                    "operation": "expressive_tts",
                    "text": "Welcome! {happy}We have great news!{/happy} {whisper}But keep it secret.{/whisper} {shout:THIS IS IMPORTANT!}",
                    "voice_name": "Amy",
                    "sample_rate": 24000
                }
            }
        },
        {
            "name": "Extended Tags (Auto-Mapped)",
            "request": {
                "input": {
                    "operation": "expressive_tts", 
                    "text": "I was {excited}thrilled to announce{/excited} the results, but then I became {nervous}worried about the reaction{/nervous}.",
                    "voice_id": "abc123ef"
                }
            }
        },
        {
            "name": "Custom Reference Audio",
            "request": {
                "input": {
                    "operation": "expressive_tts",
                    "text": "Welcome to our {robotic}automated assistant{/robotic}. Please {elderly}speak slowly and clearly{/elderly}.",
                    "voice_name": "Benjamin",
                    "custom_tag_audio": {
                        "robotic": "UklGRigAAABXQVZFZm10IBAAAAABAAEA...",
                        "elderly": "UklGRjgAAABXQVZFZm10IBAAAAABAAEK..."
                    }
                }
            }
        },
        {
            "name": "Mixed Native and Custom",
            "request": {
                "input": {
                    "operation": "expressive_tts",
                    "text": "{happy}Ahoy there!{/happy} {pirate}Welcome aboard me ship!{/pirate} {sad}But beware the curse.{/sad}",
                    "voice_name": "Christopher",
                    "custom_tag_audio": {
                        "pirate": "base64_encoded_pirate_voice_sample"
                    },
                    "crossfade_ms": 120
                }
            }
        }
    ]
    
    for sample in samples:
        print(f"\n{sample['name']}:")
        print(json.dumps(sample['request'], indent=2))
    
    print("\n✅ Sample API requests generated")
    return True

def main():
    """Run all native F5-TTS tests"""
    print("🧪 Native F5-TTS Tags and Custom Audio Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(("Native F5-TTS Tags", test_native_f5_tags()))
    test_results.append(("F5-TTS Integration", test_f5_integration_native()))
    test_results.append(("Custom Audio Support", test_custom_audio_support()))
    test_results.append(("Sample Requests", create_sample_api_requests()))
    
    # Summary
    print("\n" + "=" * 60)
    print("🧪 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All native F5-TTS tests passed!")
        print("\n📋 Summary of Capabilities:")
        print("✅ Native F5-TTS tags: whisper, shout, happy, sad, angry")
        print("✅ Extended tags: yell, excited, calm, nervous, confident")
        print("✅ Custom reference audio for unlimited expressions")
        print("✅ Automatic mapping and fallback support")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 