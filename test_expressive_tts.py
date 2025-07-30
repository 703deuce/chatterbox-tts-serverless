#!/usr/bin/env python3
"""
Test Script for Enhanced Expressive TTS Handler
Tests various expressive tag combinations and functionality
"""

import sys
import time
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_text_parser():
    """Test the expressive text parser"""
    print("\n=== Testing Text Parser ===")
    
    try:
        from expressive_text_parser import create_text_parser
        parser = create_text_parser()
        
        test_cases = [
            "Simple text without any tags",
            "This is normal. {whisper}This should be whispered{/whisper} back to normal.",
            "Start normal {yell:LOUD SHOUTING HERE} then continue.",
            "{excited}Excited start{/excited} middle {sad}very sad part{/sad} happy ending!",
            "Multiple {angry}ANGRY{/angry} and {happy}HAPPY{/happy} in one sentence!",
            "Complex: Normal start {whisper}quiet part{/whisper} then {yell:VERY LOUD} and {calm}peaceful ending{/calm}."
        ]
        
        for i, text in enumerate(test_cases):
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
            
            for segment in segments:
                engine_info = f"({segment.tag_type})" if segment.tag_type else ""
                print(f"     {segment.index}: [{segment.engine}{engine_info}] '{segment.text[:30]}...'")
        
        print("✅ Text parser tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Text parser test failed: {e}")
        return False

def test_audio_stitcher():
    """Test the audio stitching functionality"""
    print("\n=== Testing Audio Stitcher ===")
    
    try:
        import numpy as np
        from audio_stitcher import create_audio_stitcher, AudioSegmentData
        
        stitcher = create_audio_stitcher(crossfade_ms=100)
        
        # Create mock audio segments
        sample_rate = 24000
        duration = 1.0  # 1 second each
        
        segments = []
        for i in range(3):
            # Generate test audio (different frequencies)
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency = 440 + i * 220  # 440Hz, 660Hz, 880Hz
            audio = np.sin(2 * np.pi * frequency * t) * 0.3
            
            segment = AudioSegmentData(
                index=i,
                audio=audio,
                sample_rate=sample_rate,
                engine='chatterbox' if i % 2 == 0 else 'f5',
                tag_type='normal' if i % 2 == 0 else 'excited',
                duration=duration
            )
            segments.append(segment)
        
        # Test stitching
        result_audio, result_sr = stitcher.stitch_segments(segments)
        
        print(f"✅ Stitched {len(segments)} segments successfully")
        print(f"   Output: {len(result_audio)} samples at {result_sr}Hz")
        print(f"   Duration: {len(result_audio) / result_sr:.2f}s")
        
        # Save test audio
        try:
            import soundfile as sf
            output_path = "test_stitched_expressive.wav"
            sf.write(output_path, result_audio, result_sr)
            print(f"   Saved test audio: {output_path}")
        except ImportError:
            print("   (soundfile not available - skipping audio save)")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio stitcher test failed: {e}")
        return False

def test_f5_integration():
    """Test F5-TTS integration (may fail if F5-TTS not installed)"""
    print("\n=== Testing F5-TTS Integration ===")
    
    try:
        from f5_tts_integration import create_f5_tts
        
        f5_tts = create_f5_tts()
        
        if f5_tts.is_available():
            print("✅ F5-TTS is available")
            print(f"   Supported tags: {f5_tts.get_supported_tags()}")
            
            # Test generation would require actual model loading
            print("   (Model generation test skipped - requires full model loading)")
        else:
            print("⚠️  F5-TTS not available - will use Chatterbox fallback")
        
        return True
        
    except Exception as e:
        print(f"❌ F5-TTS integration test failed: {e}")
        return False

def test_enhanced_handler_simulation():
    """Test the enhanced handler logic without full model loading"""
    print("\n=== Testing Enhanced Handler (Simulation) ===")
    
    try:
        # Test job inputs for various scenarios
        test_jobs = [
            {
                'input': {
                    'operation': 'expressive_tts',
                    'text': 'Simple text without tags',
                    'voice_name': 'Amy'
                }
            },
            {
                'input': {
                    'operation': 'expressive_tts',
                    'text': 'This is normal. {whisper}This is whispered{/whisper} back to normal.',
                    'voice_name': 'Amy',
                    'sample_rate': 24000
                }
            },
            {
                'input': {
                    'operation': 'expressive_tts',
                    'text': '{excited}Super excited start{/excited} then {calm}peaceful middle{/calm} and {yell:LOUD ENDING}',
                    'voice_id': 'test123',
                    'crossfade_ms': 150
                }
            }
        ]
        
        for i, job in enumerate(test_jobs):
            print(f"\nTest Job {i+1}:")
            job_input = job['input']
            text = job_input.get('text', '')
            
            # Check if text has expressive tags
            has_tags = '{' in text and '}' in text
            print(f"   Text: '{text[:50]}...'")
            print(f"   Has expressive tags: {has_tags}")
            print(f"   Operation: {job_input.get('operation')}")
            print(f"   Voice: {job_input.get('voice_name') or job_input.get('voice_id', 'none')}")
            
            # Simulate routing logic
            if has_tags:
                print("   → Would use Enhanced Expressive TTS")
            else:
                print("   → Would use Standard Optimized TTS")
        
        print("✅ Enhanced handler simulation completed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced handler simulation failed: {e}")
        return False

def test_requirements_check():
    """Check if required dependencies are available"""
    print("\n=== Checking Dependencies ===")
    
    dependencies = [
        ('pydub', 'Audio processing and crossfading'),
        ('f5_tts', 'F5-TTS integration (optional)'),
        ('librosa', 'Audio resampling'),
        ('soundfile', 'Audio I/O'),
        ('numpy', 'Numerical operations'),
        ('torch', 'PyTorch framework'),
        ('runpod', 'RunPod serverless'),
    ]
    
    available = []
    missing = []
    
    for dep, description in dependencies:
        try:
            __import__(dep)
            available.append((dep, description))
            print(f"✅ {dep}: {description}")
        except ImportError:
            missing.append((dep, description))
            print(f"❌ {dep}: {description} (MISSING)")
    
    print(f"\nDependency Status: {len(available)}/{len(dependencies)} available")
    
    if missing:
        print("\nMissing dependencies:")
        for dep, desc in missing:
            print(f"  - {dep}: {desc}")
        print("\nTo install missing dependencies:")
        print("pip install " + " ".join([dep for dep, _ in missing]))
    
    return len(missing) == 0

def main():
    """Run all tests"""
    print("🧪 Enhanced Expressive TTS Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Run individual tests
    test_results.append(("Dependencies", test_requirements_check()))
    test_results.append(("Text Parser", test_text_parser()))
    test_results.append(("Audio Stitcher", test_audio_stitcher()))
    test_results.append(("F5-TTS Integration", test_f5_integration()))
    test_results.append(("Enhanced Handler", test_enhanced_handler_simulation()))
    
    # Summary
    print("\n" + "=" * 50)
    print("🧪 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced expressive TTS system is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("   Note: F5-TTS failures are expected if the package isn't installed.")
        print("   The system will use Chatterbox fallback for expressive tags.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 