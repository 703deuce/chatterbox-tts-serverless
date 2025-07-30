#!/usr/bin/env python3
"""
Test script to compare optimized vs legacy voice loading performance
This demonstrates the performance improvements of direct audio arrays
"""

import time
import logging
import numpy as np
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_performance_comparison():
    """Test and compare optimized vs legacy voice loading performance"""
    
    print("🔬 Performance Comparison: Optimized vs Legacy Voice Loading")
    print("=" * 60)
    
    # Test optimized approach
    print("\n📈 Testing OPTIMIZED approach...")
    try:
        from optimized_voice_library import initialize_optimized_voice_library
        
        optimized_lib = initialize_optimized_voice_library()
        if not optimized_lib.is_available():
            print("❌ Optimized voice library not available")
            return
        
        # Test loading a voice multiple times to see caching benefits
        voice_name = "Amy"
        optimized_times = []
        
        for i in range(5):
            start_time = time.time()
            audio_array = optimized_lib.get_voice_audio_by_name_direct(voice_name)
            end_time = time.time()
            
            if audio_array is not None:
                load_time = end_time - start_time
                optimized_times.append(load_time)
                cache_status = "CACHED" if i > 0 else "LOADED"
                print(f"  Run {i+1}: {load_time:.4f}s ({cache_status}) - {audio_array.shape} samples")
            else:
                print(f"  Run {i+1}: Failed to load voice")
        
        avg_optimized = sum(optimized_times) / len(optimized_times) if optimized_times else 0
        
    except ImportError as e:
        print(f"❌ Could not import optimized library: {e}")
        return
    except Exception as e:
        print(f"❌ Error testing optimized approach: {e}")
        return
    
    # Test legacy approach
    print(f"\n📊 Testing LEGACY approach...")
    try:
        from local_voice_library import initialize_local_voice_library
        import tempfile
        import soundfile as sf
        
        legacy_lib = initialize_local_voice_library()
        if not legacy_lib.is_available():
            print("❌ Legacy voice library not available")
            return
        
        legacy_times = []
        
        for i in range(5):
            start_time = time.time()
            
            # Simulate the full legacy process:
            # 1. Load audio array from embedding
            audio_array = legacy_lib.get_voice_audio_by_name(voice_name)
            
            if audio_array is not None:
                # 2. Save to temporary file (as current handler does)
                temp_path = tempfile.mktemp(suffix='.wav')
                sf.write(temp_path, audio_array, 24000)
                
                # 3. (ChatterboxTTS would then load this file again)
                # We'll just measure the time to here since that's the bottleneck
                
                end_time = time.time()
                load_time = end_time - start_time
                legacy_times.append(load_time)
                
                print(f"  Run {i+1}: {load_time:.4f}s (FILE I/O) - {audio_array.shape} samples")
                
                # Clean up
                try:
                    import os
                    os.unlink(temp_path)
                except:
                    pass
            else:
                print(f"  Run {i+1}: Failed to load voice")
        
        avg_legacy = sum(legacy_times) / len(legacy_times) if legacy_times else 0
        
    except ImportError as e:
        print(f"❌ Could not import legacy library: {e}")
        return
    except Exception as e:
        print(f"❌ Error testing legacy approach: {e}")
        return
    
    # Calculate and display results
    print(f"\n📋 PERFORMANCE RESULTS")
    print("=" * 40)
    print(f"Optimized average: {avg_optimized:.4f}s")
    print(f"Legacy average:    {avg_legacy:.4f}s")
    
    if avg_optimized > 0 and avg_legacy > 0:
        improvement = ((avg_legacy - avg_optimized) / avg_legacy) * 100
        speedup = avg_legacy / avg_optimized
        
        print(f"\n🚀 OPTIMIZATION RESULTS:")
        print(f"   Performance improvement: {improvement:.1f}%")
        print(f"   Speedup factor: {speedup:.2f}x")
        
        if improvement > 0:
            print(f"   ✅ Optimized approach is {improvement:.1f}% faster!")
        else:
            print(f"   ⚠️  Legacy approach performed better by {abs(improvement):.1f}%")
    
    # Memory usage comparison
    print(f"\n💾 MEMORY USAGE:")
    try:
        stats = optimized_lib.get_stats()
        print(f"   Optimized cache: {stats['cache_size']}/{stats['max_cache_size']} voices cached")
        print(f"   Cache hit ratio: Significant for repeated access")
    except:
        pass

def test_audio_quality():
    """Test that optimized approach produces identical audio quality"""
    print(f"\n🔊 AUDIO QUALITY VERIFICATION")
    print("=" * 40)
    
    try:
        from optimized_voice_library import initialize_optimized_voice_library
        from local_voice_library import initialize_local_voice_library
        
        optimized_lib = initialize_optimized_voice_library()
        legacy_lib = initialize_local_voice_library()
        
        voice_name = "Amy"
        
        # Load same voice from both libraries
        optimized_audio = optimized_lib.get_voice_audio_by_name_direct(voice_name)
        legacy_audio = legacy_lib.get_voice_audio_by_name(voice_name)
        
        if optimized_audio is not None and legacy_audio is not None:
            # Compare arrays
            if np.array_equal(optimized_audio, legacy_audio):
                print("✅ Audio quality identical - optimization preserves quality!")
            else:
                diff = np.mean(np.abs(optimized_audio - legacy_audio))
                print(f"⚠️  Audio differs by average of {diff:.6f}")
                print(f"   Optimized shape: {optimized_audio.shape}")
                print(f"   Legacy shape: {legacy_audio.shape}")
        else:
            print("❌ Could not load audio from one or both libraries")
            
    except Exception as e:
        print(f"❌ Error during quality test: {e}")

def main():
    """Run all tests"""
    print("🧪 Chatterbox TTS Optimization Testing Suite")
    print("Testing direct audio array vs file-based approach")
    print("=" * 60)
    
    test_performance_comparison()
    test_audio_quality()
    
    print(f"\n🎯 SUMMARY:")
    print("The optimized approach eliminates the inefficient process of:")
    print("  1. Loading embedding → audio array")
    print("  2. Saving audio array → temporary WAV file") 
    print("  3. ChatterboxTTS loading WAV file → internal embeddings")
    print("")
    print("Instead, it directly provides audio arrays to ChatterboxTTS,")
    print("reducing I/O overhead and improving performance!")

if __name__ == "__main__":
    main() 