#!/usr/bin/env python3
"""
Temporary handler modification to test optimization performance via API
Add this to your handler.py temporarily to test the optimization
"""

# Add this to your handler.py temporarily:

def test_optimization_performance():
    """Test optimization performance - to be called via API"""
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    results = {
        "optimization_test": "performance_comparison",
        "results": {}
    }
    
    try:
        # Test optimized approach
        logger.info("Testing optimized voice loading...")
        from optimized_voice_library import initialize_optimized_voice_library
        
        optimized_lib = initialize_optimized_voice_library()
        if optimized_lib.is_available():
            # Test loading a voice multiple times
            optimized_times = []
            voice_name = "Amy"
            
            for i in range(3):
                start_time = time.time()
                audio_array = optimized_lib.get_voice_audio_by_name_direct(voice_name)
                end_time = time.time()
                
                if audio_array is not None:
                    load_time = end_time - start_time
                    optimized_times.append(load_time)
                    logger.info(f"Optimized run {i+1}: {load_time:.4f}s - {audio_array.shape} samples")
            
            avg_optimized = sum(optimized_times) / len(optimized_times) if optimized_times else 0
            results["results"]["optimized"] = {
                "times": optimized_times,
                "average": avg_optimized,
                "status": "success"
            }
            
        else:
            results["results"]["optimized"] = {"status": "library_not_available"}
        
        # Test legacy approach
        logger.info("Testing legacy voice loading...")
        from local_voice_library import initialize_local_voice_library
        import tempfile
        import soundfile as sf
        
        legacy_lib = initialize_local_voice_library()
        if legacy_lib.is_available():
            legacy_times = []
            
            for i in range(3):
                start_time = time.time()
                
                # Simulate full legacy process
                audio_array = legacy_lib.get_voice_audio_by_name(voice_name)
                if audio_array is not None:
                    # Save to temp file (current process)
                    temp_path = tempfile.mktemp(suffix='.wav')
                    sf.write(temp_path, audio_array, 24000)
                    
                    end_time = time.time()
                    load_time = end_time - start_time
                    legacy_times.append(load_time)
                    
                    # Cleanup
                    try:
                        import os
                        os.unlink(temp_path)
                    except:
                        pass
                    
                    logger.info(f"Legacy run {i+1}: {load_time:.4f}s - {audio_array.shape} samples")
            
            avg_legacy = sum(legacy_times) / len(legacy_times) if legacy_times else 0
            results["results"]["legacy"] = {
                "times": legacy_times,
                "average": avg_legacy,
                "status": "success"
            }
            
        else:
            results["results"]["legacy"] = {"status": "library_not_available"}
        
        # Calculate improvement
        if results["results"].get("optimized", {}).get("average") and results["results"].get("legacy", {}).get("average"):
            avg_optimized = results["results"]["optimized"]["average"]
            avg_legacy = results["results"]["legacy"]["average"]
            
            improvement = ((avg_legacy - avg_optimized) / avg_legacy) * 100
            speedup = avg_legacy / avg_optimized
            
            results["performance_analysis"] = {
                "improvement_percentage": improvement,
                "speedup_factor": speedup,
                "faster": improvement > 0
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Optimization test failed: {e}")
        return {
            "optimization_test": "performance_comparison",
            "error": str(e),
            "status": "failed"
        }

# Add this to the handler function in handler.py:

# Add this inside your handler function, after the other operation checks:
elif operation == 'test_optimization_performance':
    return test_optimization_performance() 