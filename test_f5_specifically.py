#!/usr/bin/env python3
"""
Test F5-TTS Specific Functionality
Tests only the expressive tags that should use F5-TTS
"""

import requests
import base64
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RunPod API configuration
RUNPOD_API_URL = "https://api.runpod.ai/v2/c2wmx1ln5ccp6c/run"
API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def poll_job_result(job_id, timeout=120):
    """Poll RunPod job until completion"""
    status_url = f"https://api.runpod.ai/v2/c2wmx1ln5ccp6c/status/{job_id}"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(status_url, headers=HEADERS)
            result = response.json()
            
            status = result.get('status', 'UNKNOWN')
            logger.info(f"Job status: {status} (waiting... {int(time.time() - start_time)}s)")
            
            if status == 'COMPLETED':
                elapsed = int(time.time() - start_time)
                logger.info(f"Job completed successfully after {elapsed} seconds")
                return result.get('output', {})
            elif status == 'FAILED':
                error = result.get('error', 'Unknown error')
                logger.error(f"Job failed: {error}")
                return None
                
        except Exception as e:
            logger.error(f"Error polling job: {e}")
        
        time.sleep(3)
    
    logger.error(f"Job timed out after {timeout} seconds")
    return None

def test_f5_expressive_tags():
    """Test F5-TTS with native expressive tags"""
    logger.info("🧪 Testing F5-TTS Native Expressive Tags")
    logger.info("=" * 60)
    
    # Test with ONLY F5-TTS native tags (no regular text)
    payload = {
        "input": {
            "operation": "expressive_tts",
            "text": "{whisper}This should be whispered.{/whisper} {happy}This should sound happy!{/happy} {shout}THIS SHOULD BE LOUD!{/shout}",
            "voice_name": "Benjamin",
            "sample_rate": 24000
        }
    }
    
    logger.info(f"Payload: {payload}")
    
    try:
        # Submit job
        response = requests.post(RUNPOD_API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        
        result = response.json()
        job_id = result.get('id')
        
        if not job_id:
            logger.error(f"No job ID returned: {result}")
            return False
            
        logger.info(f"Job submitted with ID: {job_id}")
        
        # Poll for result
        output = poll_job_result(job_id)
        
        if not output:
            logger.error("❌ Failed to get job result")
            return False
            
        # Check if F5-TTS was actually used
        if 'processing_stats' in output:
            stats = output['processing_stats']
            engines_used = stats.get('engines_used', [])
            segment_breakdown = stats.get('segment_breakdown', {})
            
            logger.info(f"✅ Processing successful!")
            logger.info(f"   Engines used: {engines_used}")
            logger.info(f"   Segment breakdown: {segment_breakdown}")
            
            # Check if F5-TTS was actually used
            f5_segments = segment_breakdown.get('f5_segments', 0)
            chatterbox_segments = segment_breakdown.get('chatterbox_segments', 0)
            
            if f5_segments > 0 and 'f5' in engines_used:
                logger.info(f"🎉 F5-TTS IS WORKING! Processed {f5_segments} segments")
                
                # Save audio
                if 'audio' in output:
                    audio_data = base64.b64decode(output['audio'])
                    with open('test_f5_expressive.wav', 'wb') as f:
                        f.write(audio_data)
                    logger.info(f"🎵 Audio saved to test_f5_expressive.wav ({len(audio_data)} bytes)")
                
                return True
            else:
                logger.warning(f"⚠️  F5-TTS not used - all {chatterbox_segments} segments went to Chatterbox")
                logger.warning("F5-TTS may not be properly installed or configured")
                return False
        else:
            logger.error("❌ No processing stats returned")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_f5_single_tag():
    """Test with a single F5-TTS tag to isolate the issue"""
    logger.info("\n🧪 Testing Single F5-TTS Tag")
    logger.info("=" * 40)
    
    payload = {
        "input": {
            "operation": "expressive_tts", 
            "text": "{whisper}this is a whispered test{/whisper}",
            "sample_rate": 24000
        }
    }
    
    logger.info(f"Testing single whisper tag...")
    
    try:
        response = requests.post(RUNPOD_API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        
        result = response.json()
        job_id = result.get('id')
        logger.info(f"Job ID: {job_id}")
        
        output = poll_job_result(job_id)
        
        if output and 'processing_stats' in output:
            stats = output['processing_stats']
            engines_used = stats.get('engines_used', [])
            logger.info(f"Engines used: {engines_used}")
            
            if 'f5' in engines_used:
                logger.info("✅ F5-TTS working for single tag!")
                return True
            else:
                logger.warning("⚠️  Single tag also fell back to Chatterbox")
                return False
        
    except Exception as e:
        logger.error(f"Single tag test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 F5-TTS Specific Testing")
    logger.info("=" * 50)
    
    # Test 1: Multiple F5-TTS tags
    success1 = test_f5_expressive_tags()
    
    # Test 2: Single F5-TTS tag
    success2 = test_f5_single_tag()
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 F5-TTS Test Results:")
    logger.info(f"Multiple Tags: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"Single Tag:   {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 or success2:
        logger.info("🎉 F5-TTS IS WORKING!")
    else:
        logger.info("❌ F5-TTS is not working - check installation") 