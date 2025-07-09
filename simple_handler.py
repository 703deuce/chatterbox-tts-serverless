import runpod
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_handler(job):
    """Simplified handler for debugging"""
    try:
        job_input = job.get('input', {})
        operation = job_input.get('operation', 'tts')
        
        logger.info(f"Received job: {job_input}")
        
        # Just return a simple response to test connectivity
        if operation == 'tts':
            return {
                "message": "Handler is working!",
                "text": job_input.get('text', 'No text provided'),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "operation": operation
            }
        else:
            return {"error": f"Unknown operation: {operation}"}
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": str(e)}

# Start the handler
if __name__ == "__main__":
    logger.info("Starting simple test handler...")
    runpod.serverless.start({"handler": simple_handler}) 