#!/usr/bin/env python3
"""
Enhanced Handler with Expressive Tag Support
Combines F5-TTS and Chatterbox for seamless expressive TTS
"""

import runpod
import torch
import base64
import io
import logging
import time
import soundfile as sf
import librosa
import numpy as np
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

# Import existing optimized components
from optimized_chatterbox import OptimizedChatterboxTTS, create_optimized_tts
from optimized_voice_library import OptimizedVoiceLibrary, initialize_optimized_voice_library

# Import new expressive components
from expressive_text_parser import ExpressiveTextParser, TextSegment, create_text_parser
from f5_tts_integration import F5TTSWrapper, create_f5_tts
from audio_stitcher import AudioStitcher, AudioSegmentData, create_audio_stitcher

# Import legacy components for fallback
from local_voice_library import LocalVoiceLibrary
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.s3gen import S3GEN_SR, S3Gen

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result from segment processing"""
    index: int
    audio: np.ndarray
    sample_rate: int
    engine: str
    tag_type: Optional[str]
    processing_time: float
    success: bool
    error: Optional[str] = None

class EnhancedTTSHandler:
    """Enhanced TTS Handler with expressive tag support"""
    
    def __init__(self, device: str = "auto"):
        # Device setup
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Model instances
        self.chatterbox_model: Optional[OptimizedChatterboxTTS] = None
        self.f5_model: Optional[F5TTSWrapper] = None
        self.voice_library: Optional[OptimizedVoiceLibrary] = None
        self.legacy_voice_library: Optional[LocalVoiceLibrary] = None
        self.s3gen_model: Optional[S3Gen] = None
        
        # Processing components
        self.text_parser: Optional[ExpressiveTextParser] = None
        self.audio_stitcher: Optional[AudioStitcher] = None
        
        # Configuration
        self.max_parallel_segments = 4  # Limit concurrent processing
        self.crossfade_ms = 100  # Default crossfade duration
        
        # State tracking
        self.models_loaded = False
        
    def load_models(self):
        """Load all required models and components"""
        if self.models_loaded:
            return
        
        logger.info(f"Loading enhanced TTS models on device: {self.device}")
        start_time = time.time()
        
        try:
            # Load text parser
            logger.info("Initializing text parser...")
            self.text_parser = create_text_parser()
            
            # Load audio stitcher
            logger.info("Initializing audio stitcher...")
            self.audio_stitcher = create_audio_stitcher(
                crossfade_ms=self.crossfade_ms,
                normalize_segments=True
            )
            
            # Load Chatterbox TTS (optimized)
            logger.info("Loading OptimizedChatterboxTTS...")
            self.chatterbox_model = create_optimized_tts(device=self.device)
            logger.info("OptimizedChatterboxTTS loaded successfully")
            
            # Load F5-TTS
            logger.info("Loading F5-TTS...")
            self.f5_model = create_f5_tts(device=self.device)
            if self.f5_model.is_available():
                logger.info("F5-TTS loaded successfully")
            else:
                logger.warning("F5-TTS not available - expressive tags will use Chatterbox")
            
            # Load voice library
            logger.info("Loading voice library...")
            self.voice_library = initialize_optimized_voice_library()
            if self.voice_library and self.voice_library.is_available():
                stats = self.voice_library.get_stats()
                logger.info(f"Voice library loaded with {stats['total_voices']} voices")
            else:
                logger.warning("Optimized voice library not available")
            
            # Load legacy voice library as fallback
            try:
                from local_voice_library import initialize_local_voice_library
                self.legacy_voice_library = initialize_local_voice_library()
                logger.info("Legacy voice library loaded as fallback")
            except Exception as e:
                logger.warning(f"Legacy voice library failed to load: {e}")
            
            # Load S3Gen model for voice conversion
            logger.info("Loading S3Gen model...")
            self.s3gen_model = S3Gen()
            s3g_checkpoint = "checkpoints/s3gen.pt"
            
            Path("checkpoints").mkdir(exist_ok=True)
            map_location = torch.device('cpu') if self.device in ['cpu', 'mps'] else None
            
            if Path(s3g_checkpoint).exists():
                self.s3gen_model.load_state_dict(torch.load(s3g_checkpoint, map_location=map_location))
                self.s3gen_model.to(self.device)
                self.s3gen_model.eval()
                logger.info("S3Gen model loaded successfully")
            else:
                logger.warning("S3Gen model not available")
                self.s3gen_model = None
            
            self.models_loaded = True
            load_time = time.time() - start_time
            logger.info(f"All models loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise e
    
    def get_speaker_embedding(self, job_input: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get speaker embedding from various sources"""
        try:
            voice_id = job_input.get('voice_id')
            voice_name = job_input.get('voice_name')
            reference_audio_b64 = job_input.get('reference_audio')
            max_reference_duration_sec = job_input.get('max_reference_duration_sec', 30)
            
            # Option 1: Use voice from optimized embeddings by ID
            if voice_id:
                if self.voice_library and self.voice_library.is_available():
                    audio_array = self.voice_library.get_voice_audio_direct(voice_id)
                    if audio_array is not None:
                        logger.info(f"Using voice embedding: {voice_id}")
                        return audio_array
                    else:
                        raise ValueError(f"Voice ID '{voice_id}' not found")
                else:
                    raise ValueError("Voice library not available")
            
            # Option 2: Use voice from library by name
            elif voice_name:
                if self.voice_library and self.voice_library.is_available():
                    audio_array = self.voice_library.get_voice_audio_by_name_direct(voice_name)
                    if audio_array is not None:
                        logger.info(f"Using voice embedding by name: {voice_name}")
                        return audio_array
                
                # Fallback to legacy library
                if self.legacy_voice_library and self.legacy_voice_library.is_available():
                    audio_array = self.legacy_voice_library.get_voice_audio_by_name(voice_name)
                    if audio_array is not None:
                        logger.info(f"Using voice from legacy library: {voice_name}")
                        return audio_array
                
                raise ValueError(f"Voice '{voice_name}' not found")
            
            # Option 3: Use provided reference audio
            elif reference_audio_b64:
                logger.info("Using provided reference audio")
                audio_data = base64.b64decode(reference_audio_b64)
                audio_buffer = io.BytesIO(audio_data)
                audio_array, sample_rate = sf.read(audio_buffer)
                
                # Trim to max duration
                max_samples = int(max_reference_duration_sec * sample_rate)
                if len(audio_array) > max_samples:
                    audio_array = audio_array[:max_samples]
                    logger.info(f"Trimmed reference audio to {max_reference_duration_sec} seconds")
                
                return audio_array
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting speaker embedding: {e}")
            raise e
    
    def process_segment_chatterbox(
        self, 
        segment: TextSegment, 
        speaker_embedding: Optional[np.ndarray],
        generation_params: Dict[str, Any]
    ) -> ProcessingResult:
        """Process a text segment using Chatterbox"""
        start_time = time.time()
        
        try:
            logger.debug(f"Processing Chatterbox segment {segment.index}: '{segment.text[:50]}...'")
            
            # Generate audio using Chatterbox
            audio = self.chatterbox_model.generate(
                text=segment.text,
                audio_prompt_array=speaker_embedding,
                **generation_params
            )
            
            # Convert to numpy if needed
            if isinstance(audio, torch.Tensor):
                audio = audio.squeeze().cpu().numpy()
            
            processing_time = time.time() - start_time
            sample_rate = getattr(self.chatterbox_model, 'sr', 24000)
            
            return ProcessingResult(
                index=segment.index,
                audio=audio,
                sample_rate=sample_rate,
                engine='chatterbox',
                tag_type=segment.tag_type,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Chatterbox processing failed for segment {segment.index}: {e}")
            
            return ProcessingResult(
                index=segment.index,
                audio=np.array([]),
                sample_rate=24000,
                engine='chatterbox',
                tag_type=segment.tag_type,
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def process_segment_f5(
        self, 
        segment: TextSegment, 
        speaker_embedding: Optional[np.ndarray],
        generation_params: Dict[str, Any],
        custom_tag_audio: Optional[Dict[str, np.ndarray]] = None
    ) -> ProcessingResult:
        """Process a text segment using F5-TTS"""
        start_time = time.time()
        
        try:
            logger.debug(f"Processing F5-TTS segment {segment.index} ({segment.tag_type}): '{segment.text[:50]}...'")
            
            if not self.f5_model.is_available():
                # Fallback to Chatterbox if F5-TTS not available
                logger.warning(f"F5-TTS not available for segment {segment.index}, using Chatterbox")
                return self.process_segment_chatterbox(segment, speaker_embedding, generation_params)
            
            # Generate expressive audio using F5-TTS
            audio = self.f5_model.generate_expressive(
                text=segment.text,
                tag_type=segment.tag_type,
                speaker_embedding=speaker_embedding,
                custom_tag_audio=custom_tag_audio,
                **generation_params
            )
            
            if audio is None:
                # Fallback to Chatterbox if F5-TTS generation failed
                logger.warning(f"F5-TTS generation failed for segment {segment.index}, using Chatterbox")
                return self.process_segment_chatterbox(segment, speaker_embedding, generation_params)
            
            processing_time = time.time() - start_time
            sample_rate = 24000  # F5-TTS default sample rate
            
            return ProcessingResult(
                index=segment.index,
                audio=audio,
                sample_rate=sample_rate,
                engine='f5',
                tag_type=segment.tag_type,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"F5-TTS processing failed for segment {segment.index}: {e}")
            
            # Fallback to Chatterbox
            logger.warning(f"Falling back to Chatterbox for segment {segment.index}")
            return self.process_segment_chatterbox(segment, speaker_embedding, generation_params)
    
    def _process_custom_tag_audio(self, job_input: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """Process custom reference audio for extended expressive tags"""
        try:
            custom_tags = job_input.get('custom_tag_audio', {})
            if not custom_tags:
                return None
            
            processed_audio = {}
            
            for tag_name, audio_b64 in custom_tags.items():
                try:
                    # Decode custom reference audio
                    audio_data = base64.b64decode(audio_b64)
                    audio_buffer = io.BytesIO(audio_data)
                    audio_array, sample_rate = sf.read(audio_buffer)
                    
                    # Limit duration (10 seconds max for performance)
                    max_samples = int(10 * sample_rate)
                    if len(audio_array) > max_samples:
                        audio_array = audio_array[:max_samples]
                    
                    processed_audio[tag_name] = audio_array
                    logger.info(f"Loaded custom audio for tag '{tag_name}': {len(audio_array)} samples")
                    
                except Exception as e:
                    logger.warning(f"Failed to process custom audio for tag '{tag_name}': {e}")
            
            return processed_audio if processed_audio else None
            
        except Exception as e:
            logger.error(f"Error processing custom tag audio: {e}")
            return None
    
    def process_segments_parallel(
        self, 
        segments: List[TextSegment], 
        speaker_embedding: Optional[np.ndarray],
        generation_params: Dict[str, Any],
        custom_tag_audio: Optional[Dict[str, np.ndarray]] = None
    ) -> List[ProcessingResult]:
        """Process segments in parallel for efficiency"""
        logger.info(f"Processing {len(segments)} segments in parallel")
        
        results = []
        
        # Group segments by engine for batch processing
        chatterbox_segments = [s for s in segments if s.engine == 'chatterbox']
        f5_segments = [s for s in segments if s.engine == 'f5']
        
        logger.info(f"Chatterbox segments: {len(chatterbox_segments)}, F5-TTS segments: {len(f5_segments)}")
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_segments) as executor:
            # Submit all tasks
            future_to_segment = {}
            
            # Submit Chatterbox tasks
            for segment in chatterbox_segments:
                future = executor.submit(
                    self.process_segment_chatterbox, 
                    segment, 
                    speaker_embedding, 
                    generation_params
                )
                future_to_segment[future] = segment
            
            # Submit F5-TTS tasks
            for segment in f5_segments:
                future = executor.submit(
                    self.process_segment_f5, 
                    segment, 
                    speaker_embedding, 
                    generation_params,
                    custom_tag_audio
                )
                future_to_segment[future] = segment
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_segment):
                result = future.result()
                results.append(result)
                
                if result.success:
                    duration = len(result.audio) / result.sample_rate
                    logger.debug(f"Segment {result.index} completed: {duration:.2f}s audio in {result.processing_time:.2f}s")
                else:
                    logger.error(f"Segment {result.index} failed: {result.error}")
        
        # Sort results by index to maintain order
        results.sort(key=lambda x: x.index)
        
        return results
    
    def generate_expressive_tts(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for expressive TTS generation"""
        try:
            # Load models if not already loaded
            if not self.models_loaded:
                self.load_models()
            
            # Extract and validate text
            text = job_input.get('text', '')
            if not text or len(text.strip()) == 0:
                raise ValueError("Text cannot be empty")
            
            # Validate text and parse segments
            is_valid, error_msg = self.text_parser.validate_text(text)
            if not is_valid:
                raise ValueError(f"Text validation failed: {error_msg}")
            
            logger.info(f"Processing expressive TTS request - Text: {len(text)} chars")
            
            # Parse text into segments
            segments = self.text_parser.parse_text(text)
            stats = self.text_parser.get_segment_stats(segments)
            
            logger.info(f"Parsed into {stats['total_segments']} segments: "
                       f"{stats['chatterbox_segments']} Chatterbox, {stats['f5_segments']} F5-TTS")
            
            # Get speaker embedding once for all segments
            speaker_embedding = self.get_speaker_embedding(job_input)
            voice_cloning_enabled = speaker_embedding is not None
            
            # Handle custom tag audio for extended expressions
            custom_tag_audio = self._process_custom_tag_audio(job_input)
            
            # Prepare generation parameters
            generation_params = {
                'sample_rate': job_input.get('sample_rate', None),
                'audio_normalization': job_input.get('audio_normalization', None),
                'chunk_size': job_input.get('chunk_size', 50),
                'exaggeration': job_input.get('exaggeration', 0.5),
                'cfg_weight': job_input.get('cfg_weight', 0.5),
                'temperature': job_input.get('temperature', 0.8)
            }
            
            # Process all segments in parallel
            start_time = time.time()
            processing_results = self.process_segments_parallel(
                segments, 
                speaker_embedding, 
                generation_params,
                custom_tag_audio
            )
            
            # Check for any failures
            failed_segments = [r for r in processing_results if not r.success]
            if failed_segments:
                logger.warning(f"{len(failed_segments)} segments failed processing")
            
            # Convert processing results to audio segments
            audio_segments = []
            for result in processing_results:
                if result.success and len(result.audio) > 0:
                    audio_segment = AudioSegmentData(
                        index=result.index,
                        audio=result.audio,
                        sample_rate=result.sample_rate,
                        engine=result.engine,
                        tag_type=result.tag_type,
                        duration=len(result.audio) / result.sample_rate
                    )
                    audio_segments.append(audio_segment)
            
            if not audio_segments:
                raise RuntimeError("No audio segments were generated successfully")
            
            # Stitch segments together
            logger.info("Stitching audio segments...")
            output_sr = generation_params['sample_rate'] or 24000
            final_audio, final_sr = self.audio_stitcher.stitch_segments(audio_segments, output_sr)
            
            total_time = time.time() - start_time
            duration = len(final_audio) / final_sr
            
            # Convert to base64
            buffer = io.BytesIO()
            sf.write(buffer, final_audio, final_sr, format='WAV')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Prepare response
            response = {
                "audio": audio_base64,
                "sample_rate": final_sr,
                "text": text,
                "mode": "expressive_enhanced",
                "parameters": {
                    "voice_cloning": voice_cloning_enabled,
                    "voice_id": job_input.get('voice_id', 'none'),
                    "voice_name": job_input.get('voice_name', 'none'),
                    "crossfade_ms": self.crossfade_ms
                },
                "processing_stats": {
                    "total_segments": len(segments),
                    "successful_segments": len(audio_segments),
                    "failed_segments": len(failed_segments),
                    "segment_breakdown": stats,
                    "total_processing_time": total_time,
                    "audio_duration": duration,
                    "rtf": total_time / duration if duration > 0 else None,
                    "engines_used": list(set(r.engine for r in processing_results if r.success))
                }
            }
            
            logger.info(f"Expressive TTS completed: {duration:.2f}s audio in {total_time:.2f}s "
                       f"(RTF: {total_time/duration:.3f})")
            
            return response
            
        except Exception as e:
            logger.error(f"Expressive TTS generation failed: {e}")
            raise RuntimeError(f"Expressive TTS generation failed: {str(e)}")

# Global handler instance
enhanced_handler: Optional[EnhancedTTSHandler] = None

def handler_enhanced(job):
    """Enhanced handler function for RunPod"""
    global enhanced_handler
    
    try:
        # Initialize handler if not already done
        if enhanced_handler is None:
            enhanced_handler = EnhancedTTSHandler()
        
        # Get job input
        job_input = job.get('input', {})
        
        # Determine operation type
        operation = job_input.get('operation', 'tts')
        
        if operation == 'expressive_tts' or operation == 'tts':
            # Check if expressive tags are present
            text = job_input.get('text', '')
            if '{' in text and '}' in text:
                # Use enhanced expressive TTS
                return enhanced_handler.generate_expressive_tts(job_input)
            else:
                # Fall back to original optimized handler for simple TTS
                from handler import handler_optimized
                return handler_optimized(job)
        
        else:
            # Fall back to original handler for other operations
            from handler import handler_optimized
            return handler_optimized(job)
            
    except Exception as e:
        logger.error(f"Enhanced handler error: {e}")
        return {"error": str(e)}

# Load models on startup
if __name__ == "__main__":
    logger.info("Starting Enhanced Expressive TTS handler...")
    
    # Initialize handler
    enhanced_handler = EnhancedTTSHandler()
    enhanced_handler.load_models()
    
    # Start the enhanced handler
    runpod.serverless.start({"handler": handler_enhanced}) 