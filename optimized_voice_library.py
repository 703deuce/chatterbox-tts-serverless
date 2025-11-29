#!/usr/bin/env python3
"""
Optimized Voice Library that returns audio arrays directly
instead of creating temporary files, eliminating I/O overhead.
"""

import os
import json
import pickle
import gzip
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class OptimizedVoiceLibrary:
    """
    Optimized voice library that returns audio arrays directly
    for use with OptimizedChatterboxTTS
    """
    
    def __init__(self, embeddings_dir: str = "voice_embeddings"):
        """Initialize optimized voice library"""
        self.embeddings_dir = Path(embeddings_dir)
        self.voice_catalog = {}
        self.quick_index = {}
        self.name_mapping = {}
        self.stats = {}
        
        # Cache for frequently accessed voices
        self._audio_cache = {}
        self._max_cache_size = 10  # Cache up to 10 voices in memory
        
        # Load voice library
        self._load_voice_library()
    
    def _load_voice_library(self):
        """Load voice catalog and indexes"""
        try:
            # Check if embeddings directory exists
            if not self.embeddings_dir.exists():
                logger.warning(f"Voice embeddings directory not found: {self.embeddings_dir}")
                # Create directory if it doesn't exist (for RunPod Serverless)
                self.embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            # Load voice catalog
            catalog_file = self.embeddings_dir / "voice_catalog.json"
            if catalog_file.exists():
                with open(catalog_file, 'r') as f:
                    self.voice_catalog = json.load(f)
                logger.info(f"Loaded {len(self.voice_catalog)} voices from catalog")
            else:
                logger.warning(f"Voice catalog not found: {catalog_file}")
                # Start with empty catalog - voices can be added dynamically via API
            
            # Load quick index
            index_file = self.embeddings_dir / "quick_index.json"
            if index_file.exists():
                with open(index_file, 'r') as f:
                    self.quick_index = json.load(f)
            
            # Load name mapping (for optimized lookup)
            name_mapping_file = self.embeddings_dir / "name_mapping.json"
            if name_mapping_file.exists():
                with open(name_mapping_file, 'r') as f:
                    self.name_mapping = json.load(f)
            
            # Load stats
            stats_file = self.embeddings_dir / "stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.stats = json.load(f)
            
            logger.info(f"Optimized voice library initialized with {len(self.voice_catalog)} voices")
            
        except Exception as e:
            logger.error(f"Failed to load voice library: {e}")
    
    def add_voice_to_catalog(self, voice_metadata: Dict[str, Any]):
        """
        Add a voice to the catalog dynamically (for RunPod Serverless)
        
        This allows adding voices from external apps without rebuilding the Docker image.
        The voice metadata should include firebase_storage_path and firebase_storage_bucket.
        
        Args:
            voice_metadata: Voice metadata dictionary with voice_id, speaker_name, etc.
        """
        try:
            voice_id = voice_metadata.get('voice_id')
            if not voice_id:
                logger.error("voice_id is required in voice_metadata")
                return False
            
            # Add to catalog
            self.voice_catalog[voice_id] = voice_metadata
            
            # Update name mapping if speaker_name is provided
            speaker_name = voice_metadata.get('speaker_name')
            if speaker_name:
                self.name_mapping[speaker_name] = voice_id
            
            logger.info(f"Added voice {voice_id} ({speaker_name}) to catalog")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add voice to catalog: {e}")
            return False
    
    def get_voice_audio_direct(self, voice_id: str) -> Optional[np.ndarray]:
        """
        OPTIMIZED: Get voice audio array with Firebase Storage fallback for RunPod Serverless
        
        Supports ephemeral filesystem by downloading from Firebase Storage when local file
        doesn't exist. Perfect for RunPod Serverless workers.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Audio array ready for direct use with OptimizedChatterboxTTS
        """
        try:
            # Check cache first (in-memory, persists during worker lifetime)
            if voice_id in self._audio_cache:
                logger.debug(f"Cache hit for voice {voice_id}")
                return self._audio_cache[voice_id]
            
            # Check if voice exists in catalog
            if voice_id not in self.voice_catalog:
                logger.error(f"Voice not found in catalog: {voice_id}")
                return None
            
            voice_info = self.voice_catalog[voice_id]
            
            # Determine embedding file path
            if 'embedding_file' in voice_info:
                embedding_path = voice_info['embedding_file']
            else:
                embedding_path = f"embeddings/{voice_id}.pkl.gz"
            
            # Normalize path separators
            embedding_path = embedding_path.replace('\\', '/')
            embedding_file = self.embeddings_dir / embedding_path
            
            # Try to load from local filesystem first (for built-in voices in Docker image)
            if embedding_file.exists():
                logger.info(f"Loading embedding from local filesystem: {voice_id}")
                with gzip.open(embedding_file, 'rb') as f:
                    embedding_data = pickle.load(f)
            else:
                # Local file doesn't exist - download from Firebase Storage
                logger.info(f"Local file not found, downloading from Firebase Storage: {voice_id}")
                embedding_data = self._download_embedding_from_firebase(voice_info, voice_id)
                if embedding_data is None:
                    logger.error(f"Failed to download embedding from Firebase: {voice_id}")
                    return None
            
            # Extract audio array
            if isinstance(embedding_data, dict) and 'audio' in embedding_data:
                audio_array = embedding_data['audio']
            elif isinstance(embedding_data, np.ndarray):
                audio_array = embedding_data
            else:
                logger.error(f"Unknown embedding format for voice {voice_id}")
                return None
            
            # Cache in memory (not filesystem - works with ephemeral storage)
            self._manage_cache(voice_id, audio_array)
            
            logger.info(f"Loaded voice {voice_id}: {audio_array.shape} samples")
            return audio_array
            
        except Exception as e:
            logger.error(f"Failed to load voice {voice_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _download_embedding_from_firebase(self, voice_info: Dict[str, Any], voice_id: str) -> Optional[Any]:
        """
        Download embedding file from Firebase Storage for RunPod Serverless
        
        Uses temp file that's cleaned up after loading. Works with ephemeral filesystem.
        
        Args:
            voice_info: Voice metadata dictionary
            voice_id: Voice identifier
            
        Returns:
            Embedding data (dict or np.ndarray) or None if download fails
        """
        try:
            import tempfile
            import os
            import urllib.request
            import urllib.parse
            
            # Get Firebase Storage path from metadata
            storage_bucket = voice_info.get('firebase_storage_bucket', 'aitts-d4c6d.firebasestorage.app')
            storage_path = voice_info.get('firebase_storage_path')
            
            if not storage_path:
                # Construct default path if not specified
                storage_path = f"voices/embeddings/{voice_id}.pkl.gz"
                logger.info(f"No firebase_storage_path in metadata, using default: {storage_path}")
            
            # Construct Firebase download URL (reuse existing function if available)
            try:
                # Try to import from handler if available
                from handler import construct_firebase_url
                firebase_url = construct_firebase_url(storage_bucket, storage_path)
            except ImportError:
                # Fallback: construct URL manually
                encoded_path = urllib.parse.quote(storage_path, safe='/')
                firebase_url = f"https://firebasestorage.googleapis.com/v0/b/{storage_bucket}/o/{encoded_path}?alt=media"
            
            logger.info(f"Downloading embedding from Firebase: {firebase_url}")
            
            # Download to temporary file (ephemeral - cleaned up after use)
            temp_file = tempfile.mktemp(suffix='.pkl.gz')
            
            try:
                # Download the file
                urllib.request.urlretrieve(firebase_url, temp_file)
                
                # Load embedding data from temp file
                with gzip.open(temp_file, 'rb') as f:
                    embedding_data = pickle.load(f)
                
                logger.info(f"Successfully downloaded and loaded embedding from Firebase: {voice_id}")
                return embedding_data
                
            finally:
                # Always clean up temp file (important for RunPod Serverless)
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
        
        except Exception as e:
            logger.error(f"Failed to download embedding from Firebase: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_voice_audio_by_name_direct(self, voice_name: str) -> Optional[np.ndarray]:
        """
        OPTIMIZED: Get voice audio array by name directly
        
        Args:
            voice_name: Voice name (e.g., "Amy", "Benjamin")
            
        Returns:
            Audio array ready for direct use with OptimizedChatterboxTTS
        """
        try:
            # Find voice by name
            voice_id = self.find_voice_id_by_name(voice_name)
            if voice_id is None:
                logger.error(f"Voice name not found: {voice_name}")
                return None
            
            return self.get_voice_audio_direct(voice_id)
            
        except Exception as e:
            logger.error(f"Failed to load voice by name {voice_name}: {e}")
            return None
    
    def _manage_cache(self, voice_id: str, audio_array: np.ndarray):
        """Manage in-memory cache for frequently accessed voices"""
        if len(self._audio_cache) >= self._max_cache_size:
            # Remove oldest cached voice (simple FIFO)
            oldest_voice = next(iter(self._audio_cache))
            del self._audio_cache[oldest_voice]
            logger.debug(f"Removed {oldest_voice} from cache")
        
        self._audio_cache[voice_id] = audio_array.copy()
        logger.debug(f"Cached voice {voice_id}")
    
    def find_voice_id_by_name(self, voice_name: str) -> Optional[str]:
        """Find voice ID by name (same logic as original)"""
        try:
            search_name = voice_name.strip()
            search_name_lower = search_name.lower()
            
            # Direct name mapping lookup (fastest)
            if search_name in self.name_mapping:
                return self.name_mapping[search_name]
            
            # Case-insensitive name mapping lookup
            for name, voice_id in self.name_mapping.items():
                if name.lower() == search_name_lower:
                    return voice_id
            
            # Exact speaker name match
            for voice_id, metadata in self.voice_catalog.items():
                speaker_name = metadata.get('speaker_name', '')
                if speaker_name.lower() == search_name_lower:
                    return voice_id
            
            # Legacy full name match
            for voice_id, metadata in self.voice_catalog.items():
                name = metadata.get('name', '')
                if name.lower() == search_name_lower:
                    return voice_id
            
            # Partial matches
            for voice_id, metadata in self.voice_catalog.items():
                speaker_name = metadata.get('speaker_name', '').lower()
                name = metadata.get('name', '').lower()
                
                if (search_name_lower in speaker_name or 
                    search_name_lower in name or
                    speaker_name.startswith(search_name_lower)):
                    return voice_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding voice by name {voice_name}: {e}")
            return None
    
    def list_voices(self) -> List[Dict[str, Any]]:
        """List all available voices (same as original)"""
        voices = []
        for voice_id, metadata in self.voice_catalog.items():
            voice_data = {
                'voice_id': voice_id,
                'speaker_name': metadata.get('speaker_name', 'Unknown'),
                'gender': metadata.get('gender', 'unknown'),
                'duration': metadata.get('duration', 0),
                'original_file': metadata.get('original_file', ''),
                'type': metadata.get('type', 'embedding')
            }
            
            # Legacy compatibility
            if 'name' in metadata:
                voice_data['name'] = metadata['name']
            if 'description' in metadata:
                voice_data['description'] = metadata['description']
            if 'category' in metadata:
                voice_data['category'] = metadata['category']
            
            voices.append(voice_data)
        return voices
    
    def get_stats(self) -> Dict[str, Any]:
        """Get voice library statistics"""
        gender_counts = {}
        for metadata in self.voice_catalog.values():
            gender = metadata.get('gender', 'unknown')
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        
        return {
            'total_voices': len(self.voice_catalog),
            'gender_distribution': gender_counts,
            'generation_time': self.stats.get('generation_time'),
            'average_duration': self.stats.get('average_duration'),
            'type': 'optimized_chattts_embeddings',
            'cache_size': len(self._audio_cache),
            'max_cache_size': self._max_cache_size
        }
    
    def is_available(self) -> bool:
        """Check if voice library is available"""
        return len(self.voice_catalog) > 0
    
    def clear_cache(self):
        """Clear the audio cache to free memory"""
        self._audio_cache.clear()
        logger.info("Voice audio cache cleared")

def initialize_optimized_voice_library(embeddings_dir: str = "voice_embeddings") -> OptimizedVoiceLibrary:
    """Initialize optimized voice library"""
    return OptimizedVoiceLibrary(embeddings_dir) 