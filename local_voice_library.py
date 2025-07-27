#!/usr/bin/env python3
"""
Local Voice Library Loader
Loads voice embeddings from local voice_embeddings/ folder for TTS API usage
"""

import os
import json
import pickle
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import base64
import io

logger = logging.getLogger(__name__)

class LocalVoiceLibrary:
    """Load and manage local voice embeddings for TTS"""
    
    def __init__(self, embeddings_dir: str = "voice_embeddings"):
        """Initialize local voice library"""
        self.embeddings_dir = Path(embeddings_dir)
        self.voice_catalog = {}
        self.quick_index = {}
        self.name_mapping = {}
        self.stats = {}
        
        # Load voice library
        self._load_voice_library()
    
    def _load_voice_library(self):
        """Load voice catalog and indexes"""
        try:
            # Check if embeddings directory exists
            if not self.embeddings_dir.exists():
                logger.warning(f"Voice embeddings directory not found: {self.embeddings_dir}")
                return
            
            # Load voice catalog
            catalog_file = self.embeddings_dir / "voice_catalog.json"
            if catalog_file.exists():
                with open(catalog_file, 'r') as f:
                    self.voice_catalog = json.load(f)
                logger.info(f"Loaded {len(self.voice_catalog)} voices from catalog")
            
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
            
            logger.info(f"Local voice library initialized with {len(self.voice_catalog)} voices")
            
        except Exception as e:
            logger.error(f"Failed to load voice library: {e}")
    
    def list_voices(self) -> List[Dict[str, Any]]:
        """List all available voices"""
        voices = []
        for voice_id, metadata in self.voice_catalog.items():
            # Handle both old and new metadata structures
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
    
    def get_voice_info(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Get voice metadata by ID"""
        return self.voice_catalog.get(voice_id)
    
    def get_voice_info_by_name(self, voice_name: str) -> Optional[Dict[str, Any]]:
        """Get voice metadata by name"""
        voice_id = self.find_voice_id_by_name(voice_name)
        if voice_id:
            return self.voice_catalog.get(voice_id)
        return None
    
    def get_voice_audio(self, voice_id: str) -> Optional[np.ndarray]:
        """Load voice audio data by ID"""
        try:
            # Check if voice exists
            if voice_id not in self.voice_catalog:
                logger.error(f"Voice not found: {voice_id}")
                return None
            
            voice_info = self.voice_catalog[voice_id]
            
            # Determine embedding file path
            if 'embedding_file' in voice_info:
                # New ChatTTS structure
                embedding_path = voice_info['embedding_file']
            else:
                # Fallback: construct path from voice_id
                embedding_path = f"embeddings/{voice_id}.pkl.gz"
            
            # Normalize path separators for cross-platform compatibility
            embedding_path = embedding_path.replace('\\', '/')
            embedding_file = self.embeddings_dir / embedding_path
            
            # Load compressed audio
            if not embedding_file.exists():
                logger.error(f"Embedding file not found: {embedding_file}")
                return None
            
            with gzip.open(embedding_file, 'rb') as f:
                embedding_data = pickle.load(f)
            
            # Handle different embedding formats
            if isinstance(embedding_data, dict) and 'audio' in embedding_data:
                # New ChatTTS format: extract audio array from dict
                audio_array = embedding_data['audio']
            elif isinstance(embedding_data, np.ndarray):
                # Legacy format: direct audio array
                audio_array = embedding_data
            else:
                logger.error(f"Unknown embedding format for voice {voice_id}")
                return None
            
            logger.info(f"Loaded voice {voice_id}: {audio_array.shape} samples")
            return audio_array
            
        except Exception as e:
            logger.error(f"Failed to load voice {voice_id}: {e}")
            return None
    
    def get_voice_audio_by_name(self, voice_name: str) -> Optional[np.ndarray]:
        """Load voice audio data by name"""
        try:
            # Find voice by name
            voice_id = self.find_voice_id_by_name(voice_name)
            if voice_id is None:
                logger.error(f"Voice name not found: {voice_name}")
                return None
            
            return self.get_voice_audio(voice_id)
            
        except Exception as e:
            logger.error(f"Failed to load voice by name {voice_name}: {e}")
            return None
    
    def find_voice_id_by_name(self, voice_name: str) -> Optional[str]:
        """Find voice ID by name (prioritizes optimized name mapping then speaker name matching)"""
        try:
            # Normalize the search name
            search_name = voice_name.strip()
            search_name_lower = search_name.lower()
            
            # First priority: direct name mapping lookup (fastest)
            if search_name in self.name_mapping:
                return self.name_mapping[search_name]
            
            # Second priority: case-insensitive name mapping lookup
            for name, voice_id in self.name_mapping.items():
                if name.lower() == search_name_lower:
                    return voice_id
            
            # Third priority: exact speaker name match (e.g., "Amy" matches "Amy")
            for voice_id, metadata in self.voice_catalog.items():
                speaker_name = metadata.get('speaker_name', '')
                if speaker_name.lower() == search_name_lower:
                    return voice_id
            
            # Fourth priority: legacy full name match (for backwards compatibility)
            for voice_id, metadata in self.voice_catalog.items():
                name = metadata.get('name', '')
                if name.lower() == search_name_lower:
                    return voice_id
            
            # Fifth priority: partial matches
            for voice_id, metadata in self.voice_catalog.items():
                speaker_name = metadata.get('speaker_name', '').lower()
                name = metadata.get('name', '').lower()
                
                # Check if search matches any part of the names
                if (search_name_lower in speaker_name or 
                    search_name_lower in name or
                    speaker_name.startswith(search_name_lower)):
                    return voice_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding voice by name {voice_name}: {e}")
            return None
    
    def get_voice_audio_base64(self, voice_id: str) -> Optional[str]:
        """Get voice audio as base64 encoded WAV"""
        try:
            audio_array = self.get_voice_audio(voice_id)
            if audio_array is None:
                return None
            
            # Convert to base64 WAV
            import soundfile as sf
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, 24000, format='WAV')
            buffer.seek(0)
            
            audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            return audio_b64
            
        except Exception as e:
            logger.error(f"Failed to convert voice {voice_id} to base64: {e}")
            return None
    
    def search_voices(self, query: str = "", category: str = "", speaker: str = "", gender: str = "") -> List[Dict[str, Any]]:
        """Search voices by query, category, speaker, or gender"""
        results = []
        
        for voice_id, metadata in self.voice_catalog.items():
            # Apply filters
            if category and metadata.get('category', '').lower() != category.lower():
                continue
            
            if gender and metadata.get('gender', '').lower() != gender.lower():
                continue
            
            if speaker and speaker.lower() not in metadata.get('speaker_name', '').lower():
                continue
            
            if query:
                # Search in speaker name, description, original file
                searchable_text = f"{metadata.get('speaker_name', '')} {metadata.get('description', '')} {metadata.get('original_file', '')}".lower()
                if query.lower() not in searchable_text:
                    continue
            
            result = {
                'voice_id': voice_id,
                'speaker_name': metadata.get('speaker_name', 'Unknown'),
                'gender': metadata.get('gender', 'unknown'),
                'duration': metadata.get('duration', 0),
                'original_file': metadata.get('original_file', ''),
                'type': metadata.get('type', 'embedding')
            }
            
            # Legacy compatibility
            if 'name' in metadata:
                result['name'] = metadata['name']
            if 'description' in metadata:
                result['description'] = metadata['description']
            if 'category' in metadata:
                result['category'] = metadata['category']
            
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get voice library statistics"""
        # Calculate gender distribution
        gender_counts = {}
        for metadata in self.voice_catalog.values():
            gender = metadata.get('gender', 'unknown')
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        
        return {
            'total_voices': len(self.voice_catalog),
            'gender_distribution': gender_counts,
            'generation_time': self.stats.get('generation_time'),
            'average_duration': self.stats.get('average_duration'),
            'type': 'chattts_embeddings'
        }
    
    def is_available(self) -> bool:
        """Check if local voice library is available"""
        return len(self.voice_catalog) > 0

def initialize_local_voice_library(embeddings_dir: str = "voice_embeddings") -> LocalVoiceLibrary:
    """Initialize local voice library"""
    return LocalVoiceLibrary(embeddings_dir)

# Example usage
if __name__ == "__main__":
    # Test the library
    library = initialize_local_voice_library()
    
    if library.is_available():
        print("🎭 Local Voice Library Test")
        print("=" * 40)
        
        # Show stats
        stats = library.get_stats()
        print(f"📊 Total voices: {stats['total_voices']}")
        print(f"👥 Gender distribution: {stats['gender_distribution']}")
        
        # List first few voices
        voices = library.list_voices()[:5]
        print(f"\n🎤 Sample voices:")
        for voice in voices:
            print(f"  • {voice['voice_id']}: {voice['speaker_name']} ({voice['gender']})")
        
        # Test loading a voice by name
        if voices:
            test_name = voices[0]['speaker_name']
            print(f"\n🧪 Testing voice load by name: {test_name}")
            
            audio = library.get_voice_audio_by_name(test_name)
            if audio is not None:
                print(f"✅ Successfully loaded: {audio.shape} samples")
            else:
                print("❌ Failed to load voice")
    else:
        print("❌ No local voice library found")
        print("Run generate_chattts_embeddings.py first to create embeddings") 