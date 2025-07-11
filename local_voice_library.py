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
            voices.append({
                'voice_id': voice_id,
                'name': metadata['name'],
                'description': metadata['description'],
                'category': metadata['category'],
                'speaker_name': metadata['speaker_name'],
                'duration': metadata['duration'],
                'file_size': metadata['file_size'],
                'compressed_size': metadata['compressed_size'],
                'compression_ratio': metadata['compression_ratio']
            })
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
            embedding_file = self.embeddings_dir / voice_info['embedding_file']
            
            # Load compressed audio
            if not embedding_file.exists():
                logger.error(f"Embedding file not found: {embedding_file}")
                return None
            
            with open(embedding_file, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress and unpickle
            decompressed_data = gzip.decompress(compressed_data)
            audio_array = pickle.loads(decompressed_data)
            
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
        """Find voice ID by name (prioritizes speaker name matching)"""
        try:
            # Normalize the search name
            search_name = voice_name.lower().strip()
            
            # First priority: exact speaker name match (e.g., "Amy" matches "Amy")
            for voice_id, metadata in self.voice_catalog.items():
                if metadata['speaker_name'].lower() == search_name:
                    return voice_id
            
            # Second priority: exact full name match (e.g., "female/Amy")
            for voice_id, metadata in self.voice_catalog.items():
                if metadata['name'].lower() == search_name:
                    return voice_id
            
            # Third priority: partial matches
            for voice_id, metadata in self.voice_catalog.items():
                voice_full_name = metadata['name'].lower()
                speaker_name = metadata['speaker_name'].lower()
                
                # Check if search matches any part of the names
                if (search_name in voice_full_name or 
                    search_name in speaker_name or
                    voice_full_name.endswith(search_name)):
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
    
    def search_voices(self, query: str = "", category: str = "", speaker: str = "") -> List[Dict[str, Any]]:
        """Search voices by query, category, or speaker"""
        results = []
        
        for voice_id, metadata in self.voice_catalog.items():
            # Apply filters
            if category and metadata['category'].lower() != category.lower():
                continue
            
            if speaker and speaker.lower() not in metadata['speaker_name'].lower():
                continue
            
            if query:
                # Search in name, description, speaker name
                searchable_text = f"{metadata['name']} {metadata['description']} {metadata['speaker_name']}".lower()
                if query.lower() not in searchable_text:
                    continue
            
            results.append({
                'voice_id': voice_id,
                'name': metadata['name'],
                'description': metadata['description'],
                'category': metadata['category'],
                'speaker_name': metadata['speaker_name'],
                'duration': metadata['duration']
            })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get voice library statistics"""
        return {
            'total_voices': len(self.voice_catalog),
            'categories': self.stats.get('categories', {}),
            'storage': self.stats.get('storage', {}),
            'version': self.stats.get('version', 'unknown')
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
        print(f"📂 Categories: {list(stats['categories'].keys())}")
        
        # List first few voices
        voices = library.list_voices()[:5]
        print(f"\n🎤 Sample voices:")
        for voice in voices:
            print(f"  • {voice['voice_id']}: {voice['name']} ({voice['category']})")
        
        # Test loading a voice
        if voices:
            test_voice_id = voices[0]['voice_id']
            print(f"\n🧪 Testing voice load: {test_voice_id}")
            
            audio = library.get_voice_audio(test_voice_id)
            if audio is not None:
                print(f"✅ Successfully loaded: {audio.shape} samples")
            else:
                print("❌ Failed to load voice")
    else:
        print("❌ No local voice library found")
        print("Run generate_local_embeddings.py first to create embeddings") 