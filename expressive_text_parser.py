#!/usr/bin/env python3
"""
Expressive Text Parser for F5-TTS Integration
Handles parsing text with expressive tags like {whisper}, {yell}, etc.
"""

import re
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TextSegment:
    """Represents a segment of text with processing information"""
    index: int
    text: str
    engine: str  # 'chatterbox' or 'f5'
    tag_type: str = None  # whisper, yell, etc. (only for f5 segments)
    original_position: int = 0  # Position in original text for ordering

class ExpressiveTextParser:
    """Parser for handling expressive tags in text"""
    
    # Define supported expressive tags and their F5-TTS mappings
    # Native F5-TTS tags (built-in support)
    NATIVE_F5_TAGS = {
        'whisper': 'whisper',
        'shout': 'shout',
        'happy': 'happy',
        'sad': 'sad',
        'angry': 'angry'
    }
    
    # Additional tags that can be supported via custom reference audio
    CUSTOM_TAGS = {
        'yell': 'shout',  # Map to native shout
        'excited': 'happy',  # Map to native happy
        'calm': 'whisper',  # Map to native whisper
        'nervous': 'sad',  # Map to native sad
        'confident': 'happy'  # Map to native happy
    }
    
    # Combined supported tags
    SUPPORTED_TAGS = {**NATIVE_F5_TAGS, **CUSTOM_TAGS}
    
    def __init__(self):
        # Create regex pattern to match expressive tags
        # Matches {tag}text{/tag} or {tag:text} patterns
        tag_names = '|'.join(self.SUPPORTED_TAGS.keys())
        self.tag_pattern = re.compile(
            rf'\{{({tag_names})\}}(.*?)\{{/\1\}}|\{{({tag_names}):([^}}]+)\}}',
            re.IGNORECASE | re.DOTALL
        )
    
    def parse_text(self, text: str) -> List[TextSegment]:
        """
        Parse text and return ordered segments for processing
        
        Args:
            text: Input text with expressive tags
            
        Returns:
            List of TextSegment objects in original order
        """
        segments = []
        current_pos = 0
        segment_index = 0
        
        logger.info(f"Parsing text with {len(text)} characters")
        
        # Find all tag matches
        for match in self.tag_pattern.finditer(text):
            start, end = match.span()
            
            # Add any text before this tag as a Chatterbox segment
            if start > current_pos:
                before_text = text[current_pos:start].strip()
                if before_text:
                    segments.append(TextSegment(
                        index=segment_index,
                        text=before_text,
                        engine='chatterbox',
                        original_position=current_pos
                    ))
                    segment_index += 1
            
            # Extract tag information
            if match.group(1):  # {tag}text{/tag} format
                tag_type = match.group(1).lower()
                tagged_text = match.group(2).strip()
            else:  # {tag:text} format
                tag_type = match.group(3).lower()
                tagged_text = match.group(4).strip()
            
            # Add the tagged segment for F5-TTS
            if tagged_text and tag_type in self.SUPPORTED_TAGS:
                segments.append(TextSegment(
                    index=segment_index,
                    text=tagged_text,
                    engine='f5',
                    tag_type=self.SUPPORTED_TAGS[tag_type],
                    original_position=start
                ))
                segment_index += 1
                logger.debug(f"Found expressive segment: {tag_type} -> '{tagged_text[:50]}...'")
            
            current_pos = end
        
        # Add any remaining text after the last tag
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                segments.append(TextSegment(
                    index=segment_index,
                    text=remaining_text,
                    engine='chatterbox',
                    original_position=current_pos
                ))
        
        # If no tags found, return entire text as Chatterbox segment
        if not segments:
            segments.append(TextSegment(
                index=0,
                text=text.strip(),
                engine='chatterbox',
                original_position=0
            ))
        
        logger.info(f"Parsed into {len(segments)} segments: "
                   f"{sum(1 for s in segments if s.engine == 'chatterbox')} Chatterbox, "
                   f"{sum(1 for s in segments if s.engine == 'f5')} F5-TTS")
        
        return segments
    
    def validate_text(self, text: str) -> Tuple[bool, str]:
        """
        Validate text for proper tag formatting
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text or not text.strip():
            return False, "Text cannot be empty"
        
        if len(text) > 5000:
            return False, "Text too long (max 5000 characters)"
        
        # Check for unclosed tags
        open_tags = re.findall(r'\{([^}]+)\}', text)
        close_tags = re.findall(r'\{/([^}]+)\}', text)
        
        # Find tags that are opened but not closed
        for tag in open_tags:
            if tag.startswith('/'):
                continue
            if ':' in tag:  # {tag:text} format is self-closing
                continue
            if tag not in close_tags:
                return False, f"Unclosed tag: {{{tag}}}"
        
        # Check for unsupported tags
        tag_pattern = re.compile(r'\{([^}:]+)(?::|[^}]*)\}')
        for match in tag_pattern.finditer(text):
            tag_name = match.group(1).lower()
            if tag_name not in self.SUPPORTED_TAGS and not tag_name.startswith('/'):
                return False, f"Unsupported tag: {{{tag_name}}}. Supported tags: {list(self.SUPPORTED_TAGS.keys())}"
        
        return True, ""
    
    def get_segment_stats(self, segments: List[TextSegment]) -> Dict[str, Any]:
        """Get statistics about parsed segments"""
        chatterbox_count = sum(1 for s in segments if s.engine == 'chatterbox')
        f5_count = sum(1 for s in segments if s.engine == 'f5')
        
        tag_counts = {}
        for segment in segments:
            if segment.tag_type:
                tag_counts[segment.tag_type] = tag_counts.get(segment.tag_type, 0) + 1
        
        total_chars = sum(len(s.text) for s in segments)
        chatterbox_chars = sum(len(s.text) for s in segments if s.engine == 'chatterbox')
        f5_chars = sum(len(s.text) for s in segments if s.engine == 'f5')
        
        return {
            'total_segments': len(segments),
            'chatterbox_segments': chatterbox_count,
            'f5_segments': f5_count,
            'tag_distribution': tag_counts,
            'character_counts': {
                'total': total_chars,
                'chatterbox': chatterbox_chars,
                'f5': f5_chars
            }
        }

def create_text_parser() -> ExpressiveTextParser:
    """Factory function to create a text parser instance"""
    return ExpressiveTextParser()

# Example usage and testing
if __name__ == "__main__":
    # Test the parser
    parser = create_text_parser()
    
    test_texts = [
        "This is normal text. {whisper}This should be whispered{/whisper} and back to normal.",
        "Start normal {yell:LOUD SHOUTING} then {calm}peaceful again{/calm} ending normal.",
        "No tags in this text at all.",
        "{excited}Excited start{/excited} middle text {sad:sad ending}",
        "Mixed with {angry}ANGER{/angry} and {happy}JOY{/happy} in one sentence!"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Test {i+1} ---")
        print(f"Input: {text}")
        
        is_valid, error = parser.validate_text(text)
        if not is_valid:
            print(f"Validation Error: {error}")
            continue
            
        segments = parser.parse_text(text)
        stats = parser.get_segment_stats(segments)
        
        print(f"Stats: {stats}")
        print("Segments:")
        for segment in segments:
            engine_info = f"({segment.tag_type})" if segment.tag_type else ""
            print(f"  {segment.index}: [{segment.engine}{engine_info}] '{segment.text}'") 