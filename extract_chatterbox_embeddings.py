#!/usr/bin/env python3
"""
Extract Speaker Embeddings from Audio using ChatterboxTTS VoiceEncoder

This script extracts speaker embeddings the EXACT same way ChatterboxTTS does internally,
so you can generate embeddings separately and use them directly with ChatterboxTTS.

Usage:
    from extract_chatterbox_embeddings import extract_speaker_embedding
    
    # Extract embedding from audio file
    embedding = extract_speaker_embedding("my_voice.wav")
    
    # Save for later use
    np.save("my_voice_embedding.npy", embedding)
"""

import numpy as np
import librosa
import torch
from pathlib import Path
from typing import Optional, Union, Tuple
import tempfile
import soundfile as sf

# Try to import ChatterboxTTS components
try:
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.s3tokenizer import S3_SR
    from chatterbox.models.s3gen import S3GEN_SR, S3Gen
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.tts import Conditionals
    CHATTERTTS_AVAILABLE = True
except ImportError:
    CHATTERTTS_AVAILABLE = False
    VoiceEncoder = None
    S3_SR = 16000
    S3GEN_SR = 24000


def extract_speaker_embedding(
    audio_input: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
    device: str = "cpu",
    chatterbox_model: Optional[ChatterboxTTS] = None
) -> Tuple[np.ndarray, dict]:
    """
    Extract speaker embedding from audio using ChatterboxTTS VoiceEncoder
    
    This extracts embeddings the EXACT same way ChatterboxTTS does internally.
    
    Args:
        audio_input: Audio file path or numpy array
        sample_rate: Sample rate (if audio_input is numpy array)
        device: Device to run on ("cpu", "cuda", "mps")
        chatterbox_model: Optional pre-loaded ChatterboxTTS model (for efficiency)
    
    Returns:
        Tuple of:
        - speaker_embedding: numpy array (1, embedding_dim) - the actual speaker embedding
        - conditionals_dict: dict containing all conditionals (for direct use with ChatTTS)
    """
    if not CHATTERTTS_AVAILABLE:
        raise ImportError(
            "ChatterboxTTS is required. Install with:\n"
            "pip install git+https://github.com/davidbrowne17/chatterbox-streaming.git"
        )
    
    # Load or create ChatterboxTTS model
    if chatterbox_model is None:
        chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
    
    # Handle different input types
    if isinstance(audio_input, (str, Path)):
        # File path - load audio
        wav_fpath = str(audio_input)
    elif isinstance(audio_input, np.ndarray):
        # Numpy array - save to temp file
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio_input is a numpy array")
        
        temp_path = tempfile.mktemp(suffix='.wav')
        sf.write(temp_path, audio_input, sample_rate)
        wav_fpath = temp_path
    else:
        raise ValueError(f"Unsupported audio_input type: {type(audio_input)}")
    
    try:
        # Extract embeddings using the EXACT same method ChatterboxTTS uses
        # This is what prepare_conditionals() does internally
        
        # 1. Load reference audio at 24kHz (S3GEN_SR)
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        
        # 2. Resample to 16kHz (S3_SR) for voice encoder
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        
        # 3. Trim to max length (10 seconds for decoder, 6 seconds for encoder)
        DEC_COND_LEN = 10 * S3GEN_SR  # 10 seconds at 24kHz
        ENC_COND_LEN = 6 * S3_SR      # 6 seconds at 16kHz
        
        s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]
        ref_16k_wav_trimmed = ref_16k_wav[:ENC_COND_LEN]
        
        # 4. Extract S3Gen reference dictionary
        s3gen_ref_dict = chatterbox_model.s3gen.embed_ref(
            s3gen_ref_wav, 
            S3GEN_SR, 
            device=device
        )
        
        # 5. Extract speech condition prompt tokens (if needed)
        t3_cond_prompt_tokens = None
        if hasattr(chatterbox_model.t3.hp, 'speech_cond_prompt_len'):
            plen = chatterbox_model.t3.hp.speech_cond_prompt_len
            if plen:
                s3_tokzr = chatterbox_model.s3gen.tokenizer
                t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                    [ref_16k_wav_trimmed], 
                    max_len=plen
                )
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(device)
        
        # 6. Extract speaker embedding using VoiceEncoder (THE KEY STEP!)
        ve_embed = torch.from_numpy(
            chatterbox_model.ve.embeds_from_wavs(
                [ref_16k_wav], 
                sample_rate=S3_SR
            )
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(device)
        
        # Convert to numpy for storage
        speaker_embedding = ve_embed.cpu().numpy()
        
        # Create conditionals dict (for direct use with ChatTTS if needed)
        conditionals_dict = {
            'speaker_emb': ve_embed,
            's3gen_ref_dict': s3gen_ref_dict,
            't3_cond_prompt_tokens': t3_cond_prompt_tokens,
            'ref_16k_wav': ref_16k_wav,
            's3gen_ref_wav': s3gen_ref_wav
        }
        
        return speaker_embedding, conditionals_dict
        
    finally:
        # Clean up temp file if we created one
        if isinstance(audio_input, np.ndarray) and 'temp_path' in locals():
            try:
                Path(temp_path).unlink()
            except:
                pass


def extract_and_save_conditionals(
    audio_file: Union[str, Path, np.ndarray],
    output_file: Optional[str] = None,
    device: str = "cpu",
    exaggeration: float = 0.5,
    sample_rate: Optional[int] = None
) -> str:
    """
    Extract full Conditionals object and save (can be loaded directly into ChatTTS)
    
    This creates a Conditionals object that can be loaded directly into ChatterboxTTS,
    allowing you to generate TTS without providing audio_prompt_path.
    
    Args:
        audio_file: Path to audio file or numpy array
        output_file: Path to save conditionals (default: audio_file + _conditionals.pt)
        device: Device to use
        exaggeration: Emotion exaggeration (0.0-1.0)
        sample_rate: Sample rate (required if audio_file is numpy array)
    
    Returns:
        Path to saved conditionals file
    """
    if not CHATTERTTS_AVAILABLE:
        raise ImportError("ChatterboxTTS is required")
    
    # Load ChatterboxTTS model
    model = ChatterboxTTS.from_pretrained(device=device)
    
    # Handle different input types
    temp_path = None
    try:
        if isinstance(audio_file, np.ndarray):
            # If numpy array, save to temp file first
            if sample_rate is None:
                raise ValueError("sample_rate is required when audio_file is a numpy array")
            temp_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_path, audio_file, sample_rate)
            wav_fpath = temp_path
        else:
            wav_fpath = str(audio_file)
        
        # Use the model's prepare_conditionals method (exact same as internal)
        model.prepare_conditionals(wav_fpath, exaggeration=exaggeration)
        
        # Save the conditionals object
        if output_file is None:
            if isinstance(audio_file, np.ndarray):
                output_file = "extracted_conditionals.pt"
            else:
                output_file = str(Path(audio_file).with_suffix('')) + '_conditionals.pt'
        
        model.conds.save(output_file)
        
        print(f"✅ Conditionals saved to: {output_file}")
        print(f"   You can load this directly into ChatterboxTTS!")
        
        return output_file
    finally:
        # Clean up temp file if we created one
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except:
                pass


def extract_and_save_embedding(
    audio_file: Union[str, Path],
    output_file: Optional[str] = None,
    device: str = "cpu"
) -> str:
    """
    Extract embedding and save to file
    
    Args:
        audio_file: Path to audio file
        output_file: Path to save embedding (default: audio_file + .npy)
        device: Device to use
    
    Returns:
        Path to saved embedding file
    """
    embedding, conditionals = extract_speaker_embedding(audio_file, device=device)
    
    if output_file is None:
        output_file = str(Path(audio_file).with_suffix('.npy'))
    
    # Save the speaker embedding
    np.save(output_file, embedding)
    
    # Optionally save full conditionals (larger file, but can be used directly)
    conditionals_file = output_file.replace('.npy', '_conditionals.pt')
    torch.save(conditionals, conditionals_file)
    
    print(f"✅ Speaker embedding saved to: {output_file}")
    print(f"✅ Full conditionals saved to: {conditionals_file}")
    print(f"   Embedding shape: {embedding.shape}")
    
    return output_file


def load_embedding(embedding_file: str) -> np.ndarray:
    """Load saved speaker embedding"""
    return np.load(embedding_file)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Extract Speaker Embeddings using ChatterboxTTS VoiceEncoder")
        print("=" * 60)
        print("\nUsage:")
        print("  python extract_chatterbox_embeddings.py <audio_file> [output_file] [device] [--conditionals]")
        print("\nExamples:")
        print("  # Extract just the embedding")
        print("  python extract_chatterbox_embeddings.py my_voice.wav my_voice_embedding.npy cuda")
        print("\n  # Extract full Conditionals object (can be loaded directly into ChatTTS)")
        print("  python extract_chatterbox_embeddings.py my_voice.wav my_voice_conditionals.pt cuda --conditionals")
        print("\nOr use as a module:")
        print("  from extract_chatterbox_embeddings import extract_speaker_embedding")
        print("  embedding = extract_speaker_embedding('my_voice.wav')")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    device = "cpu"
    use_conditionals = False
    
    # Parse arguments
    for arg in sys.argv[2:]:
        if arg == "--conditionals":
            use_conditionals = True
        elif arg in ["cpu", "cuda", "mps"]:
            device = arg
    
    try:
        if use_conditionals:
            extract_and_save_conditionals(audio_file, output_file, device)
        else:
            extract_and_save_embedding(audio_file, output_file, device)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

