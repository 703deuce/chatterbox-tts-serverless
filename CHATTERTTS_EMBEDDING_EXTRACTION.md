# How ChatterboxTTS Extracts Speaker Embeddings

## The Exact Process

Based on the ChatterboxTTS source code, here's exactly how it extracts embeddings:

### Step-by-Step Process (from `prepare_conditionals` method)

```python
# 1. Load audio at 24kHz (S3GEN_SR)
s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)  # 24000 Hz

# 2. Resample to 16kHz (S3_SR) for voice encoder
ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)  # 16000 Hz

# 3. Trim to max lengths
DEC_COND_LEN = 10 * S3GEN_SR  # 10 seconds at 24kHz
ENC_COND_LEN = 6 * S3_SR       # 6 seconds at 16kHz
s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]
ref_16k_wav = ref_16k_wav[:ENC_COND_LEN]

# 4. Extract S3Gen reference dictionary (for audio generation)
s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=device)

# 5. Extract speech condition prompt tokens (optional, for T3 model)
if speech_cond_prompt_len:
    t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav], max_len=plen)

# 6. Extract speaker embedding using VoiceEncoder (THE KEY!)
ve_embed = torch.from_numpy(
    self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
)
ve_embed = ve_embed.mean(axis=0, keepdim=True).to(device)

# 7. Create T3Cond object
t3_cond = T3Cond(
    speaker_emb=ve_embed,  # <-- This is the speaker embedding!
    cond_prompt_speech_tokens=t3_cond_prompt_tokens,
    emotion_adv=exaggeration * torch.ones(1, 1, 1),
)

# 8. Create Conditionals object
self.conds = Conditionals(t3_cond, s3gen_ref_dict)
```

## Key Components

### 1. VoiceEncoder (`self.ve`)
- **Purpose**: Extracts speaker embeddings from audio
- **Input**: Audio at 16kHz (S3_SR)
- **Output**: Speaker embedding tensor
- **Method**: `ve.embeds_from_wavs([audio], sample_rate=16000)`

### 2. S3Gen Reference Dictionary
- **Purpose**: Reference audio features for generation
- **Input**: Audio at 24kHz (S3GEN_SR), max 10 seconds
- **Output**: Dictionary with audio features
- **Method**: `s3gen.embed_ref(audio, S3GEN_SR, device)`

### 3. Speech Condition Tokens
- **Purpose**: Speech prompt tokens for T3 model
- **Input**: Audio at 16kHz, max 6 seconds
- **Output**: Token sequence
- **Method**: `s3_tokenizer.forward([audio], max_len=plen)`

## Using Extracted Embeddings

### Option 1: Extract and Store Embeddings (Recommended)

```python
from extract_chatterbox_embeddings import extract_speaker_embedding
import numpy as np

# Extract embedding
embedding, conditionals = extract_speaker_embedding("my_voice.wav", device="cuda")

# Save embedding
np.save("my_voice_embedding.npy", embedding)

# Later, use with ChatTTS
from chatterbox.tts import ChatterboxTTS
import torch

model = ChatterboxTTS.from_pretrained(device="cuda")

# Load embedding
embedding = np.load("my_voice_embedding.npy")
embedding_tensor = torch.from_numpy(embedding).to("cuda")

# Create T3Cond with the embedding
from chatterbox.models.t3.modules.cond_enc import T3Cond
t3_cond = T3Cond(
    speaker_emb=embedding_tensor,
    cond_prompt_speech_tokens=None,  # Optional
    emotion_adv=0.5 * torch.ones(1, 1, 1),
).to(device="cuda")

# You'd need to also create s3gen_ref_dict, but this is complex
# So Option 2 is simpler...
```

### Option 2: Store Full Conditionals Object (Easier)

```python
from extract_chatterbox_embeddings import extract_speaker_embedding
import torch

# Extract and save full conditionals
embedding, conditionals = extract_speaker_embedding("my_voice.wav", device="cuda")

# Save conditionals (includes everything ChatTTS needs)
torch.save(conditionals, "my_voice_conditionals.pt")

# Later, load and use
from chatterbox.tts import ChatterboxTTS, Conditionals

model = ChatterboxTTS.from_pretrained(device="cuda")

# Load conditionals
conds = Conditionals.load("my_voice_conditionals.pt", map_location="cuda")
model.conds = conds

# Now generate without audio_prompt_path!
wav = model.generate(
    text="Hello world!",
    # No audio_prompt_path needed - using pre-loaded conditionals!
    exaggeration=0.5
)
```

### Option 3: Store Audio Array (Current Approach - Simplest)

```python
# This is what you're currently doing - and it's correct!
# Store raw audio array, ChatTTS extracts embeddings internally

import librosa
import numpy as np

# Load and resample audio
audio, sr = librosa.load("my_voice.wav", sr=24000)

# Save audio array
np.save("my_voice_audio.npy", audio)

# Later, use with ChatTTS
import tempfile
import soundfile as sf

# Load audio
audio = np.load("my_voice_audio.npy")

# Save to temp file for ChatTTS
temp_path = tempfile.mktemp(suffix='.wav')
sf.write(temp_path, audio, 24000)

# Use with ChatTTS
wav = model.generate(text="Hello", audio_prompt_path=temp_path)
```

## Which Approach to Use?

### ✅ **Option 3 (Current) - RECOMMENDED**
- **Pros**: Simplest, no embedding extraction needed, works perfectly
- **Cons**: Slightly larger files (audio vs embeddings)
- **When to use**: Always - it's the simplest and works great

### Option 2 (Full Conditionals)
- **Pros**: Can skip embedding extraction during generation
- **Cons**: More complex, requires ChatTTS to load conditionals
- **When to use**: If you want to avoid embedding extraction on every generation

### Option 1 (Just Embeddings)
- **Pros**: Smallest file size
- **Cons**: Most complex - need to reconstruct full Conditionals object
- **When to use**: Only if file size is critical and you're willing to handle complexity

## The Reality

**Your current approach (storing raw audio) is actually the BEST approach** because:

1. ✅ **Simpler** - No embedding extraction logic needed
2. ✅ **More flexible** - Can use audio for other purposes
3. ✅ **Works perfectly** - ChatTTS extracts embeddings internally anyway
4. ✅ **No performance loss** - Embedding extraction is fast
5. ✅ **Easier to maintain** - Less code, fewer dependencies

The only reason to extract embeddings separately would be if:
- You want to avoid loading ChatTTS model in the standalone code
- You need very small file sizes
- You want to pre-compute embeddings for faster generation

But for your use case (external app generating embeddings), **storing raw audio is perfect**!

