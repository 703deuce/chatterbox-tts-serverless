# Summary: How ChatterboxTTS Creates Embeddings

## The Answer

ChatterboxTTS extracts speaker embeddings using its **VoiceEncoder** model. Here's the exact process:

### Internal Process (from `ChatterboxTTS.prepare_conditionals()`)

1. **Load audio at 24kHz** (S3GEN_SR = 24000 Hz)
2. **Resample to 16kHz** (S3_SR = 16000 Hz) for voice encoder
3. **Extract speaker embedding** using `VoiceEncoder.embeds_from_wavs()`
4. **Extract S3Gen reference dictionary** for audio generation
5. **Extract speech condition tokens** (optional, for T3 model)
6. **Create Conditionals object** containing all of the above

### Key Code (from ChatterboxTTS source):

```python
# Load and resample audio
s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)  # 24kHz
ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)  # 16kHz

# Extract speaker embedding (THE KEY STEP!)
ve_embed = torch.from_numpy(
    self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
)
ve_embed = ve_embed.mean(axis=0, keepdim=True).to(device)

# Create conditionals
t3_cond = T3Cond(
    speaker_emb=ve_embed,  # <-- This is the speaker embedding!
    cond_prompt_speech_tokens=t3_cond_prompt_tokens,
    emotion_adv=exaggeration * torch.ones(1, 1, 1),
)
self.conds = Conditionals(t3_cond, s3gen_ref_dict)
```

## Files Created

### 1. `extract_chatterbox_embeddings.py`
Standalone script that extracts embeddings **the exact same way** ChatterboxTTS does internally.

**Features:**
- ✅ Uses ChatterboxTTS's VoiceEncoder directly
- ✅ Extracts speaker embeddings
- ✅ Can save full Conditionals object (for direct use with ChatTTS)
- ✅ Can save just the embedding
- ✅ Works with audio files or numpy arrays

**Usage:**
```python
from extract_chatterbox_embeddings import extract_speaker_embedding, extract_and_save_conditionals

# Extract full conditionals (can be loaded directly into ChatTTS)
extract_and_save_conditionals("my_voice.wav", "my_voice_conditionals.pt", device="cuda")

# Or extract just the embedding
embedding, conditionals = extract_speaker_embedding("my_voice.wav", device="cuda")
```

### 2. `CHATTERTTS_EMBEDDING_EXTRACTION.md`
Detailed documentation explaining:
- Exact embedding extraction process
- All components involved (VoiceEncoder, S3Gen, etc.)
- How to use extracted embeddings
- Comparison of different approaches

### 3. `USE_EXTRACTED_EMBEDDINGS.md`
Guide on how to use extracted embeddings with ChatterboxTTS:
- Method 1: Using full Conditionals object (recommended)
- Method 2: Using just the embedding (more complex)
- Method 3: Storing raw audio (simplest - your current approach)

## How to Use Extracted Embeddings

### Option A: Extract Full Conditionals (Direct Embedding Use)

```python
# 1. Extract conditionals
from extract_chatterbox_embeddings import extract_and_save_conditionals
extract_and_save_conditionals("my_voice.wav", "my_voice_conditionals.pt", device="cuda")

# 2. Load and use with ChatTTS
from chatterbox.tts import ChatterboxTTS, Conditionals

model = ChatterboxTTS.from_pretrained(device="cuda")
conds = Conditionals.load("my_voice_conditionals.pt", map_location="cuda")
model.conds = conds

# 3. Generate without audio_prompt_path!
wav = model.generate(text="Hello world!")
```

### Option B: Store Raw Audio (Current Approach - Simplest)

```python
# This is what you're currently doing - and it's correct!
import librosa
import numpy as np
import tempfile
import soundfile as sf

# Load and save audio
audio, sr = librosa.load("my_voice.wav", sr=24000)
np.save("my_voice_audio.npy", audio)

# Later, use with ChatTTS
model = ChatterboxTTS.from_pretrained(device="cuda")
audio = np.load("my_voice_audio.npy")

temp_path = tempfile.mktemp(suffix='.wav')
sf.write(temp_path, audio, 24000)
wav = model.generate(text="Hello", audio_prompt_path=temp_path)
```

## Recommendation

**Your current approach (storing raw audio) is actually the BEST approach** because:

1. ✅ **Simplest** - No embedding extraction needed
2. ✅ **No ChatTTS dependency** in standalone code
3. ✅ **Works perfectly** - ChatTTS extracts embeddings internally
4. ✅ **Fast** - Embedding extraction is quick
5. ✅ **Flexible** - Can use audio for other purposes

**Only use embedding extraction if:**
- You want to avoid loading ChatTTS model in standalone code
- You need very small file sizes
- You want to pre-compute for faster generation

## Next Steps

1. **If you want to extract embeddings separately:**
   - Use `extract_chatterbox_embeddings.py`
   - See `USE_EXTRACTED_EMBEDDINGS.md` for usage

2. **If you want to keep current approach (recommended):**
   - Continue using `generate_voice_embedding.py` (stores raw audio)
   - It works perfectly as-is!

3. **If you want to understand the internals:**
   - Read `CHATTERTTS_EMBEDDING_EXTRACTION.md`
   - Check the ChatterboxTTS source code at:
     `C:\Users\Owner\AppData\Roaming\Python\Python311\site-packages\chatterbox\tts.py`

