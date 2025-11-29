# How to Use Extracted Embeddings with ChatterboxTTS

## Overview

After extracting embeddings using `extract_chatterbox_embeddings.py`, you can use them directly with ChatterboxTTS without providing audio files.

## Method 1: Using Full Conditionals Object (Recommended for Direct Embedding Use)

### Step 1: Extract and Save Conditionals

```python
from extract_chatterbox_embeddings import extract_and_save_conditionals

# Extract full Conditionals object
extract_and_save_conditionals(
    audio_file="my_voice.wav",
    output_file="my_voice_conditionals.pt",
    device="cuda"
)
```

### Step 2: Load and Use with ChatterboxTTS

```python
from chatterbox.tts import ChatterboxTTS, Conditionals

# Load ChatterboxTTS model
model = ChatterboxTTS.from_pretrained(device="cuda")

# Load the saved conditionals
conds = Conditionals.load("my_voice_conditionals.pt", map_location="cuda")
model.conds = conds

# Generate TTS WITHOUT audio_prompt_path!
wav = model.generate(
    text="Hello world! This is using a pre-extracted embedding!",
    # No audio_prompt_path needed!
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8
)
```

## Method 2: Using Just the Speaker Embedding

If you only extracted the speaker embedding (not full conditionals), you need to reconstruct the Conditionals object:

```python
from chatterbox.tts import ChatterboxTTS, Conditionals
from chatterbox.models.t3.modules.cond_enc import T3Cond
import torch
import numpy as np

# Load model
model = ChatterboxTTS.from_pretrained(device="cuda")

# Load extracted embedding
embedding = np.load("my_voice_embedding.npy")
embedding_tensor = torch.from_numpy(embedding).to("cuda")

# Create T3Cond (you still need s3gen_ref_dict though)
# This is more complex - Method 1 is easier
t3_cond = T3Cond(
    speaker_emb=embedding_tensor,
    cond_prompt_speech_tokens=None,
    emotion_adv=0.5 * torch.ones(1, 1, 1),
).to(device="cuda")

# You'd need to also extract s3gen_ref_dict...
# So Method 1 (full conditionals) is recommended
```

## Method 3: Store Raw Audio (Current Approach - Simplest)

This is what you're currently doing, and it's actually the **simplest and best approach**:

```python
import librosa
import numpy as np
import tempfile
import soundfile as sf
from chatterbox.tts import ChatterboxTTS

# Load and resample audio
audio, sr = librosa.load("my_voice.wav", sr=24000)

# Save audio array
np.save("my_voice_audio.npy", audio)

# Later, use with ChatTTS
model = ChatterboxTTS.from_pretrained(device="cuda")

# Load audio
audio = np.load("my_voice_audio.npy")

# Save to temp file
temp_path = tempfile.mktemp(suffix='.wav')
sf.write(temp_path, audio, 24000)

# Use with ChatTTS (it extracts embeddings internally)
wav = model.generate(text="Hello", audio_prompt_path=temp_path)
```

## Comparison

| Method | Complexity | File Size | Speed | Use Case |
|--------|-----------|-----------|-------|----------|
| **Method 3 (Raw Audio)** | ⭐ Simplest | Medium | Fast | ✅ **Recommended** |
| Method 1 (Full Conditionals) | Medium | Large | Fastest | Pre-compute for speed |
| Method 2 (Just Embedding) | ⭐⭐⭐ Complex | Smallest | Medium | Only if file size critical |

## Recommendation

**Use Method 3 (Raw Audio)** - it's:
- ✅ Simplest to implement
- ✅ No ChatTTS needed in standalone code
- ✅ Works perfectly
- ✅ ChatTTS extracts embeddings internally anyway (fast)

Only use Method 1 if you want to avoid embedding extraction during generation (marginal speed gain).

