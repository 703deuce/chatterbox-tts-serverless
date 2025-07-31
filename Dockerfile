# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Cachebuster to force fresh builds - COMPLETE OPTIMIZATION: All Voice Embedding Operations
ARG CACHEBUSTER=2025012702

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/workspace"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    pkg-config \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Note: PyTorch will be installed by chatterbox-streaming with correct versions (torch==2.6.0)

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (chatterbox-streaming will handle PyTorch 2.6.0 + CUDA)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Debug: Verify chatterbox installation
RUN python3 -c "import chatterbox; print('✅ chatterbox imported successfully')" || echo "❌ chatterbox import failed"
RUN pip show chatterbox-tts || echo "❌ chatterbox-tts not found in pip list"

# Create checkpoints directory and download S3Gen model
RUN mkdir -p checkpoints
RUN echo "Downloading S3Gen model checkpoint (approx 1.06 GB)..."
RUN wget -O checkpoints/s3gen.pt https://huggingface.co/ResembleAI/chatterbox/resolve/main/s3gen.pt
RUN echo "S3Gen model download completed. File size:" && ls -lh checkpoints/s3gen.pt

# Download F5-TTS models during build (like S3Gen)
RUN echo "Downloading F5-TTS models..."
RUN mkdir -p /root/.cache/huggingface/hub
RUN pip install huggingface_hub
RUN python3 -c "
from huggingface_hub import snapshot_download; \
import os; \
print('Downloading F5-TTS base model...'); \
try: \
    snapshot_download( \
        repo_id='SWivid/F5-TTS', \
        local_dir='/workspace/f5_models', \
        allow_patterns=['F5TTS_v1_Base/*'] \
    ); \
    print('✅ F5-TTS model downloaded successfully'); \
    import os; \
    for root, dirs, files in os.walk('/workspace/f5_models'): \
        for file in files: \
            print(f'  {os.path.join(root, file)}') \
except Exception as e: \
    print(f'❌ F5-TTS download failed: {e}') \
"

# Copy application code
COPY . .

# Force fresh copy of voice embeddings (cache-busting for voice library updates)
ARG VOICE_EMBEDDINGS_VERSION=a42e2c6
RUN echo "Voice embeddings version: $VOICE_EMBEDDINGS_VERSION" > /workspace/voice_embeddings_version.txt

# Expose port
EXPOSE 8000

# Run the handler
CMD ["python3", "-u", "handler.py"] 