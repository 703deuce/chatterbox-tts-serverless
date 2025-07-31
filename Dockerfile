# Use specific Python version for F5-TTS compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Force rebuild timestamp: 2025-07-31-02:50
RUN echo "Build timestamp: $(date)" && echo "F5-TTS integration build"

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

# F5-TTS models setup (following official F5-TTS Docker pattern)
# Updated: 2025-07-31 - Fixed Docker parse error, using VOLUME approach
RUN echo "Setting up F5-TTS model cache directory..."
RUN mkdir -p /root/.cache/huggingface/hub
# Models are downloaded into this folder, F5-TTS will download them at runtime if needed
VOLUME /root/.cache/huggingface/hub/

# Copy application code
COPY . .

# Force fresh copy of voice embeddings (cache-busting for voice library updates)
ARG VOICE_EMBEDDINGS_VERSION=a42e2c6
RUN echo "Voice embeddings version: $VOICE_EMBEDDINGS_VERSION" > /workspace/voice_embeddings_version.txt

# Expose port
EXPOSE 8000

# Run the handler
CMD ["python3", "-u", "handler.py"] 