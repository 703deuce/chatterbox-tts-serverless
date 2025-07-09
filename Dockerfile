# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Cachebuster to force fresh builds
ARG CACHEBUSTER=1

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

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the handler
CMD ["python3", "-u", "handler.py"] 