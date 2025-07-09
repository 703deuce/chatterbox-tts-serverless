# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

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

# Install PyTorch with CUDA support first (to avoid conflicts)
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the handler
CMD ["python", "-u", "handler.py"] 