# Use CUDA-enabled Python base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Upgrade pip first
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements and install all dependencies (search both PyPI and PyTorch index)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Copy handler code
COPY handler.py .

# Expose port for the serverless handler
EXPOSE 8000

# Run the handler
CMD ["python3", "handler.py"] 