#!/bin/bash

# Chatterbox TTS Serverless Build and Deploy Script
# Usage: ./build_and_deploy.sh <your-dockerhub-username>

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if username is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Docker Hub username not provided${NC}"
    echo "Usage: $0 <your-dockerhub-username>"
    echo "Example: $0 myusername"
    exit 1
fi

USERNAME=$1
IMAGE_NAME="chatterbox-tts-serverless"
FULL_IMAGE_NAME="${USERNAME}/${IMAGE_NAME}"

echo -e "${GREEN}🚀 Building and deploying Chatterbox TTS Serverless API${NC}"
echo -e "${YELLOW}Docker Hub Username: ${USERNAME}${NC}"
echo -e "${YELLOW}Image Name: ${FULL_IMAGE_NAME}${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Build the Docker image
echo -e "${GREEN}🔨 Building Docker image...${NC}"
docker build -t "${FULL_IMAGE_NAME}:latest" .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully!${NC}"
else
    echo -e "${RED}❌ Docker build failed!${NC}"
    exit 1
fi

# Check if user wants to push to Docker Hub
echo ""
read -p "Do you want to push the image to Docker Hub? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}🚀 Pushing to Docker Hub...${NC}"
    docker push "${FULL_IMAGE_NAME}:latest"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Image pushed successfully!${NC}"
        echo ""
        echo -e "${GREEN}🎉 Deployment ready!${NC}"
        echo -e "${YELLOW}Next steps:${NC}"
        echo "1. Go to https://runpod.io/serverless"
        echo "2. Create a new endpoint"
        echo "3. Use this Docker image: ${FULL_IMAGE_NAME}:latest"
        echo "4. Select a GPU (recommended: RTX 4090 or A100)"
        echo "5. Configure settings:"
        echo "   - Container Disk: 20GB minimum"
        echo "   - Memory: 16GB minimum"
        echo "   - Timeout: 300 seconds"
        echo "6. Deploy and test with test_api.py"
    else
        echo -e "${RED}❌ Push failed!${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⏭️  Skipping push to Docker Hub${NC}"
    echo -e "${GREEN}✅ Build completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}To push later, run:${NC}"
    echo "docker push ${FULL_IMAGE_NAME}:latest"
fi

echo ""
echo -e "${GREEN}📋 Summary:${NC}"
echo "- Image built: ${FULL_IMAGE_NAME}:latest"
echo "- Files created:"
echo "  - handler.py (Serverless handler)"
echo "  - Dockerfile (Container configuration)"
echo "  - requirements.txt (Dependencies)"
echo "  - test_api.py (Testing script)"
echo "  - README.md (Documentation)"
echo ""
echo -e "${GREEN}🧪 To test locally:${NC}"
echo "docker run --rm -it --gpus all -p 8080:8080 ${FULL_IMAGE_NAME}:latest" 