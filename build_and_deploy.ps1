# Chatterbox TTS Serverless Build and Deploy Script (PowerShell)
# Usage: .\build_and_deploy.ps1 <your-dockerhub-username>

param(
    [Parameter(Mandatory=$true)]
    [string]$Username
)

# Colors for output
function Write-ColorText {
    param(
        [string]$Text,
        [string]$Color = "White"
    )
    Write-Host $Text -ForegroundColor $Color
}

$IMAGE_NAME = "chatterbox-tts-serverless"
$FULL_IMAGE_NAME = "${Username}/${IMAGE_NAME}"

Write-ColorText "🚀 Building and deploying Chatterbox TTS Serverless API" "Green"
Write-ColorText "Docker Hub Username: $Username" "Yellow"
Write-ColorText "Image Name: $FULL_IMAGE_NAME" "Yellow"
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
    Write-ColorText "✅ Docker is running" "Green"
} catch {
    Write-ColorText "❌ Docker is not running. Please start Docker and try again." "Red"
    exit 1
}

# Build the Docker image
Write-ColorText "🔨 Building Docker image..." "Green"
docker build -t "${FULL_IMAGE_NAME}:latest" .

if ($LASTEXITCODE -eq 0) {
    Write-ColorText "✅ Docker image built successfully!" "Green"
} else {
    Write-ColorText "❌ Docker build failed!" "Red"
    exit 1
}

# Check if user wants to push to Docker Hub
Write-Host ""
$pushChoice = Read-Host "Do you want to push the image to Docker Hub? (y/n)"

if ($pushChoice -match "^[Yy]$") {
    Write-ColorText "🚀 Pushing to Docker Hub..." "Green"
    docker push "${FULL_IMAGE_NAME}:latest"
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorText "✅ Image pushed successfully!" "Green"
        Write-Host ""
        Write-ColorText "🎉 Deployment ready!" "Green"
        Write-ColorText "Next steps:" "Yellow"
        Write-Host "1. Go to https://runpod.io/serverless"
        Write-Host "2. Create a new endpoint"
        Write-Host "3. Use this Docker image: ${FULL_IMAGE_NAME}:latest"
        Write-Host "4. Select a GPU (recommended: RTX 4090 or A100)"
        Write-Host "5. Configure settings:"
        Write-Host "   - Container Disk: 20GB minimum"
        Write-Host "   - Memory: 16GB minimum"
        Write-Host "   - Timeout: 300 seconds"
        Write-Host "6. Deploy and test with test_api.py"
    } else {
        Write-ColorText "❌ Push failed!" "Red"
        exit 1
    }
} else {
    Write-ColorText "⏭️  Skipping push to Docker Hub" "Yellow"
    Write-ColorText "✅ Build completed successfully!" "Green"
    Write-Host ""
    Write-ColorText "To push later, run:" "Yellow"
    Write-Host "docker push ${FULL_IMAGE_NAME}:latest"
}

Write-Host ""
Write-ColorText "📋 Summary:" "Green"
Write-Host "- Image built: ${FULL_IMAGE_NAME}:latest"
Write-Host "- Files created:"
Write-Host "  - handler.py (Serverless handler)"
Write-Host "  - Dockerfile (Container configuration)"
Write-Host "  - requirements.txt (Dependencies)"
Write-Host "  - test_api.py (Testing script)"
Write-Host "  - README.md (Documentation)"
Write-Host ""
Write-ColorText "🧪 To test locally:" "Green"
Write-Host "docker run --rm -it --gpus all -p 8080:8080 ${FULL_IMAGE_NAME}:latest" 