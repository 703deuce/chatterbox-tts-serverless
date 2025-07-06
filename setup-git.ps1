# Quick GitHub Setup Script for Chatterbox TTS Serverless API
# Usage: .\setup-git.ps1

Write-Host "Setting up GitHub repository for Chatterbox TTS Serverless API" -ForegroundColor Green
Write-Host ""

# Check if we have the required files
$requiredFiles = @("handler.py", "Dockerfile", "requirements.txt", "README.md", "test_api.py")
$missingFiles = @()

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "Missing required files:" -ForegroundColor Red
    $missingFiles | ForEach-Object { Write-Host "   - $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Please ensure all files are in the current directory." -ForegroundColor Yellow
    exit 1
}

Write-Host "All required files found!" -ForegroundColor Green
Write-Host ""

# Get repository details
$username = Read-Host "Enter your GitHub username"
$repoName = Read-Host "Enter repository name (default: chatterbox-tts-serverless)"

if ([string]::IsNullOrWhiteSpace($repoName)) {
    $repoName = "chatterbox-tts-serverless"
}

$repoUrl = "https://github.com/$username/$repoName.git"

Write-Host ""
Write-Host "Repository Setup:" -ForegroundColor Cyan
Write-Host "   Username: $username" -ForegroundColor White
Write-Host "   Repository: $repoName" -ForegroundColor White
Write-Host "   URL: $repoUrl" -ForegroundColor White
Write-Host ""

# Confirm setup
$confirm = Read-Host "Proceed with Git setup? (y/n)"
if ($confirm -notmatch "^[Yy]$") {
    Write-Host "Setup cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Setting up Git repository..." -ForegroundColor Green

try {
    # Initialize git repo
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    
    # Add all files
    Write-Host "Adding files..." -ForegroundColor Yellow
    git add .
    
    # Create initial commit
    Write-Host "Creating initial commit..." -ForegroundColor Yellow
    git commit -m "Initial commit - Chatterbox TTS Serverless API"
    
    # Set main branch
    Write-Host "Setting main branch..." -ForegroundColor Yellow
    git branch -M main
    
    # Add remote origin
    Write-Host "Adding remote origin..." -ForegroundColor Yellow
    git remote add origin $repoUrl
    
    # Push to GitHub
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    git push -u origin main
    
    Write-Host ""
    Write-Host "Git setup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "1. Go to https://runpod.io/serverless" -ForegroundColor White
    Write-Host "2. Create new endpoint" -ForegroundColor White
    Write-Host "3. Choose 'GitHub' as source" -ForegroundColor White
    Write-Host "4. Repository: $username/$repoName" -ForegroundColor White
    Write-Host "5. Branch: main" -ForegroundColor White
    Write-Host "6. Configure hardware (RTX 4090, 25GB disk, 16GB memory)" -ForegroundColor White
    Write-Host "7. Deploy!" -ForegroundColor White
    Write-Host ""
    Write-Host "Full setup guide: See setup-github.md" -ForegroundColor Cyan
    
} catch {
    Write-Host ""
    Write-Host "Error during Git setup:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual steps:" -ForegroundColor Yellow
    Write-Host "1. Create repository manually on GitHub: https://github.com/new" -ForegroundColor White
    Write-Host "2. Upload files using GitHub web interface" -ForegroundColor White
    Write-Host "3. Follow setup-github.md for Runpod deployment" -ForegroundColor White
} 