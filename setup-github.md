# Setup Guide: Chatterbox TTS Serverless API with GitHub + Runpod

## 📋 Prerequisites

- GitHub account
- Runpod account
- All project files ready (handler.py, Dockerfile, requirements.txt, etc.)

## 🚀 Step-by-Step Setup

### Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it something like `chatterbox-tts-serverless`
3. Make it **public** (easier for Runpod to access)
4. Don't initialize with README (we have our own files)

### Step 2: Upload Project Files

**Option A: GitHub Web Interface**
1. Click "uploading an existing file"
2. Drag and drop all these files:
   - `handler.py`
   - `Dockerfile`
   - `requirements.txt`
   - `README.md`
   - `test_api.py`
3. Commit the files

**Option B: Git Command Line**
```bash
# In your project directory (C:\Users\Owner\chatter)
git init
git add .
git commit -m "Initial commit - Chatterbox TTS Serverless API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/chatterbox-tts-serverless.git
git push -u origin main
```

### Step 3: Deploy on Runpod Serverless

1. **Go to Runpod Dashboard**
   - Visit [runpod.io](https://runpod.io)
   - Navigate to "Serverless" section

2. **Create New Endpoint**
   - Click "Create Endpoint"
   - Choose "GitHub" as the source

3. **Configure GitHub Integration**
   - **Repository**: `YOUR_USERNAME/chatterbox-tts-serverless`
   - **Branch**: `main`
   - **Dockerfile Path**: `Dockerfile` (default)

4. **Configure Hardware**
   - **GPU**: RTX 4090 or A100 (recommended)
   - **Container Disk**: 25GB (for model downloads)
   - **Memory**: 16GB minimum

5. **Configure Settings**
   - **Timeout**: 600 seconds (10 minutes)
   - **Max Workers**: 1-3 (depending on your needs)
   - **Environment Variables**: None needed

6. **Deploy**
   - Click "Deploy"
   - Wait for build to complete (5-10 minutes)

### Step 4: Test Your Endpoint

1. **Get Your Endpoint URL**
   - Copy the endpoint URL from Runpod dashboard
   - Format: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync`

2. **Update Test Script**
   ```powershell
   # Set environment variables
   $env:RUNPOD_API_KEY = "your_api_key_here"
   $env:ENDPOINT_ID = "your_endpoint_id_here"
   
   # Run tests
   python test_api.py
   ```

### Step 5: Monitor and Debug

1. **Check Logs**
   - View build logs in Runpod dashboard
   - Monitor runtime logs for errors

2. **Common Issues**
   - **Build failures**: Check Dockerfile syntax
   - **Runtime errors**: Check handler.py imports
   - **Timeout errors**: Increase timeout in settings

## 📁 Required Files in Repository

Make sure your GitHub repository contains:

```
chatterbox-tts-serverless/
├── handler.py              # Main serverless handler
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── test_api.py           # Test script
└── build_and_deploy.ps1  # Build script (optional)
```

## 🔧 Updating Your Deployment

To update your API:

1. **Make changes** to your local files
2. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Update API features"
   git push origin main
   ```
3. **Runpod automatically rebuilds** your endpoint

## 💡 Pro Tips

1. **Use GitHub Releases** for stable versions
2. **Enable branch protection** for production
3. **Use environment variables** for sensitive data
4. **Monitor costs** in Runpod dashboard
5. **Set up alerts** for high usage

## 🆚 Alternative: Docker Hub Option

If you prefer Docker Hub:

1. **Build and push image**:
   ```bash
   docker build -t your-username/chatterbox-tts-serverless .
   docker push your-username/chatterbox-tts-serverless
   ```

2. **Use Docker option** in Runpod
   - Image: `your-username/chatterbox-tts-serverless:latest`

## 🎯 Next Steps

1. **Test all endpoints** with comprehensive parameters
2. **Monitor performance** and optimize if needed
3. **Set up monitoring** for production use
4. **Scale** workers based on demand

## 🆘 Troubleshooting

- **Build fails**: Check Dockerfile and requirements.txt
- **Handler errors**: Verify imports and model loading
- **Timeout issues**: Increase timeout or optimize model loading
- **VRAM errors**: Use smaller GPU or optimize batch sizes

Your Chatterbox TTS API will be ready for production use! 