#!/usr/bin/env bash

set -e  # Exit immediately on error

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PYTHON_VERSION=3.10
VENV_NAME=tts-venv
REPO_URL="https://github.com/Svastikkka/TTS"

echo "🚀 Starting TTS VM setup..."

# -----------------------------------------------------------------------------
# Update and install system dependencies
# -----------------------------------------------------------------------------
echo "📦 Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    build-essential \
    git \
    wget \
    curl \
    libsndfile1 \
    libnvinfer8 libnvinfer-dev libnvinfer-plugin8 \
    && sudo apt-get clean

# -----------------------------------------------------------------------------
# Install NVIDIA CUDA + cuDNN + TensorRT (if not already installed)
# -----------------------------------------------------------------------------
echo "🔧 Checking for NVIDIA driver & CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA driver not found. Please install GPU driver before running."
    exit 1
else
    echo "✅ NVIDIA driver detected:"
    nvidia-smi
fi

# NOTE: Assuming CUDA & TensorRT are pre-installed in VM image or via NVIDIA repo.
# For manual installation, follow: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

# -----------------------------------------------------------------------------
# Clone Repository
# -----------------------------------------------------------------------------
if [ ! -d "TTS" ]; then
    echo "📥 Cloning repository..."
    git clone $REPO_URL
else
    echo "🔄 Repository already exists. Pulling latest changes..."
    cd TTS && git pull && cd ..
fi

cd TTS

# -----------------------------------------------------------------------------
# Create Virtual Environment
# -----------------------------------------------------------------------------
echo "🐍 Creating Python virtual environment..."
python${PYTHON_VERSION} -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# -----------------------------------------------------------------------------
# Upgrade pip & install Python dependencies
# -----------------------------------------------------------------------------
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Final Instructions
# -----------------------------------------------------------------------------
echo "✅ Setup complete!"
echo "Activate your virtual environment with:"
echo "  source $VENV_NAME/bin/activate"
echo "Run the TTS server with:"
echo "  python src/main.py"
