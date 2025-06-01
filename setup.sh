#!/bin/bash

# YunMin-Mamba 3B Project Setup Script
# Automates initial project setup

set -e

echo "🚀 Setting up YunMin-Mamba 3B Training Environment..."

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p checkpoints logs dataset

# Copy .env template if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "📄 Creating .env file from template..."
    cp env.example .env
    echo "✅ .env file created. Please edit it to match your configuration."
else
    echo "✅ .env file already exists."
fi

# Check Docker and Docker Compose
echo "🐳 Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are available."

# Check NVIDIA Docker runtime
echo "🖥️  Checking NVIDIA Docker runtime..."
if docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "✅ NVIDIA Docker runtime is working."
else
    echo "⚠️  NVIDIA Docker runtime check failed. Please ensure:"
    echo "   - NVIDIA drivers are installed"
    echo "   - NVIDIA Container Toolkit is installed"
    echo "   - Docker daemon is configured to use nvidia runtime"
fi

# Check AWS CLI
echo "☁️  Checking AWS CLI..."
if command -v aws &> /dev/null; then
    echo "✅ AWS CLI is available."
    if aws sts get-caller-identity > /dev/null 2>&1; then
        echo "✅ AWS credentials are configured."
    else
        echo "⚠️  AWS credentials not configured. Run 'aws configure' or use IAM roles."
    fi
else
    echo "⚠️  AWS CLI is not installed. Install it for S3 dataset access."
fi

# Validate config files
if [ ! -f "mamba_config.json" ]; then
    echo "⚠️  mamba_config.json not found. Please ensure model config is available."
fi

if [ ! -f "accelerate_config.yaml" ]; then
    echo "⚠️  accelerate_config.yaml not found. Please ensure accelerate config is available."
fi

echo ""
echo "🎉 Setup completed!"
echo ""
echo "📋 Next steps:"
echo "  1. Edit .env file to configure your S3 bucket and paths"
echo "  2. Ensure mamba_config.json and accelerate_config.yaml exist"
echo "  3. Run: docker-compose up --build yunmin-mamba-train"
echo ""
echo "📚 Documentation: See README.md for detailed usage instructions" 