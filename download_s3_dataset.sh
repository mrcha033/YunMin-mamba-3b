#!/bin/bash

# YunMin-Mamba S3 Dataset Download Script (Environment + Parameter Support)
# Usage: ./download_s3_dataset.sh [S3_BUCKET] [S3_PATH] [LOCAL_PATH]
# Or set variables in .env file

set -e  # Exit on any error

# Load environment variables from .env if available
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuration with defaults (command line args override env vars)
S3_BUCKET="${1:-${S3_BUCKET:-yeongjopt-ai-bucket}}"
S3_PATH="${2:-${S3_PATH:-dataset/tagged}}"
LOCAL_PATH="${3:-${LOCAL_DATASET_PATH:-dataset/tagged}}"

echo "🚀 Starting S3 dataset download for YunMin-Mamba training..."
echo "📍 S3 URI: s3://${S3_BUCKET}/${S3_PATH}"
echo "📂 Local Path: ${LOCAL_PATH}"

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    echo "❌ Error: AWS CLI is not installed"
    echo "Please install AWS CLI first"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ Error: AWS CLI is not configured or credentials are invalid"
    echo "Please configure AWS CLI with: aws configure"
    echo "Or use IAM Instance Role on EC2"
    exit 1
fi

# Display AWS identity
echo "✅ AWS Identity:"
aws sts get-caller-identity --output table

# Create local directory
echo "📁 Creating local directory: ${LOCAL_PATH}"
mkdir -p "${LOCAL_PATH}"

# Download dataset with progress
echo "📥 Downloading dataset..."
aws s3 cp "s3://${S3_BUCKET}/${S3_PATH}/" "${LOCAL_PATH}/" \
    --recursive \
    --progress \
    --no-follow-symlinks

# Verify download
if [ $? -eq 0 ]; then
    echo "✅ Dataset download completed successfully!"
    echo ""
    echo "📊 Dataset Summary:"
    echo "─────────────────────"
    echo "📂 Directory: ${LOCAL_PATH}"
    echo "💾 Total Size: $(du -sh "${LOCAL_PATH}" | cut -f1)"
    
    # Count different file types
    arrow_files=$(find "${LOCAL_PATH}" -name "*.arrow" | wc -l)
    json_files=$(find "${LOCAL_PATH}" -name "*.json" | wc -l)
    
    echo "🗂️  Arrow files: ${arrow_files}"
    echo "📄 JSON files: ${json_files}"
    echo "📝 Total files: $(find "${LOCAL_PATH}" -type f | wc -l)"
    echo ""
    echo "🎯 Dataset ready for training!"
else
    echo "❌ Dataset download failed!"
    exit 1
fi 