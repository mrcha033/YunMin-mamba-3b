#!/bin/bash

# YunMin-Mamba S3 Data Download Script
# Downloads training dataset from S3 bucket

set -e  # Exit on any error

echo "🚀 Starting S3 data download for YunMin-Mamba training..."

# Configuration
S3_BUCKET="yeongjopt-ai-bucket"
S3_PATH="dataset/tagged"
LOCAL_PATH="dataset/tagged"

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ Error: AWS CLI is not configured or credentials are invalid"
    echo "Please configure AWS CLI with: aws configure"
    exit 1
fi

# Create local directory
mkdir -p "${LOCAL_PATH}"

# Download dataset
echo "📥 Downloading dataset from s3://${S3_BUCKET}/${S3_PATH}..."
aws s3 cp "s3://${S3_BUCKET}/${S3_PATH}/" "${LOCAL_PATH}/" --recursive

# Verify download
if [ $? -eq 0 ]; then
    echo "✅ Dataset download completed successfully!"
    echo "📊 Dataset info:"
    du -sh "${LOCAL_PATH}"
    find "${LOCAL_PATH}" -name "*.arrow" | wc -l | xargs echo "Arrow files found:"
else
    echo "❌ Dataset download failed!"
    exit 1
fi

echo "🎯 Ready for training!" 