#!/bin/bash

# AWS ECR Push Script for YunMin-Mamba
set -e

# Configuration
AWS_REGION="us-east-1"
ECR_REPO="869935091548.dkr.ecr.us-east-1.amazonaws.com/yunmin-mamba"
IMAGE_TAG="latest"

echo "🔧 Building Docker image..."
docker build -t yunmin-mamba:${IMAGE_TAG} .

echo "🔑 Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO}

echo "🏷️  Tagging image for ECR..."
docker tag yunmin-mamba:${IMAGE_TAG} ${ECR_REPO}:${IMAGE_TAG}

echo "⬆️  Pushing image to ECR..."
docker push ${ECR_REPO}:${IMAGE_TAG}

echo "✅ Successfully pushed to ECR: ${ECR_REPO}:${IMAGE_TAG}"
echo ""
echo "📋 Use this image URI in your SageMaker training job:"
echo "${ECR_REPO}:${IMAGE_TAG}" 