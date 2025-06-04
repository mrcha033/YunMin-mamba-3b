# ECR Docker 이미지 빌드 및 푸시 스크립트

# 에러 시 중단
$ErrorActionPreference = "Stop"

# Configuration - us-east-1 리전으로 통일
$REGION = "us-east-1"
$REPOSITORY_NAME = "yunmin-mamba-3b"
$IMAGE_TAG = "latest"

Write-Host "🚀 Building and pushing Docker image to ECR..." -ForegroundColor Green

# Step 1: Get Account ID
try {
    $ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)
    $IMAGE_URI = "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME`:$IMAGE_TAG"

    Write-Host "Account ID: $ACCOUNT_ID" -ForegroundColor Cyan
    Write-Host "Region: $REGION" -ForegroundColor Cyan
    Write-Host "Repository: $REPOSITORY_NAME" -ForegroundColor Cyan
    Write-Host "Image URI: $IMAGE_URI" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Failed to get AWS account ID. Make sure AWS CLI is configured." -ForegroundColor Red
    exit 1
}

# Step 2: Create ECR repository if it doesn't exist
Write-Host "📦 Creating ECR repository if it doesn't exist..." -ForegroundColor Yellow
try {
    aws ecr describe-repositories --repository-names $REPOSITORY_NAME --region $REGION | Out-Null
    Write-Host "✅ Repository $REPOSITORY_NAME already exists" -ForegroundColor Green
} catch {
    Write-Host "Creating new repository: $REPOSITORY_NAME" -ForegroundColor Yellow
    aws ecr create-repository --repository-name $REPOSITORY_NAME --region $REGION
    Write-Host "✅ Repository created successfully" -ForegroundColor Green
}

# Step 3: Get ECR login token and login to Docker
Write-Host "🔐 Logging into ECR..." -ForegroundColor Yellow
try {
    $loginCommand = aws ecr get-login-password --region $REGION
    $loginCommand | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
    Write-Host "✅ Successfully logged into ECR" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to login to ECR" -ForegroundColor Red
    exit 1
}

# Step 4: Build Docker image
Write-Host "🏗️ Building Docker image..." -ForegroundColor Yellow
try {
    docker build -t "$REPOSITORY_NAME`:$IMAGE_TAG" .
    Write-Host "✅ Docker image built successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to build Docker image" -ForegroundColor Red
    exit 1
}

# Step 5: Tag image for ECR
Write-Host "🏷️ Tagging image for ECR..." -ForegroundColor Yellow
try {
    docker tag "$REPOSITORY_NAME`:$IMAGE_TAG" $IMAGE_URI
    Write-Host "✅ Image tagged successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to tag image" -ForegroundColor Red
    exit 1
}

# Step 6: Push image to ECR
Write-Host "⬆️ Pushing image to ECR..." -ForegroundColor Yellow
try {
    docker push $IMAGE_URI
    Write-Host "✅ ECR deployment completed successfully!" -ForegroundColor Green
    Write-Host "Image URI: $IMAGE_URI" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Failed to push image to ECR" -ForegroundColor Red
    exit 1
}

Write-Host "🎉 Docker image is ready! You can now run the SageMaker training job." -ForegroundColor Green
Write-Host "💡 Next step: python sagemaker\sagemaker_training_job.py" -ForegroundColor Yellow
