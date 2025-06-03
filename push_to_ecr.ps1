# AWS ECR Push Script for YunMin-Mamba (PowerShell)
# 실행 정책 변경이 필요할 수 있습니다: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Configuration
$AWS_REGION = "us-east-1"
$ECR_REPO = "869935091548.dkr.ecr.us-east-1.amazonaws.com/yunmin-mamba"
$IMAGE_TAG = "latest"

Write-Host "🔧 Building Docker image..." -ForegroundColor Yellow
docker build -t "yunmin-mamba:$IMAGE_TAG" .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "🔑 Logging in to ECR..." -ForegroundColor Yellow
$loginToken = aws ecr get-login-password --region $AWS_REGION
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to get ECR login token!" -ForegroundColor Red
    exit 1
}

echo $loginToken | docker login --username AWS --password-stdin $ECR_REPO

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ECR login failed!" -ForegroundColor Red
    exit 1
}

Write-Host "🏷️  Tagging image for ECR..." -ForegroundColor Yellow
docker tag "yunmin-mamba:$IMAGE_TAG" "$ECR_REPO`:$IMAGE_TAG"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker tag failed!" -ForegroundColor Red
    exit 1
}

Write-Host "⬆️  Pushing image to ECR..." -ForegroundColor Yellow
docker push "$ECR_REPO`:$IMAGE_TAG"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker push failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Successfully pushed to ECR: $ECR_REPO`:$IMAGE_TAG" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Use this image URI in your SageMaker training job:" -ForegroundColor Cyan
Write-Host "$ECR_REPO`:$IMAGE_TAG" -ForegroundColor White 