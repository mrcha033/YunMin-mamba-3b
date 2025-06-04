#!/usr/bin/env python3
"""
SageMaker Training Job Launcher for YunMin-Mamba
This script creates and starts a SageMaker training job using the custom Docker image.
"""

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from datetime import datetime

def get_account_id():
    """Get current AWS account ID"""
    sts = boto3.client('sts')
    return sts.get_caller_identity()['Account']

def create_ecr_repository(repository_name, region):
    """Create ECR repository if it doesn't exist"""
    ecr_client = boto3.client('ecr', region_name=region)
    
    try:
        # Check if repository exists
        ecr_client.describe_repositories(repositoryNames=[repository_name])
        print(f"✅ ECR repository '{repository_name}' already exists")
        return True
    except ecr_client.exceptions.RepositoryNotFoundException:
        # Create repository
        try:
            response = ecr_client.create_repository(repositoryName=repository_name)
            print(f"✅ Created ECR repository: {repository_name}")
            print(f"Repository URI: {response['repository']['repositoryUri']}")
            return True
        except Exception as e:
            print(f"❌ Failed to create ECR repository: {e}")
            return False
    except Exception as e:
        print(f"❌ Error checking ECR repository: {e}")
        return False

def create_training_job():
    # Get account info - use us-east-1 to match S3 bucket region
    account_id = get_account_id()
    region = 'us-east-1'  # Changed to match S3 bucket region
    
    # SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Create ECR repository if needed
    repository_name = 'yunmin-mamba-3b'
    if not create_ecr_repository(repository_name, region):
        print("❌ Cannot proceed without ECR repository")
        return None
    
    # IAM role for SageMaker (replace with your actual SageMaker execution role)
    role = f"arn:aws:iam::{account_id}:role/yeongjopt-sagemaker-execution-role"
    
    # Training job configuration
    job_name = f"yunmin-mamba-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # ECR image URI - using us-east-1 region
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}:latest"
    
    # Hyperparameters
    hyperparameters = {
        'learning_rate': '5e-5',
        'batch_size': '1',          # Minimum batch size for memory efficiency
        'num_workers': '0',         # No parallel workers to save memory
        'save_steps': '1000',
        'max_seq_length': '2048',   # Increased but manageable sequence length
    }
    
    # Data paths - Using TrainingInput for proper channel mapping
    train_data_s3 = "s3://yeongjopt-us-east1-bucket/dataset/tagged/"
    output_path = "s3://yeongjopt-us-east1-bucket/yunmin-mamba-outputs/"
    
    # Configure training input channel
    train_input = TrainingInput(
        s3_data=train_data_s3,
        input_mode='File',
        s3_data_type='S3Prefix',
        distribution='FullyReplicated'
    )
    
    # Create estimator
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type='ml.p4d.24xlarge',  # High-performance GPU instance
        # instance_type='ml.g4dn.xlarge',  # Alternative for smaller training
        volume_size=200,                   # Increased EBS volume size
        max_run=28*24*3600,                  # Max training time (28 days)
        hyperparameters=hyperparameters,
        environment={
            'SAGEMAKER_PROGRAM': 'train_mamba.py',
            'SAGEMAKER_SUBMIT_DIRECTORY': '/app',
        },
        output_path=output_path,
        sagemaker_session=sagemaker_session,
        base_job_name='yunmin-mamba-training'
    )
    
    # Start training
    print(f"🚀 Starting training job: {job_name}")
    print(f"🖼️  Image URI: {image_uri}")
    print(f"🌍 Region: {region}")
    print(f"📊 Instance type: ml.p4d.24xlarge")
    print(f"🎛️  Hyperparameters: {hyperparameters}")
    print(f"📁 Training data: {train_data_s3}")
    print(f"📤 Output path: {output_path}")
    
    try:
        estimator.fit({
            'training': train_input
        }, job_name=job_name)
        
        print("✅ Training job submitted successfully!")
        return estimator
    except Exception as e:
        print(f"❌ Failed to start training job: {e}")
        return None

def monitor_training_job(job_name):
    """Monitor training job logs and status"""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        print(f"📈 Training job status: {status}")
        
        if status == 'InProgress':
            print("🔄 Training in progress...")
            print("📋 Check CloudWatch logs for detailed progress")
        elif status == 'Completed':
            print("✅ Training completed successfully!")
            print(f"📤 Model artifacts: {response.get('ModelArtifacts', {}).get('S3ModelArtifacts', 'N/A')}")
        elif status == 'Failed':
            print("❌ Training failed!")
            print(f"💥 Failure reason: {response.get('FailureReason', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Error monitoring job: {e}")

if __name__ == "__main__":
    print("🎯 YunMin-Mamba SageMaker Training Job Launcher")
    print("=" * 50)
    
    # Uncomment to create a new training job
    estimator = create_training_job()
    
    # Uncomment to monitor an existing job
    # job_name = "yunmin-mamba-training-20241202-123456"  # Replace with actual job name
    # monitor_training_job(job_name)
    
    print("\n📋 Next steps:")
    print("1. Monitor the training job in SageMaker console")
    print("2. Check CloudWatch logs for detailed progress")
    print("3. Model artifacts will be saved to S3 output path")
    print("4. Use the trained model for inference") 