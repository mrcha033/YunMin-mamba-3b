#!/usr/bin/env python3
"""
SageMaker Spot Training Job Launcher for YunMin-Mamba
This script creates and starts a SageMaker training job using Spot instances with checkpoint resumption.
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

def create_spot_training_job():
    # Get account info - use us-east-1 to match S3 bucket region
    account_id = get_account_id()
    region = 'us-east-1'
    
    # SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Create ECR repository if needed
    repository_name = 'yunmin-mamba-3b'
    if not create_ecr_repository(repository_name, region):
        print("❌ Cannot proceed without ECR repository")
        return None
    
    # IAM role for SageMaker
    role = f"arn:aws:iam::{account_id}:role/yeongjopt-sagemaker-execution-role"
    
    # Training job configuration
    job_name = f"yunmin-mamba-spot-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # ECR image URI - using us-east-1 region
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}:latest"
    
    # Hyperparameters for spot training (optimized for interruption handling)
    hyperparameters = {
        'learning_rate': '5e-5',
        'batch_size': '1',          # Keep minimal for memory efficiency
        'num_workers': '0',         # No parallel workers to save memory
        'save_steps': '500',        # More frequent saves for spot instances
        'max_seq_length': '2048',   # Manageable sequence length
    }
    
    # Data paths
    train_data_s3 = "s3://yeongjopt-us-east1-bucket/dataset/tagged/"
    output_path = "s3://yeongjopt-us-east1-bucket/yunmin-mamba-outputs/"
    checkpoint_s3 = "s3://yeongjopt-us-east1-bucket/yunmin-mamba-checkpoints/"
    
    # Configure training input channel
    train_input = TrainingInput(
        s3_data=train_data_s3,
        input_mode='File',
        s3_data_type='S3Prefix',
        distribution='FullyReplicated'
    )
    
    # Create estimator with Spot instances
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type='ml.p4d.24xlarge',  # High-performance GPU instance
        volume_size=200,
        max_run=432000,   # Max training time (5 days)
        max_wait=432000, # Max wait time including spot delays (5 days)
        
        # Spot instance configuration
        use_spot_instances=True,
        max_retry_attempts=5,  # Retry on spot interruption
        
        # Checkpoint configuration
        checkpoint_s3_uri=checkpoint_s3,
        checkpoint_local_path='/opt/ml/checkpoints',
        
        hyperparameters=hyperparameters,
        environment={
            'SAGEMAKER_PROGRAM': 'train_mamba.py',
            'SAGEMAKER_SUBMIT_DIRECTORY': '/app',
        },
        output_path=output_path,
        sagemaker_session=sagemaker_session,
        base_job_name='yunmin-mamba-spot-training'
    )
    
    # Start training
    print(f"🚀 Starting Spot training job: {job_name}")
    print(f"🖼️  Image URI: {image_uri}")
    print(f"🌍 Region: {region}")
    print(f"💰 Using Spot instances (70-90% cost savings)")
    print(f"📊 Instance type: ml.p4d.24xlarge")
    print(f"🔄 Auto-resume from latest checkpoint")
    print(f"💾 Checkpoints: {checkpoint_s3}")
    print(f"🎛️  Hyperparameters: {hyperparameters}")
    print(f"📁 Training data: {train_data_s3}")
    print(f"📤 Output path: {output_path}")
    
    try:
        estimator.fit({
            'training': train_input
        }, job_name=job_name)
        
        print("✅ Spot training job submitted successfully!")
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
        secondary_status = response.get('SecondaryStatus', 'Unknown')
        
        print(f"📈 Training job status: {status}")
        print(f"📊 Secondary status: {secondary_status}")
        
        if 'BillableTimeInSeconds' in response:
            billable_time = response['BillableTimeInSeconds']
            print(f"💰 Billable time: {billable_time} seconds ({billable_time/3600:.2f} hours)")
        
        if status == 'InProgress':
            print("🔄 Training in progress...")
            print("📋 Check CloudWatch logs for detailed progress")
        elif status == 'Completed':
            print("✅ Training completed successfully!")
            print(f"📤 Model artifacts: {response.get('ModelArtifacts', {}).get('S3ModelArtifacts', 'N/A')}")
        elif status == 'Failed':
            print("❌ Training failed!")
            print(f"💥 Failure reason: {response.get('FailureReason', 'Unknown')}")
        elif status == 'Stopped':
            print("⏹️ Training was stopped")
            
    except Exception as e:
        print(f"❌ Error monitoring job: {e}")

if __name__ == "__main__":
    print("🎯 YunMin-Mamba Spot Training Job Launcher")
    print("💰 Using Spot instances for cost optimization")
    print("🔄 Resuming from step 3000 checkpoint")
    print("=" * 60)
    
    # Create and start spot training job
    estimator = create_spot_training_job()
    
    if estimator:
        print("\n📋 Next steps:")
        print("1. Monitor the training job in SageMaker console")
        print("2. Check CloudWatch logs for detailed progress")
        print("3. Checkpoints will be saved every 500 steps")
        print("4. Training will auto-resume if spot instance is interrupted")
        print("5. Model artifacts will be saved to S3 output path")
        print(f"6. Expected cost savings: 70-90% vs on-demand instances")
    else:
        print("❌ Failed to create spot training job") 
