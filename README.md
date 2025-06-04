# YunMin-Mamba 3B Pretraining Environment

This repository provides the Docker image and helper scripts used to train the **YunMin-Mamba 3B** language model on Amazon SageMaker.

## Project Structure

```
YunMin-mamba-3b/
├── Dockerfile                # Base image for SageMaker training
├── build_and_push_ecr.ps1    # Example script to push the image to ECR
├── requirements.txt          # Python dependencies
├── train_mamba.py            # Main training script executed in SageMaker
├── accelerate_config.yaml    # HuggingFace Accelerate configuration
├── mamba_config.json         # Model configuration
├── deepspeed_config.json     # Deepspeed configuration
├── sagemaker_training_job.py # Launch standard SageMaker training
├── sagemaker_spot_training_job.py # Launch Spot training job
├── README_SAGEMAKER.md       # Detailed SageMaker instructions
└── architecture.md
```

## Running a Training Job

1. **Push the Docker image to ECR** (example for Windows PowerShell):

   ```powershell
   aws sts get-caller-identity
   ./build_and_push_ecr.ps1
   ```

   Ensure that the resulting ECR URI is reflected in `sagemaker_spot_training_job.py` or `sagemaker_training_job.py`.

2. **Start the job on SageMaker**:

   ```bash
   python sagemaker_spot_training_job.py
   ```

   The script creates a training job using Spot instances and resumes from the latest checkpoint when interrupted.

For additional configuration options and dataset layout, see [README_SAGEMAKER.md](README_SAGEMAKER.md).

## License

This project is licensed under the [MIT License](LICENSE).
