# YunMin-Mamba 3B Pretraining Environment

This repository contains the Docker environment and helper scripts used to train the **YunMin-Mamba 3B** language model on Amazon SageMaker.

YunMin‑Mamba 3B is an open 2.8B parameter language model based on the [Mamba architecture](https://github.com/state-spaces/mamba).  Training is orchestrated with HuggingFace Accelerate and DeepSpeed, and the dataset is organised into multiple categories that are processed in a curriculum order.  Most of the step‑by‑step instructions are written in Korean inside [README_SAGEMAKER.md](README_SAGEMAKER.md).

## Project Structure

```
YunMin-mamba-training/
├── docker/
│   └── Dockerfile                  # Base image for SageMaker training
├── requirements.txt                # Python dependencies
├── configs/                        # Model & training configs
│   ├── accelerate_config.yaml      # HuggingFace Accelerate configuration
│   ├── deepspeed_config.json       # DeepSpeed configuration
│   ├── mamba_config.json           # 3B model configuration
│   └── mamba_7b_config.json        # 7B model configuration
├── src/
│   └── train_mamba.py              # Main training script executed in SageMaker
├── sagemaker/
│   ├── sagemaker_training_job.py   # Launch standard SageMaker training
│   └── sagemaker_spot_training_job.py  # Launch Spot training job
├── tests/
│   └── test_imports.py             # Simple import test
├── .github/workflows/
│   └── python-tests.yml            # CI workflow
└── README_SAGEMAKER.md             # Detailed SageMaker instructions
```

Set the `MODEL_CONFIG_PATH` environment variable to point to either
`configs/mamba_3b_config.json` or `configs/mamba_7b_config.json` to choose which model size
to train.

## Model Architecture

The configuration file `configs/mamba_3b_config.json` defines the 3B model with 36 layers and a hidden size of 2,560.  A larger 7B variant is provided in `configs/mamba_7b_config.json` with 32 layers and a hidden size of 4,096.  Gradient checkpointing and DeepSpeed ZeRO Stage 2 are enabled during training to keep GPU memory usage manageable. 

## Dataset Layout

Training data is expected under an S3 bucket following the structure below.  Each category is processed in the listed order so the model gradually learns from formal text to conversational language.

```
s3://your-bucket/yunmin-mamba-data/dataset/tagged/
├── main_data/
├── korean_textbook/
├── academic_data/
├── papers_data/
├── national_data/
├── national_assembly_data/
├── web_data/
└── social_media_data/
```

## Dataset Usage and Licensing

You are responsible for ensuring that you have the right to use any dataset
referenced by this project. Consider privacy, bias and other ethical issues
before training on third-party data. The maintainers provide **no warranty**
regarding the licenses or terms that may apply to these datasets.

For a full walkthrough in Korean, refer to [README_SAGEMAKER.md](README_SAGEMAKER.md).

## Dataset Usage Notice

Make sure you have the legal right to access and use any dataset before starting
training. Review the license terms for each dataset and ensure your use complies
with applicable laws and regulations. Consider privacy implications and assess
your data for potential biases.

The maintainers provide this project without warranty of any kind and do not
guarantee that any dataset you choose is properly licensed for your purposes.

## Configuration

Copy `example.env` to `.env` and update the values for your setup. The template
includes dataset paths, S3 output locations and common hyperparameters such as
`LEARNING_RATE`, `BATCH_SIZE`, `MAX_SEQ_LENGTH` and the `MODEL_CONFIG_PATH`.

## Running a Training Job

1. **Push the Docker image to ECR**:

   ```bash
   aws sts get-caller-identity
   docker build -f docker/Dockerfile -t <your-image> .
   docker tag <your-image> <ECR_URI>
   docker push <ECR_URI>
   ```

   Ensure that the resulting ECR URI is reflected in `sagemaker/sagemaker_spot_training_job.py` or `sagemaker/sagemaker_training_job.py`.

2. **Start the job on SageMaker**:

   ```bash
   python sagemaker/sagemaker_spot_training_job.py
   ```

   The script creates a training job using Spot instances and resumes from the latest checkpoint when interrupted.

For additional configuration options and dataset layout, see [README_SAGEMAKER.md](README_SAGEMAKER.md).

## Contributing

If you want to run the test suite or make changes to this project, see
[CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. The guide explains how to
install optional dependencies like `transformers` and `mamba-ssm` before
running `pytest`.

## Installation

Install the Python packages from `requirements.txt`. If `nvcc` is not available
on your system, set `MAMBA_SKIP_CUDA_BUILD=1` so the GPU kernels are skipped:

```bash
MAMBA_SKIP_CUDA_BUILD=1 pip install -r requirements.txt
```

## Running Tests

Install the dependencies used by the unit tests:

```bash
MAMBA_SKIP_CUDA_BUILD=1 pip install -r requirements.txt
pip install pytest
```

This installs packages like `transformers`, `mamba-ssm` and `dockerfile-parse`.
Then run the test suite with `pytest`:

```bash
pytest
```

## Documentation

Additional usage notes and a detailed walkthrough of the SageMaker workflow are available in:

- [README_SAGEMAKER.md](README_SAGEMAKER.md) – Korean quick start and troubleshooting guide

## License

This project is licensed under the [MIT License](LICENSE).
