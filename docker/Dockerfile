# ========= Build-time variables =========
ARG CUDA_VERSION=12.1.1
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04

# ========= Base image with CUDA =========
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# ========= Environment setup =========
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TOKENIZERS_PARALLELISM=false \
    MAX_JOBS=4 \
    MAMBA_SKIP_CUDA_BUILD=1 \
    PYTHONPATH=/app

# ========= System packages =========
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget curl unzip tmux vim build-essential \
        python3 python3-pip python-is-python3 \
        libgl1 libgfortran5 \
        ninja-build cmake pkg-config && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ========= Python dependencies =========
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# 1) Upgrade pip & essential tools
RUN pip install --upgrade pip setuptools wheel ninja packaging numpy

# 2) Install PyTorch for CUDA 12.1
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 3) Install base deps
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir mamba-ssm==2.2.4 transformers>=4.39.3

# 4) Create necessary directories
RUN mkdir -p /app/logs /app/checkpoints

# 5) Verify installation
RUN python -c "import torch; print('✅ PyTorch imported successfully')"
RUN python - <<'PY'
from transformers import MambaLMHeadModel
print('✅ MambaLMHeadModel imported successfully')
PY
# ========= Accelerate configuration =========
RUN mkdir -p "/root/.cache/huggingface/accelerate"
COPY configs/accelerate_config.yaml /root/.cache/huggingface/accelerate/default_config.yaml
COPY configs/deepspeed_config.json /app/deepspeed_config.json

# ========= Copy configuration files =========
COPY configs/ /app/configs/

# ========= Training script =========
COPY src/train_mamba.py /app/train_mamba.py

# ========= Port and working directory =========
EXPOSE 6006
WORKDIR /app

# ========= CUDA availability test =========
RUN python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available()); print('🔢 CUDA version:', torch.version.cuda)"

# ========= Test MambaLMHeadModel =========
RUN python - <<'PY'
import json
from pathlib import Path
from transformers import MambaLMHeadModel, MambaConfig
cfg = json.load(open(Path('/app/configs/mamba_config.json')))
model = MambaLMHeadModel(MambaConfig(**cfg))
print('✅ MambaLMHeadModel instantiated successfully')
PY

# ========= SageMaker entrypoint =========
ENV SAGEMAKER_PROGRAM=train_mamba.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/app

# ========= Default entrypoint =========
CMD ["python", "train_mamba.py"]
