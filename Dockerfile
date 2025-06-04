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

# 4) Copy simple mamba implementation
COPY mamba_simple.py /app/

# 5) Create necessary directories
RUN mkdir -p /app/logs /app/checkpoints /app/configs

# 6) Verify installation
RUN python -c "import torch; print('✅ PyTorch imported successfully')"
RUN python -c "import sys; sys.path.append('/app'); from mamba_simple import Mamba; print('✅ Mamba imported successfully')"

# ========= Accelerate configuration =========
RUN mkdir -p "/root/.cache/huggingface/accelerate"
COPY accelerate_config.yaml /root/.cache/huggingface/accelerate/default_config.yaml
COPY deepspeed_config.json /app/deepspeed_config.json

# ========= Create mamba config =========
RUN echo '{\n  "vocab_size": 96000,\n  "d_model": 2560,\n  "num_hidden_layers": 64,\n  "model_type": "mamba"\n}' > /app/configs/mamba_config.json

# ========= Training script =========
COPY train_mamba.py /app/train_mamba.py

# ========= Create SageMaker train script =========
RUN echo '#!/bin/bash\ncd /app\nexec python train_mamba.py "$@"' > /usr/local/bin/train && \
    chmod +x /usr/local/bin/train

# ========= Port and working directory =========
EXPOSE 6006
WORKDIR /app

# ========= CUDA availability test =========
RUN python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available()); print('🔢 CUDA version:', torch.version.cuda)"

# ========= Test SimpleMambaLM =========
RUN python -c "import sys; sys.path.append('/app'); from train_mamba import SimpleMambaLM; print('✅ SimpleMambaLM imported successfully')"

# ========= SageMaker entrypoint =========
ENV SAGEMAKER_PROGRAM=train_mamba.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/app

# ========= Default entrypoint =========
CMD ["python", "train_mamba.py"]
