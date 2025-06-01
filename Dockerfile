# ========= Build-time variables =========
ARG CUDA_VERSION=12.1.1
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04
ARG FORCE_CUDA=1
ARG TORCH_CUDA_ARCH_LIST=7.0;8.0;8.6

# -------- Base image (nvcc 포함) --------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# ========= 환경 변수 =========
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

# ========= 시스템 패키지 =========
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget curl unzip tmux vim build-essential \
        python3 python3-pip python-is-python3 \
        libgl1 libgfortran5 ninja-build && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ========= 파이썬 의존성 =========
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# 1) 빌드 필수 툴 & 선행 의존성(Numpy/packaging) 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging numpy \
    && pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# 2) 나머지 requirements 설치
RUN pip install --no-build-isolation --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    git+https://github.com/state-spaces/mamba.git

# ========= Accelerate 기본 설정 =========
RUN mkdir -p "$HF_HOME/accelerate"
COPY accelerate_config.yaml "$HF_HOME/accelerate/default_config.yaml"

# ========= 학습 코드 복사 =========
COPY train_mamba.py ./train_mamba.py
# (필요하면) COPY configs ./configs

# ========= 포트 및 엔트리포인트 =========
EXPOSE 6006
ENTRYPOINT ["python", "train_mamba.py"]
