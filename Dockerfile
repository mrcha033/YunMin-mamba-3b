FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/root/.cache/huggingface

# 기본 패키지 설치
RUN apt-get update && \
    apt-get install -y git wget curl unzip tmux vim build-essential \
    python3 python3-pip python-is-python3 \
    libbitsandbytes-dev libnvToolsExt1 libgl1 && \
    apt-get clean

# Python 패키지 설치
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --upgrade pip && pip install -r requirements.txt

# accelerate 설정 복사
COPY accelerate_config.yaml $HF_HOME/accelerate/default_config.yaml

# 포트 및 작업 디렉토리
EXPOSE 6006
WORKDIR /app

# 학습 스크립트 복사
COPY train_mamba.py /app/train_mamba.py

# 실행
ENTRYPOINT ["python", "train_mamba.py"]
