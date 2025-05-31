FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 기본 패키지 설치
RUN apt-get update && \
    apt-get install -y git wget curl unzip tmux vim build-essential python3 python3-pip python-is-python3 && \
    apt-get clean

# Python 패키지 설치
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --upgrade pip && pip install -r requirements.txt

# accelerate config 자동 적용 (선택)
COPY accelerate_config.yaml /root/.cache/huggingface/accelerate/default_config.yaml

# 포트 열기 및 작업 디렉토리 설정
EXPOSE 6006
WORKDIR /app

# 학습 스크립트 사전 복사
COPY train_mamba.py /app/train_mamba.py

ENTRYPOINT ["python", "train_mamba.py"] 