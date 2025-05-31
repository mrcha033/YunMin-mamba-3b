# YunMin-Mamba 3B Pretraining Environment

🚀 **Docker 기반 YunMin-Mamba 3B 사전학습 환경**

완전히 자동화된 Docker 컨테이너에서 YunMin-Mamba 3B 모델을 학습할 수 있는 통합 환경입니다.

## 📦 프로젝트 구조

```
YunMin-mamba-3b/
├── Dockerfile                       # CUDA 12.1 + Python 환경
├── docker-compose.yaml             # 컨테이너 오케스트레이션
├── docker-compose.override.yml     # 개발용 명령어 오버라이드
├── requirements.txt                # Python 패키지 목록
├── train_mamba.py                 # Accelerate 기반 훈련 스크립트
├── download_s3_dataset.sh         # S3 데이터셋 다운로드 스크립트
├── setup.sh                      # 프로젝트 초기 설정 스크립트
├── accelerate_config.yaml         # Accelerate FP16 설정
├── env.example                    # 환경변수 템플릿
├── .env                          # 환경변수 설정 (git에서 제외)
├── configs/
│   └── mamba_config.json         # Mamba 3B 모델 설정
├── checkpoints/                  # 학습된 모델 저장소
├── logs/                        # 훈련 로그
└── dataset/                     # 데이터셋 저장소
```

## 🛠️ 빠른 시작

### 1️⃣ 자동 설정

```bash
# 프로젝트 클론
git clone <your-repo> && cd YunMin-mamba-3b

# 자동 설정 실행
chmod +x setup.sh && ./setup.sh
```

### 2️⃣ 환경 설정

```bash
# .env 파일 편집 (S3 버킷 정보 등)
nano .env

# AWS 자격 증명 설정
aws configure
```

### 3️⃣ 학습 시작

```bash
# 한 번에 시작
docker-compose up --build yunmin-mamba-train
```

## 🚀 상세 사용법

### 옵션 1: Docker Compose (권장)

```bash
# 1. 환경 시작
docker-compose up -d yunmin-mamba-train

# 2. 컨테이너 접속
docker-compose exec yunmin-mamba-train bash

# 3. 데이터셋 다운로드
./download_s3_dataset.sh

# 4. 훈련 시작
python train_mamba.py
```

### 옵션 2: 데이터 다운로드만 별도 실행

```bash
# 데이터만 다운로드
docker-compose --profile data run --rm data-downloader
```

### 옵션 3: 환경변수 커스터마이징

```bash
# .env 파일 수정 후
S3_BUCKET=my-custom-bucket docker-compose up yunmin-mamba-train
```

## ⚙️ 환경 설정

### 환경변수 (.env 파일)

```bash
# S3 Dataset Configuration
S3_BUCKET=yeongjopt-ai-bucket
S3_PATH=dataset/tagged
LOCAL_DATASET_PATH=dataset/tagged

# Training Configuration
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=5e-5
NUM_EPOCHS=1

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
```

### S3 데이터셋 경로 변경

```bash
# 방법 1: 환경변수로
export S3_BUCKET=my-bucket
./download_s3_dataset.sh

# 방법 2: 명령행 인자로
./download_s3_dataset.sh my-bucket my-dataset-path dataset/custom
```

## 📊 모니터링

### 훈련 진행 확인
```bash
# 로그 실시간 확인
docker-compose exec yunmin-mamba-train tail -f logs/train.log

# 체크포인트 확인
docker-compose exec yunmin-mamba-train ls -la checkpoints/
```

### TensorBoard
```bash
# 컨테이너 내부에서
tensorboard --logdir=/app/logs --host=0.0.0.0 --port=6006

# 브라우저에서 접속: http://localhost:6006
```

## 🔧 문제 해결

### GPU 인식 문제
```bash
# NVIDIA 런타임 확인
docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### AWS 자격 증명 문제
```bash
# 자격 증명 확인
aws sts get-caller-identity

# 또는 컨테이너 내부에서
docker-compose exec yunmin-mamba-train aws sts get-caller-identity
```

### 환경변수 확인
```bash
# .env 파일 로드 확인
docker-compose exec yunmin-mamba-train env | grep S3_
```

## 📈 성능 최적화

- **Multi-GPU**: `accelerate_config.yaml`에서 분산 설정
- **배치 크기**: `.env`의 `BATCH_SIZE` 조정
- **Mixed Precision**: FP16 기본 활성화

## 🎯 목표 지표

- ✅ Perplexity < 4.0
- ✅ CUDA 오류 없음
- ✅ 1 Epoch 완주

## 📋 최종 체크리스트

| 항목 | 확인 |
|------|------|
| ✅ `setup.sh` 실행 완료 | □ |
| ✅ `.env` 파일 설정 완료 | □ |
| ✅ `configs/mamba_config.json` 존재 | □ |
| ✅ AWS 자격 증명 설정 | □ |
| ✅ NVIDIA Docker 런타임 동작 | □ |
| ✅ S3 데이터셋 접근 가능 | □ |

---

**Ready to train! 🚀**

문의사항이나 이슈가 있으시면 GitHub Issues에 등록해주세요! 