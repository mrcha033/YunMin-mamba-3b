# YunMin-Mamba 3B - SageMaker Training

이 프로젝트는 Amazon SageMaker에서 YunMin-Mamba 3B 모델을 훈련하기 위한 Docker 이미지와 스크립트를 제공합니다.

## 🏗️ Architecture

- **Model**: Mamba-based Language Model (2.78B parameters)
- **Tokenizer**: YunMin-tokenizer-96k (96,000 vocab size)
- **Training**: Multi-category curriculum learning
- **Platform**: Amazon SageMaker Training Jobs
- For an English summary of this setup, see [architecture.md](architecture.md).

## 📂 Project Structure

```
YunMin-mamba-training/
├── docker/
│   └── Dockerfile
├── src/
│   └── train_mamba.py
├── configs/
│   ├── accelerate_config.yaml
│   ├── deepspeed_config.json
│   ├── mamba_3b_config.json
│   └── mamba_7b_config.json
├── sagemaker/
│   ├── sagemaker_training_job.py
│   └── sagemaker_spot_training_job.py
├── example.env
├── tests/
│   └── test_imports.py
└── .github/workflows/
    └── python-tests.yml
```

프로젝트에 포함된 `example.env` 파일을 `.env` 로 복사한 뒤
`TRAIN_DATA_S3`, `OUTPUT_PATH_S3`, `CHECKPOINT_S3`, `LEARNING_RATE` 등
필요한 값을 수정하세요.

`MODEL_CONFIG_PATH` 환경 변수를 `configs/mamba_3b_config.json` 또는
`configs/mamba_7b_config.json`으로 설정하면 3B와 7B 중 원하는 모델을 선택할 수
있습니다.

## 🚀 Quick Start

### 1. ECR에 이미지 푸시

```bash
# AWS CLI 설정 확인
aws sts get-caller-identity

docker build -f docker/Dockerfile -t <your-image> .
docker tag <your-image> <ECR_URI>
docker push <ECR_URI>
```

Dockerfile 내부에서 `MAMBA_SKIP_CUDA_BUILD=1`이 설정되어 있어 CUDA 커널
빌드를 자동으로 건너뜁니다. 따라서 GPU가 없는 환경에서도 이미지를 빌드할
수 있습니다.

### 2. 데이터 준비

S3에 다음 구조로 데이터를 업로드하세요:

```
s3://your-bucket/yunmin-mamba-data/dataset/tagged/
├── main_data/
│   ├── shard_001/
│   ├── shard_002/
│   └── ...
├── korean_textbook/
├── academic_data/
├── papers_data/
├── national_data/
├── national_assembly_data/
├── web_data/
└── social_media_data/
```

### 데이터셋 사용 주의

훈련을 시작하기 전에 사용하려는 데이터셋에 대한 법적 권한을 보유하고 있는지 확인하세요.
데이터에 개인정보가 포함되지 않았는지 점검하고 잠재적인 편향도 고려해야 합니다.
저장소 유지 관리자는 데이터 라이선스에 대해 어떠한 보증도 제공하지 않습니다.

### 3. SageMaker 훈련 작업 시작

```python
# sagemaker_training_job.py 수정 후 실행
python sagemaker/sagemaker_training_job.py
```

## ⚙️ Configuration

### Hyperparameters

SageMaker 훈련 작업에서 다음 하이퍼파라미터를 설정할 수 있습니다:

- `learning_rate`: 학습률 (기본값: 5e-5)
- `batch_size`: 배치 크기 (기본값: 8)
- `num_workers`: 데이터 로더 워커 수 (기본값: 4)
- `save_steps`: 체크포인트 저장 간격 (기본값: 1000)

### Instance Types

권장 인스턴스 타입:

- **대규모 훈련**: `ml.p4d.24xlarge` (8x A100 40GB)
- **중간 규모**: `ml.p3.8xlarge` (4x V100 16GB)
- **소규모 테스트**: `ml.g4dn.xlarge` (1x T4 16GB)

### Model Configuration

`configs/mamba_config.json`:
```json
{
  "vocab_size": 96000,
  "d_model": 2560,
  "num_hidden_layers": 64,
  "model_type": "mamba"
}
```

## 📊 Learning Order

모델은 다음 순서로 데이터셋 카테고리를 학습합니다:

1. **main_data**: 대규모 한국어 말뭉치 - 기본 언어 능력
2. **korean_textbook**: 정형화된 문체와 문법 - 언어 규칙성
3. **academic_data**: 전문 어휘와 문장 - 표현력 확장
4. **papers_data**: 연구 논문 - 고급 어휘와 복잡한 구조
5. **national_data**: 행정 문서 - 공식적 표현과 용어
6. **national_assembly_data**: 국회 발언록 - 구어체와 논리적 표현
7. **web_data**: 웹 텍스트 - 비정형 표현과 최신 용어
8. **social_media_data**: SNS 데이터 - 일상 대화체와 신조어

데이터셋 사용 전에 저작권, 개인정보 보호, 편향 가능성 등을 스스로 검토해야 합니다.
프로젝트 관리자는 이러한 데이터 라이선스나 이용 조건에 대해 어떠한 보증도 제공하지 않습니다.

## 🔧 SageMaker Environment Variables

컨테이너는 다음 SageMaker 환경 변수를 자동으로 인식합니다:

- `SM_CHANNEL_TRAINING`: 훈련 데이터 경로
- `SM_MODEL_DIR`: 모델 저장 경로
- `SM_OUTPUT_DATA_DIR`: 출력 데이터 경로
- `/opt/ml/input/config/hyperparameters.json`: 하이퍼파라미터

`src/train_mamba.py` 를 직접 실행할 때는 `--dataset-path` 옵션으로 `SM_CHANNEL_TRAINING` 값을 덮어쓸 수 있습니다.

## 📁 Output Structure

훈련 완료 후 다음과 같은 구조로 결과가 저장됩니다:

```
/opt/ml/model/
├── final/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files...
├── step-1000-main_data/
├── step-2000-main_data/
└── ...

/opt/ml/output/
├── training.log
└── logs/
    └── train.log
```

## 🐳 Docker Commands

### 로컬 테스트

```bash
# 로컬에서 테스트 실행
docker run --rm \
  -v /path/to/local/data:/opt/ml/input/data/training \
  -v /path/to/output:/opt/ml/model \
  yunmin-mamba:latest

# GPU가 있는 경우
docker run --rm --gpus all \
  -v /path/to/local/data:/opt/ml/input/data/training \
  -v /path/to/output:/opt/ml/model \
  yunmin-mamba:latest
```

### 디버깅

```bash
# 컨테이너 내부 접속
docker run -it --rm yunmin-mamba:latest bash

# 모델 초기화 테스트
docker run --rm yunmin-mamba:latest python -c "
from transformers import AutoConfig, AutoModelForCausalLM
AutoModelForCausalLM.from_config(AutoConfig.from_pretrained('configs/mamba_config.json'))
print('✅ Model can be imported')
"
```

## 📋 Monitoring

### CloudWatch Logs

SageMaker 콘솔에서 다음 로그를 확인할 수 있습니다:

- `/aws/sagemaker/TrainingJobs`: 훈련 로그
- Training job의 CloudWatch 링크에서 실시간 모니터링

### 훈련 진행률

로그에서 다음 메트릭을 확인하세요:

- 각 카테고리별 훈련 진행률
- 50스텝마다 loss 출력
- 1000스텝마다 체크포인트 저장
- GPU 메모리 사용량

## 🛠️ Troubleshooting

### 일반적인 문제

1. **CUDA Out of Memory**
   - `batch_size` 줄이기
   - `num_workers` 줄이기
   - 더 큰 인스턴스 타입 사용

2. **데이터 로딩 실패**
   - S3 권한 확인
   - 데이터 경로 확인
   - 데이터 형식 검증

3. **훈련 속도 느림**
   - GPU 인스턴스 사용 확인
   - `num_workers` 조정
   - 배치 크기 최적화

### 로그 확인

```bash
# SageMaker 훈련 작업 상태 확인
aws sagemaker describe-training-job --training-job-name your-job-name

# CloudWatch 로그 확인
aws logs get-log-events \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name your-job-name/algo-1-timestamp
```

## 📞 Support

문제가 발생하면 다음을 확인하세요:

1. ECR 이미지가 올바르게 푸시되었는지
2. SageMaker IAM 역할 권한
3. S3 데이터 경로와 권한
4. 하이퍼파라미터 설정
5. 인스턴스 타입 제한

---

**ECR Repository**: `869935091548.dkr.ecr.us-east-1.amazonaws.com/yunmin-mamba:latest` 
