# YunMin-Mamba 3B EC2 Pretraining Plan

## ✅ Phase 0: EC2 환경 설치 & 그룹 구성

### Task 0.1 — EC2 인스턴스 설정

* **Instance**: `g5.12xlarge` (or better)
* **OS**: Deep Learning AMI (Ubuntu 20.04) or base Ubuntu 22.04
* **Volume**: 500 GB EBS gp3, Expand + Format
* **Security**: SSH + S3 Access Role

### Task 0.2 — CUDA & PyTorch 소환성

* CUDA 12.1 + cuDNN + GCC 11 개방
* Conda env: `mamba-env`

```bash
conda create -n mamba-env python=3.10 -y
conda activate mamba-env
```

### Task 0.3 — PyTorch + mamba-ssm

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
            transformers>=4.39.3 tokenizers==0.15.2 \
            huggingface-hub datasets sentencepiece bitsandbytes \
            mamba-ssm==2.2.4 causal-conv1d==1.5.0.post8
```

## 📚 Phase 1: 형식 구조 & 파일 구성

### Task 1.1 — 목록 구조

```
/home/ec2-user/
|│
├── train_mamba.py                 # main trainer script
├── mamba_config.json            # Mamba 3B 구성 정의
├── tokenizer/
│   └── YunMin-tokenizer-96k/   # from HuggingFace
├── dataset/
│   └── tagged/            # Arrow format from s3
├── logs/
    └── train.log
```

### Task 1.2 — 형식 설정 `mamba_config.json`

* hidden\_dim: 3072
* intermediate\_size: 8192
* num\_hidden\_layers: 30
* vocab\_size: 96000

## 🚀 Phase 2: Dataset & Tokenizer Integration

### Task 2.1 — 데이터셋 S3 -> EC2 복사

```bash
aws s3 cp s3://yeongjopt-ai-bucket/dataset/tagged/ dataset/tagged/ --recursive
```

### Task 2.2 — 토큰아이저 로드

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrcha033/YunMin-tokenizer-96k")
```

### Task 2.3 — Dataset Load

```python
from datasets import load_dataset

train_data = load_dataset("arrow", data_dir="dataset/clean-corpus")
```

## 🎓 Phase 3: Trainer Setup

### Task 3.1 — Data Collator

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
```

### Task 3.2 — TrainingArguments

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=1000,
    logging_steps=50,
    bf16=True,
    report_to="none",
)
```

### Task 3.3 — Mamba Model Init

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_config("mamba_config.json")
```

### Task 3.4 — Trainer Run

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

## 🌟 Phase 4: 결과 검사 & Checkpoint

* `checkpoints/` 파일을 S3가지 저장

```bash
aws s3 cp checkpoints/ s3://yeongjopt-ai-bucket/checkpoints/ --recursive
```

* `train.log` 종결시점 loss 관찰

## 📊 목표 지표 (PoC 기준)

* 토큰 PPL < 4.0
* 모델 오류 없음 (CUDA, init, trainer)
* 전체 corpus 제거로 1 epoch 수행
