#!/usr/bin/env python3
"""
YunMin-Mamba 3B Pretraining Script with Accelerate
Supports multiple dataset categories with learning order
"""
import subprocess, os, sys

def ensure_libgfortran():
    try:
        # 이미 있으면 아무 일도 안 일어남
        subprocess.check_call("ldconfig -p | grep libgfortran", shell=True)
        print("✅ libgfortran already present")
    except subprocess.CalledProcessError:
        print("⚠️  libgfortran not found — installing…")
        subprocess.check_call(["apt-get", "update"])
        subprocess.check_call(["apt-get", "install", "-y", "libgfortran5"])
        # optional: 심볼릭 링크 보강
        subprocess.run("ln -sf $(ldconfig -p | grep libgfortran.so.5 | head -1 | awk '{print $4}') "
                       "/usr/lib/x86_64-linux-gnu/libgfortran.so", shell=True, check=False)

ensure_libgfortran()

import math
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoConfig, get_scheduler
from torch.utils.data import DataLoader
from mamba_ssm.models.mamba_lm import MambaLMHeadModel
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("yunmin-mamba-train")

# Learning order configuration
LEARNING_ORDER = [
    "main_data",              # 대규모 한국어 말뭉치 - 기본 언어 능력
    "korean_textbook",        # 정형화된 문체와 문법 - 언어 규칙성
    "academic_data",          # 전문 어휘와 문장 - 표현력 확장
    "papers_data",            # 연구 논문 - 고급 어휘와 복잡한 구조
    "national_data",          # 행정 문서 - 공식적 표현과 용어
    "national_assembly_data", # 국회 발언록 - 구어체와 논리적 표현
    "web_data",              # 웹 텍스트 - 비정형 표현과 최신 용어
    "social_media_data"      # SNS 데이터 - 일상 대화체와 신조어
]

def load_dataset_category(dataset_path, category):
    """Load a specific dataset category with all its shards"""
    category_path = Path(dataset_path) / category
    
    if not category_path.exists():
        logger.warning(f"Dataset category {category} not found at {category_path}")
        return None
    
    logger.info(f"Loading dataset category: {category}")
    
    # Find all shard directories
    shard_dirs = [d for d in category_path.iterdir() if d.is_dir() and d.name.startswith('shard_')]
    
    if not shard_dirs:
        logger.warning(f"No shard directories found in {category_path}")
        return None
    
    shard_dirs.sort()  # Ensure consistent order
    logger.info(f"Found {len(shard_dirs)} shards in {category}")
    
    # Load all shards and concatenate
    datasets = []
    for shard_dir in shard_dirs:
        try:
            shard_dataset = load_from_disk(str(shard_dir))
            datasets.append(shard_dataset)
            logger.info(f"Loaded shard: {shard_dir.name}")
        except Exception as e:
            logger.error(f"Failed to load shard {shard_dir}: {e}")
            continue
    
    if not datasets:
        logger.error(f"No valid shards found in {category}")
        return None
    
    # Concatenate all shards
    combined_dataset = concatenate_datasets(datasets)
    logger.info(f"Combined {len(datasets)} shards into dataset with {len(combined_dataset)} examples")
    
    return combined_dataset

def train_on_category(accelerator, model, tokenizer, category_dataset, category_name, 
                     optimizer, lr_scheduler, data_collator, output_dir, step_offset=0):
    """Train on a specific dataset category"""
    logger.info(f"🚀 Starting training on {category_name}...")
    
    # Create DataLoader for this category
    train_dataloader = DataLoader(
        category_dataset, 
        shuffle=True, 
        batch_size=4, 
        collate_fn=data_collator, 
        num_workers=4
    )
    
    # Prepare with accelerator
    train_dataloader = accelerator.prepare(train_dataloader)
    
    model.train()
    global_step = step_offset
    
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training {category_name}")):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        global_step += 1

        if global_step % 50 == 0:
            logger.info(f"[{category_name}] Step {global_step}: loss = {loss.item():.4f}")

        if global_step % 1000 == 0:
            unwrapped = accelerator.unwrap_model(model)
            checkpoint_dir = f"{output_dir}/step-{global_step}-{category_name}"
            unwrapped.save_pretrained(checkpoint_dir, is_main_process=accelerator.is_main_process)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"💾 Saved checkpoint at step {global_step} ({category_name})")
    
    logger.info(f"✅ Completed training on {category_name} (total steps: {global_step - step_offset})")
    return global_step

# Main function
def main():
    accelerator = Accelerator()
    logger.info(f"Accelerator initialized. Device: {accelerator.device}")

    # Paths and configs
    model_config_path = "configs/mamba_config.json"
    dataset_path = "dataset/tagged"
    output_dir = "checkpoints"
    logging_dir = "logs"

    Path(output_dir).mkdir(exist_ok=True)
    Path(logging_dir).mkdir(exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mrcha033/YunMin-tokenizer-96k")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    config = AutoConfig.from_pretrained(model_config_path, local_files_only=True)
    model = MambaLMHeadModel(config)
    logger.info(f"Model initialized with {model.num_parameters()} parameters")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Prepare model and optimizer with accelerator
    model, optimizer = accelerator.prepare(model, optimizer)

    # Train on each category in order
    global_step = 0
    
    for category in LEARNING_ORDER:
        logger.info(f"📚 Processing dataset category: {category}")
        
        # Load dataset for this category
        category_dataset = load_dataset_category(dataset_path, category)
        
        if category_dataset is None:
            logger.warning(f"Skipping {category} - no valid data found")
            continue
        
        # Create scheduler for this category
        num_training_steps = len(DataLoader(category_dataset, batch_size=4))
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=min(500, num_training_steps // 10),
            num_training_steps=num_training_steps
        )
        lr_scheduler = accelerator.prepare(lr_scheduler)
        
        # Train on this category
        global_step = train_on_category(
            accelerator, model, tokenizer, category_dataset, category,
            optimizer, lr_scheduler, data_collator, output_dir, global_step
        )
        
        logger.info(f"🎯 Completed {category}. Total steps so far: {global_step}")

    # Final save
    accelerator.unwrap_model(model).save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    logger.info("✅ Training complete. Final model saved.")

if __name__ == "__main__":
    main() 