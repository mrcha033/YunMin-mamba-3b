#!/usr/bin/env python3
"""
YunMin-Mamba 3B Pretraining Script with Accelerate
"""
import os
import math
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
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

    # Dataset
    dataset = load_from_disk(dataset_path)["train"]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=4, collate_fn=data_collator, num_workers=4)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Scheduler
    num_training_steps = len(train_dataloader) * 1  # 1 epoch
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    model.train()
    logger.info("🚀 Starting training loop...")

    for step, batch in enumerate(tqdm(train_dataloader)):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            logger.info(f"Step {step}: loss = {loss.item():.4f}")

        if step % 1000 == 0 and step > 0:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(f"{output_dir}/step-{step}", is_main_process=accelerator.is_main_process)
            tokenizer.save_pretrained(f"{output_dir}/step-{step}")
            logger.info(f"💾 Saved checkpoint at step {step}")

    # Final save
    accelerator.unwrap_model(model).save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    logger.info("✅ Training complete. Final model saved.")

if __name__ == "__main__":
    main() 