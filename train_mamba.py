#!/usr/bin/env python3
"""
YunMin-Mamba 3B Pretraining Script
Main trainer script for EC2 environment
"""

import os
import logging
from pathlib import Path
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoConfig
)
from datasets import load_from_disk
from mamba_ssm.models.mamba_lm import MambaLMHeadModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    
    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    
    logger.info("Starting YunMin-Mamba 3B training...")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mrcha033/YunMin-tokenizer-96k")
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model config and initialize model
    logger.info("Loading model config and initializing Mamba model...")
    config = AutoConfig.from_pretrained("configs/mamba_config.json", local_files_only=True)
    model = MambaLMHeadModel(config)
    
    logger.info(f"Mamba model initialized with {model.num_parameters()} parameters")
    
    # Load dataset
    logger.info("Loading dataset...")
    train_data = load_from_disk("dataset/tagged")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=1000,
        logging_steps=50,
        fp16=True,  # Changed from bf16 for broader GPU compatibility
        report_to="none",
        warmup_steps=500,
        learning_rate=5e-5,
        weight_decay=0.01,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save final model and tokenizer
        trainer.save_model("./checkpoints/final")
        tokenizer.save_pretrained("./checkpoints/final")
        logger.info("Final model and tokenizer saved to ./checkpoints/final")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    logger.info("Training script completed.")

if __name__ == "__main__":
    main() 