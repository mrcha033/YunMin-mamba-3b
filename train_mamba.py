#!/usr/bin/env python3
"""
YunMin-Mamba 3B Pretraining Script with Accelerate
Supports multiple dataset categories with learning order
"""
import math
import torch
import logging
import sys
import os
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoConfig, get_scheduler
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator

# Add current directory to Python path for mamba_simple import
sys.path.append('/app')
sys.path.append('.')

# Updated import for mamba-minimal
from mamba_simple import Mamba

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

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

# CUDA verification
def verify_cuda():
    logger.info("🔍 Verifying CUDA environment...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA not available - training will be slow!")

# Simple Mamba Language Model wrapper
class SimpleMambaLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Token embeddings
        self.embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
        
        # Mamba layers
        self.layers = torch.nn.ModuleList([
            Mamba(config.d_model) for _ in range(config.num_hidden_layers)
        ])
        
        # Layer norm
        self.norm = torch.nn.LayerNorm(config.d_model)
        
        # Language modeling head
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, input_ids, labels=None, **kwargs):
        # Get embeddings
        x = self.embedding(input_ids)
        
        # Pass through Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        # Final norm
        x = self.norm(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        # Return in transformers format
        from types import SimpleNamespace
        return SimpleNamespace(loss=loss, logits=logits)
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save model for compatibility"""
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(save_directory) / "pytorch_model.bin")
        # Save config
        config_dict = {
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "num_hidden_layers": self.config.num_hidden_layers
        }
        with open(Path(save_directory) / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
    
    def num_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

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

def train_on_category(accelerator, model, tokenizer, category_dataset, category, 
                     optimizer, lr_scheduler, data_collator, output_dir, step_offset, 
                     batch_size, num_workers, save_steps):
    """Train on a specific dataset category"""
    logger.info(f"🚀 Starting training on {category}...")
    
    # Create DataLoader for this category
    train_dataloader = DataLoader(
        category_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=data_collator, 
        num_workers=num_workers
    )
    
    # Prepare with accelerator
    train_dataloader = accelerator.prepare(train_dataloader)
    
    model.train()
    global_step = step_offset
    
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training {category}")):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        global_step += 1

        if global_step % 50 == 0:
            logger.info(f"[{category}] Step {global_step}: loss = {loss.item():.4f}")

        if global_step % save_steps == 0:
            unwrapped = accelerator.unwrap_model(model)
            checkpoint_dir = f"{output_dir}/step-{global_step}-{category}"
            unwrapped.save_pretrained(checkpoint_dir, is_main_process=accelerator.is_main_process)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"💾 Saved checkpoint at step {global_step} ({category})")
    
    logger.info(f"✅ Completed training on {category} (total steps: {global_step - step_offset})")
    return global_step

# Main function
def main():
    # Verify CUDA first
    verify_cuda()
    
    accelerator = Accelerator()
    logger.info(f"Accelerator initialized. Device: {accelerator.device}")

    # SageMaker paths - Use actual SageMaker standard paths
    input_path = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    model_path = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')  
    output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output')
    
    # Hyperparameters (SageMaker provides these)
    hyperparams_path = "/opt/ml/input/config/hyperparameters.json"
    hyperparams = {}
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
    
    # Paths and configs
    model_config_path = "configs/mamba_config.json"
    dataset_path = input_path
    output_dir = model_path
    logging_dir = "logs"

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    Path(logging_dir).mkdir(exist_ok=True, parents=True)
    Path(output_path).mkdir(exist_ok=True, parents=True)

    # Get hyperparameters with defaults
    learning_rate = float(hyperparams.get('learning_rate', '5e-5'))
    batch_size = int(hyperparams.get('batch_size', '4'))
    num_workers = int(hyperparams.get('num_workers', '4'))
    save_steps = int(hyperparams.get('save_steps', '1000'))
    
    logger.info(f"Hyperparameters: lr={learning_rate}, batch_size={batch_size}, save_steps={save_steps}")

    # Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mrcha033/YunMin-tokenizer-96k")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    logger.info("Loading model config and initializing Mamba model...")
    with open(model_config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create a simple config object
    from types import SimpleNamespace
    config = SimpleNamespace(**config_dict)
    
    # Verify mamba-ssm import
    try:
        model = SimpleMambaLM(config)
        logger.info(f"✅ Mamba model initialized successfully with {model.num_parameters()} parameters")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Mamba model: {e}")
        raise

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

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
        num_training_steps = len(DataLoader(category_dataset, batch_size=batch_size))
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
            optimizer, lr_scheduler, data_collator, output_dir, global_step, 
            batch_size, num_workers, save_steps
        )
        
        logger.info(f"🎯 Completed {category}. Total steps so far: {global_step}")

    # Final save to SageMaker model directory
    final_model_path = f"{output_dir}/final"
    accelerator.unwrap_model(model).save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"✅ Training complete. Final model saved to {final_model_path}")
    
    # Copy logs to output directory for SageMaker
    import shutil
    if os.path.exists("logs"):
        shutil.copytree("logs", f"{output_path}/logs", dirs_exist_ok=True)
        logger.info(f"📋 Logs copied to {output_path}/logs")

if __name__ == "__main__":
    main() 