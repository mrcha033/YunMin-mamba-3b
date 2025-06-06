#!/usr/bin/env python3
"""
YunMin-Mamba 3B Pretraining Script with Accelerate
Supports multiple dataset categories with learning order
"""
import torch
import logging
import os
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
import numpy as np


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

def cleanup_memory():
    """Clean up GPU memory aggressively"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection
        import gc
        gc.collect()


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

def filter_by_length(dataset, tokenizer, max_length=1024):
    """Filter dataset by sequence length to reduce memory usage"""
    original_size = len(dataset)
    
    def is_valid_length(example):
        if 'input_ids' in example:
            return len(example['input_ids']) <= max_length
        elif 'text' in example:
            tokens = tokenizer.encode(example['text'], add_special_tokens=False)
            return len(tokens) <= max_length
        return True
    
    filtered = dataset.filter(is_valid_length)
    filtered_size = len(filtered)
    
    logger.info(f"Filtered dataset: {original_size} -> {filtered_size} examples (max_length={max_length})")
    
    # If too many samples were filtered out, increase max_length gradually
    if filtered_size < original_size * 0.1:  # Less than 10% remaining
        logger.warning(f"Too many samples filtered out ({filtered_size}/{original_size}). Trying larger max_length...")
        return filter_by_length(dataset, tokenizer, max_length * 2)
    
    # If no samples remain, use original dataset with truncation
    if filtered_size == 0:
        logger.warning(f"No samples remain after filtering. Using original dataset with truncation.")
        return dataset
    
    return filtered

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
                     optimizer_params, lr_scheduler, data_collator, checkpoint_root, step_offset,
                     batch_size, num_workers, save_steps, resume_from_checkpoint=None, load_checkpoint=False):
    """Train on a specific dataset category with DeepSpeed ZeRO optimization"""
    logger.info(f"🚀 Starting training on {category}...")
    
    # DeepSpeed handles gradient accumulation automatically
    logger.info(f"Training config: batch_size={batch_size}, using DeepSpeed ZeRO Stage 3")
    
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

    if load_checkpoint and resume_from_checkpoint:
        logger.info(f"🔄 Loading checkpoint from {resume_from_checkpoint}")
        try:
            accelerator.load_state(resume_from_checkpoint)
            logger.info(f"✅ Successfully resumed from step {step_offset}")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            logger.info("🆕 Starting fresh training instead")
            step_offset = 0
            resume_from_checkpoint = None
    
    model.train()
    global_step = step_offset
    
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training {category}")):
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Use accelerator's backward for DeepSpeed compatibility
        accelerator.backward(loss)

        global_step += 1

        # Memory cleanup
        if global_step % 10 == 0:  # More frequent cleanup
            cleanup_memory()
        
        if global_step % 50 == 0:
            logger.info(f"[{category}] Step {global_step}: loss = {loss.item():.4f}")

        if global_step % save_steps == 0:
            checkpoint_dir = f"{checkpoint_root}/step-{global_step}-{category}"
            # Use accelerator's save method for DeepSpeed
            accelerator.save_state(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"💾 Saved checkpoint at step {global_step} ({category})")
            
            # Clean up memory after saving
            cleanup_memory()
    
    logger.info(f"✅ Completed training on {category} (total steps: {global_step - step_offset})")
    
    # Final memory cleanup after completing category
    cleanup_memory()
    return global_step

def extract_text_from_example(example, category="unknown"):
    """Extract text from a single example based on its structure"""
    texts = []
    
    try:
        # 1. Simple contents structure
        if 'contents' in example and isinstance(example['contents'], str):
            texts.append(example['contents'])
        
        # 2. Learning data (korean_textbook)
        elif 'learning_data_info' in example:
            info = example['learning_data_info']
            for field in ['question', 'explanation', 'passage']:
                if field in info and isinstance(info[field], str):
                    texts.append(info[field])
        
        # 3. Academic data (training_data_info.section_info)
        elif 'training_data_info' in example and 'section_info' in example['training_data_info']:
            sections = example['training_data_info']['section_info']
            if isinstance(sections, list):
                for section in sections:
                    for field in ['original_text', 'summary_text']:
                        if field in section and isinstance(section[field], str):
                            texts.append(section[field])
        
        # 4. Web data (named_entity)
        elif 'named_entity' in example:
            entities = example['named_entity']
            if isinstance(entities, list):
                for entity in entities:
                    for field in ['title', 'subtitle', 'content']:
                        if field in entity and isinstance(entity[field], list):
                            for item in entity[field]:
                                if isinstance(item, dict) and 'sentence' in item:
                                    texts.append(item['sentence'])
        
        # 5. Papers data - list structure
        elif isinstance(example, list) and len(example) > 0 and 'data' in example[0]:
            for item in example:
                if 'data' in item and isinstance(item['data'], list):
                    for data_item in item['data']:
                        # Extract title
                        if 'title' in data_item and isinstance(data_item['title'], str):
                            texts.append(data_item['title'])
                        # Extract summary_entire
                        if 'summary_entire' in data_item and isinstance(data_item['summary_entire'], str):
                            texts.append(data_item['summary_entire'])
                        # Extract summary_section
                        if 'summary_section' in data_item and isinstance(data_item['summary_section'], list):
                            for section in data_item['summary_section']:
                                for field in ['orginal_text', 'summary_text']:  # Note: 'orginal' not 'original'
                                    if field in section and isinstance(section[field], str):
                                        texts.append(section[field])
        
        # 6. Main data (data_info)
        elif 'data_info' in example:
            data_info = example['data_info']
            if isinstance(data_info, list):
                for item in data_info:
                    for field in ['data_title', 'contents']:
                        if field in item and isinstance(item[field], str):
                            texts.append(item[field])
        
        # 7. Single paper structure
        elif 'data' in example and isinstance(example['data'], list):
            for data_item in example['data']:
                # Extract title
                if 'title' in data_item and isinstance(data_item['title'], str):
                    texts.append(data_item['title'])
                # Extract summary_entire
                if 'summary_entire' in data_item and isinstance(data_item['summary_entire'], str):
                    texts.append(data_item['summary_entire'])
                # Extract summary_section
                if 'summary_section' in data_item and isinstance(data_item['summary_section'], list):
                    for section in data_item['summary_section']:
                        for field in ['orginal_text', 'summary_text']:  # Note: 'orginal' not 'original'
                            if field in section and isinstance(section[field], str):
                                texts.append(section[field])
        
        # 8. Fallback: try to find any string field
        else:
            def extract_strings_recursive(obj, depth=0):
                if depth > 3:  # Prevent infinite recursion
                    return []
                strings = []
                if isinstance(obj, str):
                    strings.append(obj)
                elif isinstance(obj, dict):
                    for value in obj.values():
                        strings.extend(extract_strings_recursive(value, depth + 1))
                elif isinstance(obj, list):
                    for item in obj:
                        strings.extend(extract_strings_recursive(item, depth + 1))
                return strings
            
            fallback_texts = extract_strings_recursive(example)
            # Filter out very short strings (likely metadata)
            texts.extend([t for t in fallback_texts if len(t.strip()) > 10])
    
    except Exception as e:
        logger.warning(f"Error extracting text from example in {category}: {e}")
        return []
    
    # Clean and filter texts
    cleaned_texts = []
    for text in texts:
        if text and isinstance(text, str):
            cleaned = text.strip()
            if len(cleaned) > 5:  # Filter out very short texts
                cleaned_texts.append(cleaned)
    
    return cleaned_texts

def preprocess_dataset(dataset, tokenizer, max_length=1024, category="unknown"):
    """Preprocess dataset to extract and tokenize text from various JSON structures"""
    logger.info(f"📊 Processing {category} dataset")
    logger.info(f"Dataset size: {len(dataset)} examples")
    logger.info(f"Dataset columns: {dataset.column_names}")
    logger.info(f"Dataset features: {dataset.features}")
    
    # If dataset is already tokenized (has input_ids), return as is without filtering
    if 'input_ids' in dataset.column_names:
        logger.info("Dataset already tokenized, using all data without length filtering")
        
        # Sample a few examples to check sequence lengths
        sample_lengths = []
        sample_size = min(100, len(dataset))
        for i in range(sample_size):
            sample_lengths.append(len(dataset[i]['input_ids']))
        
        if sample_lengths:
            lengths_array = np.array(sample_lengths)
            logger.info(f"Sample sequence lengths (n={sample_size}):")
            logger.info(f"  Min: {lengths_array.min()}, Max: {lengths_array.max()}")
            logger.info(f"  Mean: {lengths_array.mean():.1f}, Median: {np.median(lengths_array):.1f}")
        
        logger.info(f"Using all {len(dataset)} examples for training")
        return dataset
    
    # Extract texts from examples
    all_texts = []
    failed_extractions = 0
    
    logger.info(f"🔍 Extracting texts from {category} dataset...")
    
    for i, example in enumerate(dataset):
        texts = extract_text_from_example(example, category)
        
        if texts:
            # Join multiple texts with newline
            combined_text = '\n'.join(texts)
            all_texts.append(combined_text)
        else:
            failed_extractions += 1
            if failed_extractions <= 5:  # Show first few failures for debugging
                logger.warning(f"Failed to extract text from example {i}: {list(example.keys())}")
    
    logger.info(f"✅ Extracted {len(all_texts)} texts from {len(dataset)} examples")
    logger.info(f"❌ Failed extractions: {failed_extractions}")
    
    if len(all_texts) == 0:
        logger.error("No texts extracted from dataset!")
        return None
    
    # Create new dataset with extracted texts
    from datasets import Dataset
    text_dataset = Dataset.from_dict({'text': all_texts})
    
    # Tokenize the extracted texts
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=False,  # Don't truncate - allow all lengths
            padding=False,
            return_tensors=None
        )
    
    # Apply tokenization
    tokenized_dataset = text_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc=f"Tokenizing {category} dataset"
    )
    
    logger.info(f"📝 Tokenized dataset: {len(tokenized_dataset)} examples")
    return tokenized_dataset

# Main function
def main():
    parser = argparse.ArgumentParser(description="YunMin-Mamba training")
    parser.add_argument(
        "--model-config-path",
        default=None,
        help="Path to the model configuration JSON file",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional local path to the dataset directory",
    )
    args, _ = parser.parse_known_args()
    # Set PyTorch memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Enable memory efficient attention if available
    torch.backends.cuda.enable_flash_sdp(True)
    
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
    model_config_path = (
        args.model_config_path
        or hyperparams.get("model_config_path")
        or os.environ.get("MODEL_CONFIG_PATH", "configs/mamba_config.json")
    )
    dataset_path = args.dataset_path or input_path
    logger.info(f"Dataset path resolved to: {dataset_path}")

    # Separate directories for checkpoints and final model
    checkpoint_root = os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints')
    model_dir = model_path
    logging_dir = "logs"

    Path(checkpoint_root).mkdir(exist_ok=True, parents=True)
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    Path(logging_dir).mkdir(exist_ok=True, parents=True)
    Path(output_path).mkdir(exist_ok=True, parents=True)

    # Get hyperparameters with defaults optimized for memory efficiency
    learning_rate = float(hyperparams.get('learning_rate', '5e-5'))
    batch_size = int(hyperparams.get('batch_size', '1'))  # Minimum for p4d.24xlarge with long sequences
    num_workers = int(hyperparams.get('num_workers', '0'))  # Reduce to 0 for memory
    save_steps = int(hyperparams.get('save_steps', '1000'))
    max_seq_length = int(hyperparams.get('max_seq_length', '8192'))  # Increased default limit
    
    # Automatic checkpoint discovery for SageMaker Spot training
    resume_from_checkpoint = None
    start_step = 0
    
    # SageMaker automatically restores checkpoints to /opt/ml/checkpoints
    ckpt_dir = Path(checkpoint_root)
    if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        # Find all checkpoint directories
        checkpoints = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith('step-')], 
                           key=lambda x: x.stat().st_mtime)
        
        if checkpoints:
            resume_from_checkpoint = str(checkpoints[-1])  # Latest checkpoint
            # Extract step number from directory name (e.g., "step-3000-main_data" -> 3000)
            try:
                step_parts = checkpoints[-1].name.split("-")
                if len(step_parts) >= 2:
                    start_step = int(step_parts[1])
                logger.info(f"🔍 Found checkpoint: {resume_from_checkpoint}")
                logger.info(f"📊 Will resume from step: {start_step}")
            except (ValueError, IndexError) as e:
                logger.warning(f"⚠️ Could not parse step number from checkpoint name: {e}")
                start_step = 0
        else:
            logger.info("📂 Checkpoint directory exists but no valid checkpoints found")
    else:
        logger.info("📂 No checkpoint directory found - starting fresh training")
    
    logger.info(f"Hyperparameters: lr={learning_rate}, batch_size={batch_size}, max_seq_length={max_seq_length}, save_steps={save_steps}")
    logger.info(f"🔧 Using DeepSpeed ZeRO Stage 3 with optimizer offloading for memory efficiency")
    
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        logger.info(f"🔄 Found checkpoint at {resume_from_checkpoint}. Will load after initialization")
    elif resume_from_checkpoint:
        logger.warning(f"⚠️ Checkpoint path {resume_from_checkpoint} not found. Starting fresh training.")
        start_step = 0
        resume_from_checkpoint = None
    else:
        logger.info("🆕 No valid checkpoint found. Starting from scratch.")

    # Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mrcha033/YunMin-tokenizer-96k")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Don't set max_length limit on tokenizer to allow longer sequences
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Model
    logger.info("Loading model config and initializing Mamba model via transformers...")
    config = AutoConfig.from_pretrained(model_config_path)
    try:
        model = AutoModelForCausalLM.from_config(config)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        logger.info(
            f"✅ Mamba model initialized successfully with {sum(p.numel() for p in model.parameters())} parameters"
        )
        logger.info("🔧 Gradient checkpointing enabled for memory efficiency")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Mamba model: {e}")
        raise

    # Data collator with sequence length limit
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8  # Optimize for tensor cores
    )

    # DeepSpeed will handle optimizer creation automatically
    # Prepare model with accelerator (DeepSpeed)
    model = accelerator.prepare(model)
    
    # Train on each category in order
    global_step = start_step  # Start from checkpoint step
    
    loaded_checkpoint = False
    for category in LEARNING_ORDER:
        logger.info(f"📚 Processing dataset category: {category}")
        
        # Load dataset for this category
        category_dataset = load_dataset_category(dataset_path, category)
        
        if category_dataset is None:
            logger.warning(f"Skipping {category} - no valid data found")
            continue
            
        # Preprocess dataset (tokenize and truncate)
        category_dataset = preprocess_dataset(category_dataset, tokenizer, max_seq_length, category)
        
        # Check if dataset is empty after preprocessing
        if category_dataset is None or len(category_dataset) == 0:
            logger.warning(f"Skipping {category} - dataset is empty after preprocessing")
            continue
        
        # Create scheduler for this category
        num_training_steps = len(DataLoader(category_dataset, batch_size=batch_size))
        
        # DeepSpeed handles optimizer creation, so we skip lr_scheduler for now
        # lr_scheduler will be handled by DeepSpeed configuration
        lr_scheduler = None
        
        # Train on this category
        global_step = train_on_category(
            accelerator, model, tokenizer, category_dataset, category,
            None, lr_scheduler, data_collator, checkpoint_root, global_step,
            batch_size, num_workers, save_steps,
            resume_from_checkpoint if not loaded_checkpoint else None,
            load_checkpoint=not loaded_checkpoint and resume_from_checkpoint is not None
        )
        loaded_checkpoint = True
        
        logger.info(f"🎯 Completed {category}. Total steps so far: {global_step}")

    # Final save to SageMaker model directory using Accelerator
    final_model_path = f"{model_dir}/final"
    accelerator.save_state(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"✅ Training complete. Final model saved to {final_model_path}")
    
    # Copy logs to output directory for SageMaker
    import shutil
    if os.path.exists("logs"):
        shutil.copytree("logs", f"{output_path}/logs", dirs_exist_ok=True)
        logger.info(f"📋 Logs copied to {output_path}/logs")

if __name__ == "__main__":
    main()
