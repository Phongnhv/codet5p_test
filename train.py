import os
import json
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import logging
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ====================================
# GLOBAL CONFIGURATION PARAMETERS
# ====================================

@dataclass
class Config:
    # Model parameters
    model_name: str = "Salesforce/codet5p-2b"
    max_input_length: int = 512
    max_target_length: int = 4  # For binary classification (just "0" or "1")
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Training parameters - Optimized for 1-2 A30 GPUs (24GB VRAM)
    per_device_train_batch_size: int = 2  # Small batch size due to model size
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8  # Effective batch size = 2 * 8 = 16
    learning_rate: float = 5e-5
    num_train_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Memory optimization
    fp16: bool = True  # Use mixed precision
    dataloader_pin_memory: bool = False
    gradient_checkpointing: bool = True
    
    # Paths
    output_dir: str = "./codet5p_inconsistency_model"
    logging_dir: str = "./logs"
    cache_dir: str = "./cache"
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True

    def __post_init__(self):
        if self.lora_target_modules is None:
            # Target modules for CodeT5p (both encoder and decoder)
            self.lora_target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "wi_0", "wi_1", "wo"
            ]

# Initialize global config
config = Config()

# GPU Configuration suggestions
def print_gpu_recommendations():
    """Print optimized configurations for different GPU setups"""
    print("\n" + "="*60)
    print("GPU CONFIGURATION RECOMMENDATIONS")
    print("="*60)
    
    print("\n🚀 FOR 1x A30 (24GB VRAM):")
    print("  - per_device_train_batch_size: 4")
    print("  - gradient_accumulation_steps: 8")
    print("  - fp16: True")
    print("  - gradient_checkpointing: True")
    print("  - Effective batch size: 32")
    print("  - Expected memory usage: ~20-22GB")
    
    print("\n🚀🚀 FOR 2x A30 (48GB total VRAM):")
    print("  - per_device_train_batch_size: 6")
    print("  - gradient_accumulation_steps: 6")
    print("  - fp16: True")
    print("  - gradient_checkpointing: True")
    print("  - Effective batch size: 72")
    print("  - Expected memory usage: ~20-22GB per GPU")
    print("  - Use: python -m torch.distributed.launch --nproc_per_node=2 train.py")
    
    print("\n💡 MEMORY OPTIMIZATION TIPS:")
    print("  - Use gradient_checkpointing=True (saves ~30% memory)")
    print("  - Use fp16=True (halves memory usage)")
    print("  - Reduce max_input_length if needed")
    print("  - Use LoRA instead of full fine-tuning (saves ~90% parameters)")
    print("="*60)

# ====================================
# DATA PROCESSING
# ====================================

class CodeCommentDataset:
    def __init__(self, tokenizer, max_input_length: int = 512, max_target_length: int = 4):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def prepare_input_text(self, code: str, comment: str, type_info: str) -> str:
        """
        Prepare input text by combining code, comment, and type information
        """
        # Format: [TYPE] Comment: {comment} Code: {code}
        input_text = f"[{type_info.upper()}] Comment: {comment} Code: {code}"
        return input_text
    
    def tokenize_function(self, examples):
        """Tokenize inputs and targets"""
        # Prepare input texts
        inputs = []
        for i in range(len(examples['code'])):
            input_text = self.prepare_input_text(
                examples['code'][i], 
                examples['comment'][i], 
                examples['type'][i]
            )
            inputs.append(input_text)
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            padding=False,
            truncation=True,
        )
        
        # Tokenize targets (labels: "0" or "1")
        targets = [str(label) for label in examples['label']]
        labels = self.tokenizer(
            targets,
            max_length=self.max_target_length,
            padding=False,
            truncation=True,
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def load_and_prepare_data(data_path: str, tokenizer) -> DatasetDict:
    """
    Load and prepare the dataset
    Expected format: JSON files with 'code', 'comment', 'type', 'label' fields
    """
    dataset_processor = CodeCommentDataset(tokenizer, config.max_input_length, config.max_target_length)
    
    # Load datasets
    splits = ['post_hoc', 'post_hoc_valid', 'post_hoc_test']
    datasets = {}
    
    for split in splits:
        file_path = os.path.join(data_path, f"{split}.json")
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} not found, skipping {split} split")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_dict(data)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            dataset_processor.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split} dataset"
        )
        
        datasets[split] = tokenized_dataset
        logger.info(f"Loaded {len(tokenized_dataset)} examples for {split}")
    
    return DatasetDict(datasets)

# ====================================
# MODEL SETUP
# ====================================

def setup_model_and_tokenizer():
    """Setup CodeT5p model with LoRA configuration"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir
    )
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        inference_mode=False,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# ====================================
# EVALUATION METRICS
# ====================================

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1 score"""
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Handle padding in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Convert to binary labels
    pred_labels = []
    true_labels = []
    
    for pred, true in zip(decoded_preds, decoded_labels):
        try:
            pred_label = int(pred.strip())
            true_label = int(true.strip())
            pred_labels.append(pred_label)
            true_labels.append(true_label)
        except ValueError:
            # Handle cases where prediction is not a valid integer
            pred_labels.append(0)  # Default to 0 for invalid predictions
            true_labels.append(int(true.strip()) if true.strip().isdigit() else 0)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# ====================================
# TRAINING
# ====================================

def train_model(model, tokenizer, dataset_dict):
    """Train the model with early stopping"""
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Check if we're in distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        
        # Evaluation and saving
        evaluation_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        
        # Memory optimization
        fp16=config.fp16,
        dataloader_pin_memory=config.dataloader_pin_memory,
        gradient_checkpointing=config.gradient_checkpointing,
        
        # Distributed training settings
        local_rank=local_rank,
        ddp_find_unused_parameters=False,  # Set to True if you encounter issues
        dataloader_num_workers=2 if world_size > 1 else 0,
        
        # Logging (only on main process)
        logging_dir=config.logging_dir,
        logging_steps=100,
        report_to=None,  # Disable wandb/tensorboard
        disable_tqdm=local_rank not in [-1, 0],  # Disable progress bars on non-main processes
        
        # Other
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=config.early_stopping_threshold
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['valid'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    return trainer

# ====================================
# EVALUATION
# ====================================

def evaluate_model(trainer, dataset_dict):
    """Evaluate model on test set"""
    if 'test' not in dataset_dict:
        logger.warning("No test dataset found, skipping evaluation")
        return
    
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(dataset_dict['test'])
    
    logger.info("Test Results:")
    for key, value in test_results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save test results
    with open(os.path.join(config.output_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

# ====================================
# DISTRIBUTED TRAINING SETUP
# ====================================

def setup_distributed_training():
    """Setup distributed training if multiple GPUs are available"""
    if torch.cuda.device_count() > 1:
        logger.info(f"Found {torch.cuda.device_count()} GPUs, setting up distributed training")
        # The distributed setup is handled by torchrun/torch.distributed.launch
        # This function just logs the setup
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True
    return False

def get_config_for_gpus(num_gpus: int = None):
    """Get optimized config based on number of GPUs"""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus == 1:
        # Single GPU configuration
        config.per_device_train_batch_size = 2
        config.gradient_accumulation_steps = 8
        logger.info("Using single GPU configuration")
    elif num_gpus == 2:
        # Dual GPU configuration
        config.per_device_train_batch_size = 2
        config.gradient_accumulation_steps = 10
        logger.info("Using dual GPU configuration")
    else:
        logger.info(f"Using {num_gpus} GPU configuration")
    
    effective_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps * num_gpus
    logger.info(f"Effective batch size: {effective_batch_size}")
    
    return config

# ====================================
# MAIN FUNCTION
# ====================================

def main():
    """Main training pipeline"""
    print_gpu_recommendations()
    
    # Print GPU configuration info
    print("\n" + "="*60)
    print("🔍 GPU CONFIGURATION DETECTION")
    print("="*60)
    
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"Total GPUs detected by PyTorch: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    
    # Setup distributed training
    is_distributed = setup_distributed_training()
    
    # Auto-configure based on available GPUs
    config = get_config_for_gpus()
    
    # Create directories (only on main process for distributed training)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.logging_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True)
    
    # Wait for main process to create directories
    if is_distributed:
        torch.distributed.barrier()
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load and prepare data
    data_path = "./posthoc"  # Update this path to your data directory
    logger.info(f"Loading data from {data_path}...")
    dataset_dict = load_and_prepare_data(data_path, tokenizer)
    
    # Train model
    trainer = train_model(model, tokenizer, dataset_dict)
    
    # Evaluate on test set (only on main process)
    if local_rank == 0:
        evaluate_model(trainer, dataset_dict)
        logger.info(f"Training completed! Model saved to {config.output_dir}")
    
    # Cleanup distributed training
    if is_distributed:
        torch.distributed.destroy_process_group()

# ====================================
# INFERENCE FUNCTION
# ====================================

def load_and_predict(model_path: str, code: str, comment: str, type_info: str):
    """Load trained model and make prediction"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA model
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Prepare input
    input_text = f"[{type_info.upper()}] Comment: {comment} Code: {code}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=config.max_target_length,
            num_beams=1,
            do_sample=False
        )
    
    # Decode prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        label = int(prediction.strip())
        return label, "consistent" if label == 0 else "inconsistent"
    except ValueError:
        return 0, "consistent"  # Default to consistent if prediction is invalid

if __name__ == "__main__":
    main()

# ====================================
# EXAMPLE USAGE FOR INFERENCE
# ====================================

# Example usage after training:
"""
# Load and predict
code = "def add_numbers(a, b): return a + b"
comment = "Subtracts two numbers"
type_info = "summary"

label, description = load_and_predict("./codet5p_inconsistency_model", code, comment, type_info)
print(f"Prediction: {label} ({description})")
"""