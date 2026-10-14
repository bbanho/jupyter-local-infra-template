#!/usr/bin/env python3
# =============================================================================
# TRAIN_SHARD.PY — Train Single LoRA Shard for Holographic MoLE
# =============================================================================
# Usage: python scripts/train_shard.py --shard-id 0 --model Qwen/Qwen2.5-Coder-3B
# =============================================================================

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# =============================================================================
# Configuration
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA shard for MoLE")
    parser.add_argument("--shard-id", type=int, required=True, help="Shard index (0-7)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B", help="Base model")
    parser.add_argument("--data", type=str, help="Path to shard JSONL")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--steps", type=int, default=500, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (Ampere+)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

# =============================================================================
# Data Loading
# =============================================================================

def load_shard_data(data_path: str, tokenizer, max_length: int = 2048):
    """Load shard JSONL and tokenize."""
    texts = []
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            # Format: {"text": "code/content", "metadata": {...}}
            text = item.get("text", "") or item.get("content", "")
            if text.strip():
                texts.append(text)
    
    print(f"Loaded {len(texts)} samples from {data_path}")
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset

# =============================================================================
# Model Setup
# =============================================================================

def setup_model(model_name: str, r: int, alpha: int, bf16: bool):
    """Load base model with QLoRA config."""
    
    # Quantization config
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        use_flash_attention_2=True,
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # LoRA config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# =============================================================================
# Training
# =============================================================================

def main():
    args = parse_args()
    
    # Resolve paths
    data_path = args.data or f"/home/jovyan/work/data/shards/shard_{args.shard_id}.jsonl"
    output_dir = args.output or f"/home/jovyan/work/models/lora-shard-{args.shard_id:02d}"
    
    print(f"=== Training Shard {args.shard_id} ===")
    print(f"Model: {args.model}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"LoRA: r={args.r}, alpha={args.alpha}")
    print(f"Steps: {args.steps}, Batch: {args.batch_size} x {args.grad_accum}")
    
    # Seed
    torch.manual_seed(args.seed)
    
    # Model & Tokenizer
    model, tokenizer = setup_model(args.model, args.r, args.alpha, args.bf16)
    
    # Data
    train_dataset = load_shard_data(data_path, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.steps,
        warmup_steps=min(50, args.steps // 10),
        learning_rate=args.lr,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        fp16=not args.bf16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",  # Disable wandb/tensorboard unless configured
        seed=args.seed,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("=== Starting Training ===")
    trainer.train()
    
    # Save final adapter
    print(f"=== Saving Adapter to {output_dir} ===")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training config
    config = {
        "shard_id": args.shard_id,
        "base_model": args.model,
        "data_path": data_path,
        "lora_r": args.r,
        "lora_alpha": args.alpha,
        "max_steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "bf16": args.bf16,
    }
    with open(Path(output_dir) / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Training complete!")

if __name__ == "__main__":
    main()