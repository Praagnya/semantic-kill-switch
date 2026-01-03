#!/usr/bin/env python3
"""
Fine-tune Qwen-2.5-1.5B on shutdown mechanism dataset
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class FinetuneConfig:
    """Configuration for finetuning"""
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Data
    dataset_path: str = "shutdown_dataset.jsonl"
    max_length: int = 2048

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # Training
    output_dir: str = "./qwen_shutdown_finetuned"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1  # Reduced for MPS memory
    gradient_accumulation_steps: int = 16  # Increased to maintain effective batch size
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 5
    save_steps: int = 50  # Save more frequently
    save_total_limit: int = 2
    fp16: bool = False  # Disable FP16 for MPS compatibility
    optim: str = "adamw_torch"
    dataloader_num_workers: int = 0  # Disable multiprocessing for MPS
    dataloader_pin_memory: bool = False  # Disable for MPS

    # Other
    seed: int = 42


def format_document_as_text(doc: dict) -> str:
    """
    Format a document from the dataset as training text.

    Args:
        doc: Document dict with 'document_type', 'title', 'content'

    Returns:
        Formatted text string
    """
    return f"""# {doc['title']}

Document Type: {doc['document_type']}

{doc['content']}"""


def prepare_dataset(dataset_path: str, tokenizer, max_length: int = 2048):
    """
    Load and prepare the shutdown dataset for training.

    Args:
        dataset_path: Path to the JSONL dataset
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset
    """
    # Load JSONL dataset
    dataset = load_dataset('json', data_files=dataset_path, split='train')

    def tokenize_function(examples):
        # Format each document
        texts = [format_document_as_text({
            'title': title,
            'document_type': doc_type,
            'content': content
        }) for title, doc_type, content in zip(
            examples['title'],
            examples['document_type'],
            examples['content']
        )]

        # Tokenize - DataCollatorForLanguageModeling will handle labels automatically
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )

        return tokenized

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    return tokenized_dataset


def setup_model_and_tokenizer(config: FinetuneConfig):
    """
    Load and setup the model and tokenizer.

    Args:
        config: Finetuning configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )

    # Add $STOP token if not present
    if "$STOP" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["$STOP"])

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    # For MPS (Mac), we need to use float32 and disable device_map
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,  # MPS doesn't fully support float16
        trust_remote_code=True
    )

    # Resize token embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA if enabled
    if config.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    """Main finetuning function"""
    # Configuration
    config = FinetuneConfig()

    print("=" * 50)
    print("Qwen-2.5-1.5B Shutdown Mechanism Fine-tuning")
    print("=" * 50)
    print(f"\nModel: {config.model_name}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    print(f"LoRA: {config.use_lora}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}\n")

    # Setup model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)

    # Prepare dataset
    print("Preparing dataset...")
    dataset_path = Path(__file__).parent / config.dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    train_dataset = prepare_dataset(str(dataset_path), tokenizer, config.max_length)
    print(f"Dataset size: {len(train_dataset)} documents\n")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        optim=config.optim,
        seed=config.seed,
        report_to=["tensorboard"],
        logging_dir=f"{config.output_dir}/logs",
        save_strategy="steps",
        push_to_hub=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
        pad_to_multiple_of=8  # Helps with efficiency
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving final model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
