"""
Fine-tune text models on SST-2 dataset.

This script trains DistilBERT and MiniLM models from scratch on SST-2.
Use this if you want full control over training hyperparameters or need
to train on a different dataset.

By default, experiments use pre-fine-tuned checkpoints from HuggingFace:
- distilbert-base-uncased-finetuned-sst-2-english
- philschmid/MiniLM-L6-H384-uncased-sst2

Output: training/checkpoints/{model_name}/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from src.utils.logger import get_logger

LOGGER = get_logger("training.finetune_text")

# Training hyperparameters
MODELS = {
    'distilbert': 'distilbert-base-uncased',
    'minilm': 'microsoft/MiniLM-L12-H384-uncased'
}

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_LENGTH = 128
WARMUP_STEPS = 500


def load_sst2_dataset():
    """Load SST-2 dataset from HuggingFace."""
    LOGGER.info("Loading SST-2 dataset...")
    
    dataset = load_dataset("glue", "sst2")
    
    LOGGER.info(f"Train: {len(dataset['train'])} examples")
    LOGGER.info(f"Validation: {len(dataset['validation'])} examples")
    
    return dataset


def tokenize_dataset(dataset, tokenizer):
    """Tokenize dataset."""
    LOGGER.info("Tokenizing dataset...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({'loss': loss.item(), 'acc': correct / total})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(model_name, model_key):
    """Train a single model."""
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Training {model_name}")
    LOGGER.info(f"{'='*60}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device: {device}")
    
    # Load dataset
    dataset = load_sst2_dataset()
    
    # Load tokenizer and model
    LOGGER.info(f"Loading model: {model_key}")
    tokenizer = AutoTokenizer.from_pretrained(model_key)
    model = AutoModelForSequenceClassification.from_pretrained(model_key, num_labels=2)
    model.to(device)
    
    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_dataset['train'],
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        tokenized_dataset['validation'],
        batch_size=BATCH_SIZE
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        LOGGER.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        LOGGER.info(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_dataloader, device)
        LOGGER.info(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            output_dir = Path(f"training/checkpoints/{model_name}_sst2")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            LOGGER.info(f"Saved best model to {output_dir} (val_acc={val_acc:.4f})")
    
    LOGGER.info(f"\nTraining complete. Best validation accuracy: {best_val_accuracy:.4f}")
    
    return best_val_accuracy


def main():
    LOGGER.info("Starting text model fine-tuning on SST-2...")
    LOGGER.info(f"Batch size: {BATCH_SIZE}")
    LOGGER.info(f"Learning rate: {LEARNING_RATE}")
    LOGGER.info(f"Epochs: {NUM_EPOCHS}")
    
    results = {}
    
    for model_name, model_key in MODELS.items():
        try:
            best_acc = train_model(model_name, model_key)
            results[model_name] = best_acc
        except Exception as e:
            LOGGER.error(f"Failed to train {model_name}: {e}")
            results[model_name] = None
    
    # Summary
    LOGGER.info("\n" + "="*60)
    LOGGER.info("Training Summary")
    LOGGER.info("="*60)
    for model_name, best_acc in results.items():
        if best_acc is not None:
            LOGGER.info(f"{model_name}: {best_acc:.4f}")
        else:
            LOGGER.info(f"{model_name}: FAILED")
    
    LOGGER.info("\nCheckpoints saved to training/checkpoints/")
    LOGGER.info("To use these checkpoints, update src/models/model_zoo.py")


if __name__ == "__main__":
    main()

