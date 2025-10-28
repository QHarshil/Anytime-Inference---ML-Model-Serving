"""
Accuracy profiling for text and image models.

Uses real validation/test splits:
- Text: SST-2 validation set (872 examples)
- Image: CIFAR-10 test set (10,000 examples)

Measures:
- Accuracy: Fraction of correct predictions
- Coverage: For cascade configurations, fraction that exit at Stage 1

Output: results/accuracy_profiles.csv
Schema: task,model,variant,exit_policy,threshold,accuracy,coverage,num_samples
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models.model_zoo import ModelZoo
from src.models.cascade import CascadeEvaluator
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.profile_accuracy")

# Configuration
TEXT_MODELS = ["distilbert", "minilm"]
IMAGE_MODELS = ["mobilenetv2", "resnet18"]
VARIANTS = ["fp32", "fp16", "int8"]
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
CASCADE_THRESHOLDS = [0.7, 0.8, 0.9, 0.95]
BATCH_SIZE = 32
MAX_TEXT_SAMPLES = 872  # Full SST-2 validation set
MAX_IMAGE_SAMPLES = 10000  # Full CIFAR-10 test set


def load_sst2_validation():
    """Load SST-2 validation set."""
    LOGGER.info("Loading SST-2 validation set...")
    dataset = load_dataset("glue", "sst2", split="validation")
    
    texts = [example["sentence"] for example in dataset]
    labels = [example["label"] for example in dataset]
    
    LOGGER.info(f"Loaded {len(texts)} examples from SST-2 validation set")
    return texts, labels


def load_cifar10_test():
    """Load CIFAR-10 test set."""
    LOGGER.info("Loading CIFAR-10 test set...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    
    LOGGER.info(f"Loaded {len(dataset)} examples from CIFAR-10 test set")
    return dataset


def evaluate_text_model(model_name, variant, device, texts, labels):
    """Evaluate text model accuracy on SST-2 validation set."""
    LOGGER.info(f"Evaluating {model_name} ({variant}, {device}) on SST-2...")
    
    zoo = ModelZoo()
    loaded = zoo.load_text_model(model_name, variant, device)
    model = loaded.model
    tokenizer = loaded.tokenizer
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"{model_name}-{variant}-{device}"):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_labels = labels[i:i+BATCH_SIZE]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            batch_labels_tensor = torch.tensor(batch_labels, device=device)
            
            # Forward pass
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Count correct
            correct += (predictions == batch_labels_tensor).sum().item()
            total += len(batch_labels)
    
    accuracy = correct / total if total > 0 else 0.0
    LOGGER.info(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return {
        "task": "text",
        "model": model_name,
        "variant": variant,
        "device": device,
        "exit_policy": "none",
        "threshold": None,
        "accuracy": accuracy,
        "coverage": 1.0,
        "num_samples": total
    }


def evaluate_text_cascade(model_small, model_large, variant, device, threshold, texts, labels):
    """Evaluate 2-stage cascade on SST-2 validation set."""
    LOGGER.info(f"Evaluating cascade {model_small}->{model_large} (τ={threshold}, {variant}, {device})...")
    
    zoo = ModelZoo()
    loaded_small = zoo.load_text_model(model_small, variant, device)
    loaded_large = zoo.load_text_model(model_large, variant, device)
    
    evaluator = CascadeEvaluator(
        model_small=loaded_small.model,
        model_large=loaded_large.model,
        tokenizer=loaded_small.tokenizer,
        threshold=threshold,
        device=device
    )
    
    correct = 0
    total = 0
    stage1_exits = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"cascade-τ{threshold}"):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_labels = labels[i:i+BATCH_SIZE]
            
            # Tokenize
            inputs = evaluator.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            batch_labels_tensor = torch.tensor(batch_labels, device=device)
            
            # Stage 1: Small model
            outputs_small = evaluator.model_small(**inputs)
            probs_small = torch.softmax(outputs_small.logits, dim=-1)
            max_probs, predictions_small = torch.max(probs_small, dim=-1)
            
            # Check confidence
            high_confidence_mask = max_probs >= threshold
            stage1_exits += high_confidence_mask.sum().item()
            
            # Stage 2: Large model for low confidence
            predictions = predictions_small.clone()
            if not high_confidence_mask.all():
                low_conf_indices = (~high_confidence_mask).nonzero(as_tuple=True)[0]
                
                # Re-tokenize only low-confidence samples
                low_conf_texts = [batch_texts[idx] for idx in low_conf_indices.cpu().numpy()]
                inputs_large = evaluator.tokenizer(
                    low_conf_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                inputs_large = {k: v.to(device) for k, v in inputs_large.items()}
                
                outputs_large = evaluator.model_large(**inputs_large)
                predictions_large = torch.argmax(outputs_large.logits, dim=-1)
                
                predictions[low_conf_indices] = predictions_large
            
            # Count correct
            correct += (predictions == batch_labels_tensor).sum().item()
            total += len(batch_labels)
    
    accuracy = correct / total if total > 0 else 0.0
    coverage = stage1_exits / total if total > 0 else 0.0
    
    LOGGER.info(f"  Accuracy: {accuracy:.4f}, Coverage: {coverage:.4f} ({stage1_exits}/{total} early exits)")
    
    return {
        "task": "text",
        "model": f"{model_small}->{model_large}",
        "variant": variant,
        "device": device,
        "exit_policy": "cascade",
        "threshold": threshold,
        "accuracy": accuracy,
        "coverage": coverage,
        "num_samples": total
    }


def evaluate_image_model(model_name, variant, device, dataset):
    """Evaluate image model accuracy on CIFAR-10 test set."""
    LOGGER.info(f"Evaluating {model_name} ({variant}, {device}) on CIFAR-10...")
    
    zoo = ModelZoo()
    loaded = zoo.load_image_model(model_name, variant, device)
    model = loaded.model
    
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"{model_name}-{variant}-{device}"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    LOGGER.info(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return {
        "task": "vision",
        "model": model_name,
        "variant": variant,
        "device": device,
        "exit_policy": "none",
        "threshold": None,
        "accuracy": accuracy,
        "coverage": 1.0,
        "num_samples": total
    }


def main():
    parser = argparse.ArgumentParser(description="Profile accuracy for text and image models")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (fewer samples)")
    args = parser.parse_args()
    
    # Adjust for quick mode
    global MAX_TEXT_SAMPLES, MAX_IMAGE_SAMPLES
    if args.quick:
        MAX_TEXT_SAMPLES = 100
        MAX_IMAGE_SAMPLES = 500
        LOGGER.info("Running in QUICK TEST mode")
    
    LOGGER.info("Starting accuracy profiling with real datasets...")
    
    # Load datasets
    texts, labels = load_sst2_validation()
    cifar10 = load_cifar10_test()
    
    results = []
    
    # Profile text models
    for model_name in TEXT_MODELS:
        for variant in VARIANTS:
            for device in DEVICES:
                # Skip invalid combinations
                if variant == "fp16" and device == "cpu":
                    continue
                if variant == "int8" and device == "cuda":
                    continue
                
                try:
                    result = evaluate_text_model(model_name, variant, device, texts, labels)
                    results.append(result)
                except Exception as e:
                    LOGGER.error(f"Failed to evaluate {model_name} ({variant}, {device}): {e}")
    
    # Profile text cascades
    for variant in VARIANTS:
        for device in DEVICES:
            if variant == "fp16" and device == "cpu":
                continue
            if variant == "int8" and device == "cuda":
                continue
            
            for threshold in CASCADE_THRESHOLDS:
                try:
                    result = evaluate_text_cascade(
                        "minilm", "distilbert", variant, device, threshold, texts, labels
                    )
                    results.append(result)
                except Exception as e:
                    LOGGER.error(f"Failed to evaluate cascade (τ={threshold}, {variant}, {device}): {e}")
    
    # Profile image models
    for model_name in IMAGE_MODELS:
        for variant in VARIANTS:
            for device in DEVICES:
                if variant == "fp16" and device == "cpu":
                    continue
                if variant == "int8" and device == "cuda":
                    continue
                
                try:
                    result = evaluate_image_model(model_name, variant, device, cifar10)
                    results.append(result)
                except Exception as e:
                    LOGGER.error(f"Failed to evaluate {model_name} ({variant}, {device}): {e}")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path("results/accuracy_profiles.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, output_path)
    
    LOGGER.info(f"Accuracy profiling complete. Results saved to {output_path}")
    LOGGER.info(f"Total configurations profiled: {len(results)}")
    
    # Summary statistics
    LOGGER.info("\nSummary:")
    LOGGER.info(f"  Text models: {df[df['task'] == 'text'].shape[0]} configurations")
    LOGGER.info(f"  Image models: {df[df['task'] == 'vision'].shape[0]} configurations")
    LOGGER.info(f"  Mean accuracy (text): {df[df['task'] == 'text']['accuracy'].mean():.4f}")
    LOGGER.info(f"  Mean accuracy (vision): {df[df['task'] == 'vision']['accuracy'].mean():.4f}")


if __name__ == "__main__":
    main()

