"""Accuracy profiling for text and image models on SST-2 and CIFAR-10."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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

TEXT_MODELS = ["distilbert", "minilm"]
IMAGE_MODELS = ["mobilenetv2", "resnet18"]
VARIANTS = ["fp32", "fp16", "int8"]
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
CASCADE_THRESHOLDS = [0.7, 0.8, 0.9, 0.95]
BATCH_SIZE = 32
MAX_TEXT_SAMPLES = 872
MAX_IMAGE_SAMPLES = 10000


def load_sst2_validation():
    LOGGER.info("Loading SST-2 validation set")
    dataset = load_dataset("glue", "sst2", split="validation")
    texts = [example["sentence"] for example in dataset]
    labels = [example["label"] for example in dataset]
    return texts, labels


def load_cifar10_test():
    LOGGER.info("Loading CIFAR-10 test set")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)


def evaluate_text_model(model_name, variant, device, texts, labels):
    LOGGER.info("Evaluating %s (%s, %s) on SST-2", model_name, variant, device)
    zoo = ModelZoo()
    loaded = zoo.load_text_model(model_name, variant, device)
    model = loaded.model
    tokenizer = loaded.tokenizer

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"{model_name}-{variant}-{device}"):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_labels = labels[i:i + BATCH_SIZE]
            inputs = tokenizer(
                batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == torch.tensor(batch_labels, device=device)).sum().item()
            total += len(batch_labels)

    accuracy = correct / total if total else 0.0
    LOGGER.info("  accuracy=%.4f (%d/%d)", accuracy, correct, total)
    return {
        "task": "text",
        "model": model_name,
        "variant": variant,
        "device": device,
        "exit_policy": "none",
        "threshold": None,
        "accuracy": accuracy,
        "coverage": 1.0,
        "num_samples": total,
    }


def evaluate_text_cascade(model_small, model_large, variant, device, threshold, texts, labels):
    LOGGER.info(
        "Evaluating cascade %s->%s (threshold=%.2f, %s, %s)",
        model_small, model_large, threshold, variant, device,
    )
    zoo = ModelZoo()
    loaded_small = zoo.load_text_model(model_small, variant, device)
    loaded_large = zoo.load_text_model(model_large, variant, device)

    evaluator = CascadeEvaluator(
        loaded_small.model,
        loaded_large.model,
        loaded_small.tokenizer,
        task="text",
        device=device,
    )

    correct = 0
    total = 0
    stage1_exits = 0
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"cascade-t{threshold}"):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_labels = labels[i:i + BATCH_SIZE]
        result = evaluator.evaluate(batch_texts, batch_labels, threshold=threshold)
        preds = result["predictions"]
        correct += int((preds == pd.Series(batch_labels).values).sum())
        total += len(batch_labels)
        stage1_exits += int(result["early_exits"].sum())

    accuracy = correct / total if total else 0.0
    coverage = stage1_exits / total if total else 0.0
    LOGGER.info("  accuracy=%.4f coverage=%.4f (%d/%d early exits)",
                accuracy, coverage, stage1_exits, total)
    return {
        "task": "text",
        "model": f"{model_small}->{model_large}",
        "variant": variant,
        "device": device,
        "exit_policy": "cascade",
        "threshold": threshold,
        "accuracy": accuracy,
        "coverage": coverage,
        "num_samples": total,
    }


def evaluate_image_model(model_name, variant, device, dataset):
    LOGGER.info("Evaluating %s (%s, %s) on CIFAR-10", model_name, variant, device)
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

    accuracy = correct / total if total else 0.0
    LOGGER.info("  accuracy=%.4f (%d/%d)", accuracy, correct, total)
    return {
        "task": "vision",
        "model": model_name,
        "variant": variant,
        "device": device,
        "exit_policy": "none",
        "threshold": None,
        "accuracy": accuracy,
        "coverage": 1.0,
        "num_samples": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile accuracy for text and image models")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (fewer samples)")
    args = parser.parse_args()

    global MAX_TEXT_SAMPLES, MAX_IMAGE_SAMPLES
    if args.quick:
        MAX_TEXT_SAMPLES = 100
        MAX_IMAGE_SAMPLES = 500
        LOGGER.info("Running in quick-test mode")

    texts, labels = load_sst2_validation()
    texts = texts[:MAX_TEXT_SAMPLES]
    labels = labels[:MAX_TEXT_SAMPLES]
    cifar10 = load_cifar10_test()
    if args.quick:
        cifar10 = torch.utils.data.Subset(cifar10, range(MAX_IMAGE_SAMPLES))

    results = []

    for model_name in TEXT_MODELS:
        for variant in VARIANTS:
            for device in DEVICES:
                if variant == "fp16" and device == "cpu":
                    continue
                if variant == "int8" and device == "cuda":
                    continue
                try:
                    results.append(evaluate_text_model(model_name, variant, device, texts, labels))
                except Exception as exc:
                    LOGGER.error("Failed %s %s %s: %s", model_name, variant, device, exc)

    for variant in VARIANTS:
        for device in DEVICES:
            if variant == "fp16" and device == "cpu":
                continue
            if variant == "int8" and device == "cuda":
                continue
            for threshold in CASCADE_THRESHOLDS:
                try:
                    results.append(
                        evaluate_text_cascade("minilm", "distilbert", variant, device, threshold, texts, labels)
                    )
                except Exception as exc:
                    LOGGER.error("Failed cascade t=%.2f %s %s: %s", threshold, variant, device, exc)

    for model_name in IMAGE_MODELS:
        for variant in VARIANTS:
            for device in DEVICES:
                if variant == "fp16" and device == "cpu":
                    continue
                if variant == "int8" and device == "cuda":
                    continue
                try:
                    results.append(evaluate_image_model(model_name, variant, device, cifar10))
                except Exception as exc:
                    LOGGER.error("Failed %s %s %s: %s", model_name, variant, device, exc)

    df = pd.DataFrame(results)
    output_path = Path("results/accuracy_profiles.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, output_path)
    LOGGER.info("Wrote %d rows to %s", len(df), output_path)


if __name__ == "__main__":
    main()
