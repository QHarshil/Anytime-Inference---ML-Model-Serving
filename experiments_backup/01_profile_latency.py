"""
Latency profiling for text and image models.

Uses real data samples for realistic measurements:
- Text: SST-2 validation set samples (variable length sentences)
- Image: CIFAR-10 test set samples (32x32x3 images)

Measurement protocol:
- Warmup: 10 iterations to stabilize GPU/CPU caches
- Measurement: 100 iterations per configuration
- Statistics: p50, p95, mean, std latency in milliseconds
- Throughput: items per second

Output: results/latency_profiles.csv
Schema: task,model,variant,device,batch_size,lat_p50_ms,lat_p95_ms,lat_mean_ms,lat_std_ms,throughput_items_per_sec,num_measurements
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models.model_zoo import ModelZoo
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.profile_latency")

# Configuration
TEXT_MODELS = ["distilbert", "minilm"]
IMAGE_MODELS = ["mobilenetv2", "resnet18"]
VARIANTS = ["fp32", "fp16", "int8"]
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
BATCH_SIZES = [1, 4, 8, 16]
WARMUP_ITERATIONS = 10
MEASUREMENT_ITERATIONS = 100


def load_sst2_samples(n=100):
    """Load real SST-2 samples with variable lengths."""
    LOGGER.info(f"Loading {n} SST-2 samples for latency profiling...")
    dataset = load_dataset("glue", "sst2", split="validation")
    texts = [example["sentence"] for example in dataset.select(range(min(n, len(dataset))))]
    LOGGER.info(f"Loaded {len(texts)} text samples (lengths: {min(len(t.split()) for t in texts)}-{max(len(t.split()) for t in texts)} words)")
    return texts


def load_cifar10_samples(n=100):
    """Load real CIFAR-10 samples."""
    LOGGER.info(f"Loading {n} CIFAR-10 samples for latency profiling...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    samples = [dataset[i][0] for i in range(min(n, len(dataset)))]
    LOGGER.info(f"Loaded {len(samples)} image samples (shape: {samples[0].shape})")
    return samples


def measure_latency_text(model, tokenizer, texts, batch_size, device):
    """Measure latency for text model with warmup and multiple iterations."""
    model.eval()
    
    # Prepare batches
    num_batches = len(texts) // batch_size
    if num_batches == 0:
        num_batches = 1
        batch_size = len(texts)
    
    batches = []
    for i in range(num_batches):
        batch_texts = texts[i*batch_size:(i+1)*batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        batches.append(inputs)
    
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            for inputs in batches:
                _ = model(**inputs)
    
    # Synchronize for accurate timing
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Measurement
    latencies = []
    with torch.no_grad():
        for _ in range(MEASUREMENT_ITERATIONS):
            start = time.perf_counter()
            for inputs in batches:
                _ = model(**inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000 / num_batches)  # ms per batch
    
    latencies = np.array(latencies)
    return {
        "lat_p50_ms": np.percentile(latencies, 50),
        "lat_p95_ms": np.percentile(latencies, 95),
        "lat_mean_ms": np.mean(latencies),
        "lat_std_ms": np.std(latencies),
        "throughput_items_per_sec": (batch_size * 1000) / np.mean(latencies),
        "num_measurements": len(latencies)
    }


def measure_latency_image(model, images, batch_size, device):
    """Measure latency for image model with warmup and multiple iterations."""
    model.eval()
    
    # Prepare batches
    num_batches = len(images) // batch_size
    if num_batches == 0:
        num_batches = 1
        batch_size = len(images)
    
    batches = []
    for i in range(num_batches):
        batch_images = torch.stack(images[i*batch_size:(i+1)*batch_size]).to(device)
        batches.append(batch_images)
    
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            for batch in batches:
                _ = model(batch)
    
    # Synchronize for accurate timing
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Measurement
    latencies = []
    with torch.no_grad():
        for _ in range(MEASUREMENT_ITERATIONS):
            start = time.perf_counter()
            for batch in batches:
                _ = model(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000 / num_batches)  # ms per batch
    
    latencies = np.array(latencies)
    return {
        "lat_p50_ms": np.percentile(latencies, 50),
        "lat_p95_ms": np.percentile(latencies, 95),
        "lat_mean_ms": np.mean(latencies),
        "lat_std_ms": np.std(latencies),
        "throughput_items_per_sec": (batch_size * 1000) / np.mean(latencies),
        "num_measurements": len(latencies)
    }


def profile_text_model(model_name, variant, device, batch_size, texts):
    """Profile latency for a text model configuration."""
    LOGGER.info(f"Profiling {model_name} ({variant}, {device}, batch={batch_size})...")
    
    zoo = ModelZoo()
    loaded = zoo.load_text_model(model_name, variant, device)
    
    measurements = measure_latency_text(loaded.model, loaded.tokenizer, texts, batch_size, device)
    
    result = {
        "task": "text",
        "model": model_name,
        "variant": variant,
        "device": device,
        "batch_size": batch_size,
        **measurements
    }
    
    LOGGER.info(f"  p50={measurements['lat_p50_ms']:.2f}ms, p95={measurements['lat_p95_ms']:.2f}ms, throughput={measurements['throughput_items_per_sec']:.1f} items/s")
    
    return result


def profile_image_model(model_name, variant, device, batch_size, images):
    """Profile latency for an image model configuration."""
    LOGGER.info(f"Profiling {model_name} ({variant}, {device}, batch={batch_size})...")
    
    zoo = ModelZoo()
    loaded = zoo.load_image_model(model_name, variant, device)
    
    measurements = measure_latency_image(loaded.model, images, batch_size, device)
    
    result = {
        "task": "vision",
        "model": model_name,
        "variant": variant,
        "device": device,
        "batch_size": batch_size,
        **measurements
    }
    
    LOGGER.info(f"  p50={measurements['lat_p50_ms']:.2f}ms, p95={measurements['lat_p95_ms']:.2f}ms, throughput={measurements['throughput_items_per_sec']:.1f} items/s")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Profile latency for text and image models")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (fewer iterations and configs)")
    args = parser.parse_args()
    
    # Adjust for quick mode
    global WARMUP_ITERATIONS, MEASUREMENT_ITERATIONS, BATCH_SIZES
    if args.quick:
        WARMUP_ITERATIONS = 2
        MEASUREMENT_ITERATIONS = 10
        BATCH_SIZES = [1, 8]
        LOGGER.info("Running in QUICK TEST mode")
    
    LOGGER.info("Starting latency profiling with real datasets...")
    LOGGER.info(f"Warmup iterations: {WARMUP_ITERATIONS}")
    LOGGER.info(f"Measurement iterations: {MEASUREMENT_ITERATIONS}")
    
    # Load real data samples
    texts = load_sst2_samples(n=100)
    images = load_cifar10_samples(n=100)
    
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
                
                for batch_size in BATCH_SIZES:
                    try:
                        result = profile_text_model(model_name, variant, device, batch_size, texts)
                        results.append(result)
                    except Exception as e:
                        LOGGER.error(f"Failed to profile {model_name} ({variant}, {device}, batch={batch_size}): {e}")
    
    # Profile image models
    for model_name in IMAGE_MODELS:
        for variant in VARIANTS:
            for device in DEVICES:
                if variant == "fp16" and device == "cpu":
                    continue
                if variant == "int8" and device == "cuda":
                    continue
                
                for batch_size in BATCH_SIZES:
                    try:
                        result = profile_image_model(model_name, variant, device, batch_size, images)
                        results.append(result)
                    except Exception as e:
                        LOGGER.error(f"Failed to profile {model_name} ({variant}, {device}, batch={batch_size}): {e}")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path("results/latency_profiles.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, output_path)
    
    LOGGER.info(f"Latency profiling complete. Results saved to {output_path}")
    LOGGER.info(f"Total configurations profiled: {len(results)}")
    
    # Summary statistics
    LOGGER.info("\nSummary:")
    LOGGER.info(f"  Text models: {df[df['task'] == 'text'].shape[0]} configurations")
    LOGGER.info(f"  Image models: {df[df['task'] == 'vision'].shape[0]} configurations")
    LOGGER.info(f"  Latency range (text): {df[df['task'] == 'text']['lat_p50_ms'].min():.2f}-{df[df['task'] == 'text']['lat_p50_ms'].max():.2f} ms")
    LOGGER.info(f"  Latency range (vision): {df[df['task'] == 'vision']['lat_p50_ms'].min():.2f}-{df[df['task'] == 'vision']['lat_p50_ms'].max():.2f} ms")


if __name__ == "__main__":
    main()

