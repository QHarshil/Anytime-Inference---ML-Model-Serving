"""Real inference evaluation module.

Replaces simulated latency sampling with actual batched inference runs.
Caches raw measurements and exposes configuration-level variance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..utils.logger import get_logger

LOGGER = get_logger("evaluation.real_inference")


@dataclass
class InferenceResult:
    """Container for real inference measurements."""
    
    config_id: str
    task: str
    model: str
    variant: str
    device: str
    batch_size: int
    
    # Raw measurements (all runs)
    latencies_ms: List[float]
    accuracies: List[float]
    
    # Statistics
    latency_mean: float
    latency_std: float
    latency_p50: float
    latency_p95: float
    
    accuracy_mean: float
    accuracy_std: float
    
    num_runs: int
    num_samples_per_run: int


class RealInferenceEvaluator:
    """Evaluates configurations using real batched inference."""
    
    def __init__(self, cache_dir: Path = Path("results/inference_cache")):
        """Initialize evaluator with caching."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "inference_measurements.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cached measurements."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save measurements to cache."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _get_cache_key(self, config: Dict, task: str, seed: int) -> str:
        """Generate cache key for configuration."""
        return f"{task}_{config['model']}_{config['variant']}_{config['device']}_{config['batch_size']}_seed{seed}"
    
    def evaluate_config(
        self,
        model,
        tokenizer_or_transform,
        inputs,
        labels,
        config: Dict,
        task: str,
        num_runs: int = 5,
        num_samples: int = 100,
        seed: int = 42,
        use_cache: bool = True
    ) -> InferenceResult:
        """Evaluate a configuration with real inference.
        
        Args:
            model: Model to evaluate
            tokenizer_or_transform: Tokenizer (text) or transform (vision)
            inputs: Input data (texts or images)
            labels: Ground truth labels
            config: Configuration dict with model, variant, device, batch_size
            task: Task name ('text' or 'vision')
            num_runs: Number of independent runs
            num_samples: Number of samples per run
            seed: Random seed for reproducibility
            use_cache: Whether to use cached results
        
        Returns:
            InferenceResult with raw measurements and statistics
        """
        cache_key = self._get_cache_key(config, task, seed)
        
        # Check cache
        if use_cache and cache_key in self.cache:
            LOGGER.info(f"Using cached result for {cache_key}")
            cached = self.cache[cache_key]
            return InferenceResult(**cached)
        
        LOGGER.info(f"Running real inference for {cache_key}")
        
        device = config['device']
        batch_size = config['batch_size']
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Prepare data
        np.random.seed(seed)
        indices = np.random.choice(len(inputs), min(num_samples, len(inputs)), replace=False)
        
        all_latencies = []
        all_accuracies = []
        
        for run_idx in range(num_runs):
            run_latencies = []
            run_predictions = []
            run_labels = []
            
            # Process in batches
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_inputs = [inputs[idx] for idx in batch_indices]
                batch_labels = [labels[idx] for idx in batch_indices]
                
                # Prepare batch
                if task == 'text':
                    encoded = tokenizer_or_transform(
                        batch_inputs,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    ).to(device)
                else:  # vision
                    batch_tensors = [tokenizer_or_transform(img) for img in batch_inputs]
                    encoded = torch.stack(batch_tensors).to(device)
                
                # Warmup (first batch only)
                if i == 0 and run_idx == 0:
                    with torch.no_grad():
                        for _ in range(3):
                            if task == 'text':
                                _ = model(**encoded)
                            else:
                                _ = model(encoded)
                
                # Measure latency
                torch.cuda.synchronize() if device == 'cuda' else None
                start = time.perf_counter()
                
                with torch.no_grad():
                    if task == 'text':
                        outputs = model(**encoded)
                    else:
                        outputs = model(encoded)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    predictions = torch.argmax(logits, dim=-1)
                
                torch.cuda.synchronize() if device == 'cuda' else None
                end = time.perf_counter()
                
                # Record per-sample latency
                batch_latency_ms = (end - start) * 1000.0 / len(batch_inputs)
                run_latencies.extend([batch_latency_ms] * len(batch_inputs))
                
                # Record predictions
                run_predictions.extend(predictions.cpu().numpy().tolist())
                run_labels.extend(batch_labels)
            
            # Compute accuracy for this run
            run_accuracy = np.mean(np.array(run_predictions) == np.array(run_labels))
            
            all_latencies.append(np.mean(run_latencies))
            all_accuracies.append(run_accuracy)
        
        # Compute statistics
        result = InferenceResult(
            config_id=cache_key,
            task=task,
            model=config['model'],
            variant=config['variant'],
            device=device,
            batch_size=batch_size,
            latencies_ms=all_latencies,
            accuracies=all_accuracies,
            latency_mean=float(np.mean(all_latencies)),
            latency_std=float(np.std(all_latencies)),
            latency_p50=float(np.percentile(all_latencies, 50)),
            latency_p95=float(np.percentile(all_latencies, 95)),
            accuracy_mean=float(np.mean(all_accuracies)),
            accuracy_std=float(np.std(all_accuracies)),
            num_runs=num_runs,
            num_samples_per_run=num_samples
        )
        
        # Cache result
        self.cache[cache_key] = {
            'config_id': result.config_id,
            'task': result.task,
            'model': result.model,
            'variant': result.variant,
            'device': result.device,
            'batch_size': result.batch_size,
            'latencies_ms': result.latencies_ms,
            'accuracies': result.accuracies,
            'latency_mean': result.latency_mean,
            'latency_std': result.latency_std,
            'latency_p50': result.latency_p50,
            'latency_p95': result.latency_p95,
            'accuracy_mean': result.accuracy_mean,
            'accuracy_std': result.accuracy_std,
            'num_runs': result.num_runs,
            'num_samples_per_run': result.num_samples_per_run
        }
        self._save_cache()
        
        return result


class CascadeInferenceEvaluator:
    """Evaluates two-stage cascade with real end-to-end timing."""
    
    def __init__(self, cache_dir: Path = Path("results/inference_cache")):
        """Initialize cascade evaluator."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "cascade_measurements.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cached measurements."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save measurements to cache."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def evaluate_cascade(
        self,
        small_model,
        large_model,
        tokenizer_or_transform,
        inputs,
        labels,
        threshold: float,
        task: str,
        device: str = 'cpu',
        num_runs: int = 5,
        num_samples: int = 100,
        seed: int = 42,
        use_cache: bool = True
    ) -> Dict:
        """Evaluate two-stage cascade with real timing.
        
        Measures:
        - Stage 1 latency (small model)
        - Stage 2 latency (large model, when triggered)
        - End-to-end latency per sample
        - Coverage (fraction handled by stage 1)
        - Accuracy
        
        Args:
            small_model: Fast model for stage 1
            large_model: Accurate model for stage 2
            tokenizer_or_transform: Tokenizer or transform
            inputs: Input data
            labels: Ground truth labels
            threshold: Confidence threshold for early exit
            task: 'text' or 'vision'
            device: 'cpu' or 'cuda'
            num_runs: Number of independent runs
            num_samples: Samples per run
            seed: Random seed
            use_cache: Use cached results
        
        Returns:
            Dict with measurements and statistics
        """
        cache_key = f"cascade_{task}_{threshold}_{device}_seed{seed}"
        
        if use_cache and cache_key in self.cache:
            LOGGER.info(f"Using cached cascade result for {cache_key}")
            return self.cache[cache_key]
        
        LOGGER.info(f"Running real cascade inference for {cache_key}")
        
        small_model = small_model.to(device)
        large_model = large_model.to(device)
        small_model.eval()
        large_model.eval()
        
        np.random.seed(seed)
        indices = np.random.choice(len(inputs), min(num_samples, len(inputs)), replace=False)
        
        all_latencies = []
        all_stage1_latencies = []
        all_stage2_latencies = []
        all_coverages = []
        all_accuracies = []
        
        for run_idx in range(num_runs):
            run_latencies = []
            run_stage1_latencies = []
            run_stage2_latencies = []
            run_predictions = []
            run_labels = []
            stage1_count = 0
            
            for idx in indices:
                input_data = inputs[idx]
                label = labels[idx]
                
                # Prepare input
                if task == 'text':
                    encoded = tokenizer_or_transform(
                        [input_data],
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    ).to(device)
                else:
                    img_tensor = tokenizer_or_transform(input_data).unsqueeze(0).to(device)
                    encoded = img_tensor
                
                # Stage 1: Small model
                torch.cuda.synchronize() if device == 'cuda' else None
                start_stage1 = time.perf_counter()
                
                with torch.no_grad():
                    if task == 'text':
                        outputs1 = small_model(**encoded)
                    else:
                        outputs1 = small_model(encoded)
                    logits1 = outputs1.logits if hasattr(outputs1, 'logits') else outputs1
                    probs1 = torch.softmax(logits1, dim=-1)
                    conf1, pred1 = torch.max(probs1, dim=-1)
                
                torch.cuda.synchronize() if device == 'cuda' else None
                end_stage1 = time.perf_counter()
                stage1_latency = (end_stage1 - start_stage1) * 1000.0
                
                # Check if stage 2 needed
                if conf1.item() >= threshold:
                    # Early exit
                    prediction = pred1.item()
                    total_latency = stage1_latency
                    stage2_latency = 0.0
                    stage1_count += 1
                else:
                    # Stage 2: Large model
                    torch.cuda.synchronize() if device == 'cuda' else None
                    start_stage2 = time.perf_counter()
                    
                    with torch.no_grad():
                        if task == 'text':
                            outputs2 = large_model(**encoded)
                        else:
                            outputs2 = large_model(encoded)
                        logits2 = outputs2.logits if hasattr(outputs2, 'logits') else outputs2
                        pred2 = torch.argmax(logits2, dim=-1)
                    
                    torch.cuda.synchronize() if device == 'cuda' else None
                    end_stage2 = time.perf_counter()
                    stage2_latency = (end_stage2 - start_stage2) * 1000.0
                    
                    prediction = pred2.item()
                    total_latency = stage1_latency + stage2_latency
                
                run_latencies.append(total_latency)
                run_stage1_latencies.append(stage1_latency)
                run_stage2_latencies.append(stage2_latency)
                run_predictions.append(prediction)
                run_labels.append(label)
            
            # Compute metrics for this run
            run_accuracy = np.mean(np.array(run_predictions) == np.array(run_labels))
            run_coverage = stage1_count / len(indices)
            
            all_latencies.append(np.mean(run_latencies))
            all_stage1_latencies.append(np.mean(run_stage1_latencies))
            all_stage2_latencies.append(np.mean([l for l in run_stage2_latencies if l > 0]))
            all_coverages.append(run_coverage)
            all_accuracies.append(run_accuracy)
        
        # Aggregate statistics
        result = {
            'cache_key': cache_key,
            'task': task,
            'threshold': threshold,
            'device': device,
            'latencies_ms': all_latencies,
            'stage1_latencies_ms': all_stage1_latencies,
            'stage2_latencies_ms': all_stage2_latencies,
            'coverages': all_coverages,
            'accuracies': all_accuracies,
            'latency_mean': float(np.mean(all_latencies)),
            'latency_std': float(np.std(all_latencies)),
            'latency_p50': float(np.percentile(all_latencies, 50)),
            'latency_p95': float(np.percentile(all_latencies, 95)),
            'coverage_mean': float(np.mean(all_coverages)),
            'coverage_std': float(np.std(all_coverages)),
            'accuracy_mean': float(np.mean(all_accuracies)),
            'accuracy_std': float(np.std(all_accuracies)),
            'num_runs': num_runs,
            'num_samples_per_run': num_samples
        }
        
        # Cache
        self.cache[cache_key] = result
        self._save_cache()
        
        return result

