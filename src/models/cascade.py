"""Two-stage cascade inference helpers."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch


def cascade_predict_text(
    small_model: torch.nn.Module,
    large_model: torch.nn.Module,
    tokenizer: "transformers.PreTrainedTokenizer",
    texts: Iterable[str],
    *,
    threshold: float = 0.9,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform two-stage cascade prediction for text inputs."""

    texts = list(texts)
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs_small = small_model(**inputs)
        probs_small = torch.softmax(outputs_small.logits, dim=-1)
        confidences_small, preds_small = torch.max(probs_small, dim=-1)

    early_exits = confidences_small >= threshold
    preds = preds_small.clone()
    confidences = confidences_small.clone()

    low_conf_indices = (~early_exits).nonzero(as_tuple=True)[0]
    if len(low_conf_indices) > 0:
        low_conf_texts = [texts[i] for i in low_conf_indices.tolist()]
        inputs_large = tokenizer(
            low_conf_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs_large = large_model(**inputs_large)
            probs_large = torch.softmax(outputs_large.logits, dim=-1)
            confidences_large, preds_large = torch.max(probs_large, dim=-1)
        preds[low_conf_indices] = preds_large
        confidences[low_conf_indices] = confidences_large

    return (
        preds.cpu().numpy(),
        confidences.cpu().numpy(),
        early_exits.cpu().numpy(),
    )


def cascade_predict_image(
    small_model: torch.nn.Module,
    large_model: torch.nn.Module,
    transform,
    images: Iterable["PIL.Image.Image"],
    *,
    threshold: float = 0.9,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform two-stage cascade for image classification."""

    images = list(images)
    tensors = torch.stack([transform(img) for img in images]).to(device)

    with torch.no_grad():
        outputs_small = small_model(tensors)
        probs_small = torch.softmax(outputs_small, dim=-1)
        confidences_small, preds_small = torch.max(probs_small, dim=-1)

    early_exits = confidences_small >= threshold
    preds = preds_small.clone()
    confidences = confidences_small.clone()

    low_conf_indices = (~early_exits).nonzero(as_tuple=True)[0]
    if len(low_conf_indices) > 0:
        high_res_inputs = torch.stack([transform(images[i]) for i in low_conf_indices.tolist()]).to(device)
        with torch.no_grad():
            outputs_large = large_model(high_res_inputs)
            probs_large = torch.softmax(outputs_large, dim=-1)
            confidences_large, preds_large = torch.max(probs_large, dim=-1)
        preds[low_conf_indices] = preds_large
        confidences[low_conf_indices] = confidences_large

    return (
        preds.cpu().numpy(),
        confidences.cpu().numpy(),
        early_exits.cpu().numpy(),
    )


def cascade_coverage(early_exits: Iterable[bool]) -> float:
    """Compute cascade coverage (fraction of requests that exit early)."""

    early = np.asarray(list(early_exits))
    if early.size == 0:
        return 0.0
    return float(np.mean(early))




class CascadeEvaluator:
    """Evaluator for 2-stage cascade configurations."""
    
    def __init__(self, small_model, large_model, tokenizer_or_transform, task='text', device='cpu'):
        """
        Initialize cascade evaluator.
        
        Args:
            small_model: Small/fast model
            large_model: Large/accurate model
            tokenizer_or_transform: Tokenizer (text) or transform (vision)
            task: 'text' or 'vision'
            device: 'cpu' or 'cuda'
        """
        self.small_model = small_model
        self.large_model = large_model
        self.tokenizer_or_transform = tokenizer_or_transform
        self.task = task
        self.device = device
        
        # Move models to device
        self.small_model.to(device)
        self.large_model.to(device)
        self.small_model.eval()
        self.large_model.eval()
    
    def evaluate(self, inputs, labels, threshold=0.9):
        """
        Evaluate cascade on a dataset.
        
        Args:
            inputs: List of texts (text) or images (vision)
            labels: Ground truth labels
            threshold: Confidence threshold for early exit
        
        Returns:
            dict with accuracy, coverage, predictions, confidences, early_exits
        """
        if self.task == 'text':
            preds, confidences, early_exits = cascade_predict_text(
                self.small_model,
                self.large_model,
                self.tokenizer_or_transform,
                inputs,
                threshold=threshold,
                device=self.device
            )
        elif self.task == 'vision':
            preds, confidences, early_exits = cascade_predict_image(
                self.small_model,
                self.large_model,
                self.tokenizer_or_transform,
                inputs,
                threshold=threshold,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        labels_array = np.array(labels)
        accuracy = float(np.mean(preds == labels_array))
        coverage = cascade_coverage(early_exits)
        
        return {
            'accuracy': accuracy,
            'coverage': coverage,
            'predictions': preds,
            'confidences': confidences,
            'early_exits': early_exits
        }

