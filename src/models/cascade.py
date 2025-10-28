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
