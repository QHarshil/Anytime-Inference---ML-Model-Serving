"""Two-stage cascade inference helpers.

``torch`` is imported inside the functions that run inference rather than at
module scope, so the pure-Python helpers here stay importable on installs
without it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def cascade_predict_text(
    small_model: torch.nn.Module,
    large_model: torch.nn.Module,
    tokenizer,
    texts: Iterable[str],
    *,
    threshold: float = 0.9,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

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
    images: Iterable,
    *,
    threshold: float = 0.9,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

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
        high_res_inputs = torch.stack([transform(images[i]) for i in low_conf_indices.tolist()]).to(
            device
        )
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
    early = np.asarray(list(early_exits))
    if early.size == 0:
        return 0.0
    return float(np.mean(early))


class CascadeEvaluator:
    """Two-stage cascade evaluator for text or vision inputs."""

    def __init__(
        self,
        small_model: torch.nn.Module,
        large_model: torch.nn.Module,
        tokenizer_or_transform,
        *,
        task: str = "text",
        device: str = "cpu",
    ) -> None:
        if task not in ("text", "vision"):
            raise ValueError(f"Unknown task: {task}")
        self.small_model = small_model.to(device).eval()
        self.large_model = large_model.to(device).eval()
        self.tokenizer_or_transform = tokenizer_or_transform
        self.task = task
        self.device = device

    def evaluate(self, inputs, labels, *, threshold: float = 0.9) -> dict:
        if self.task == "text":
            preds, confidences, early_exits = cascade_predict_text(
                self.small_model,
                self.large_model,
                self.tokenizer_or_transform,
                inputs,
                threshold=threshold,
                device=self.device,
            )
        else:
            preds, confidences, early_exits = cascade_predict_image(
                self.small_model,
                self.large_model,
                self.tokenizer_or_transform,
                inputs,
                threshold=threshold,
                device=self.device,
            )

        labels_array = np.asarray(list(labels))
        accuracy = float(np.mean(preds == labels_array))
        coverage = cascade_coverage(early_exits)
        return {
            "accuracy": accuracy,
            "coverage": coverage,
            "predictions": preds,
            "confidences": confidences,
            "early_exits": early_exits,
        }
