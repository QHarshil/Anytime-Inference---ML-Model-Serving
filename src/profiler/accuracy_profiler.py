"""Accuracy profiling utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

from ..models.model_zoo import ModelZoo
from ..utils.logger import get_logger
from .profiler_utils import compute_accuracy

LOGGER = get_logger("profiler.accuracy")


@dataclass
class AccuracyProfile:
    task: str
    model: str
    variant: str
    device: str
    accuracy: float
    coverage: float


class AccuracyProfiler:
    """Compute accuracy and cascade coverage for models."""

    def __init__(self, model_zoo: ModelZoo) -> None:
        self.model_zoo = model_zoo

    def profile_text(
        self,
        model: str,
        variant: str,
        device: str,
        texts: List[str],
        labels: Iterable[int],
    ) -> AccuracyProfile:
        LOGGER.info("Profiling text accuracy for %s (%s)", model, variant)
        preds, _ = self.model_zoo.predict_text(model, variant, device, texts)
        accuracy = compute_accuracy(preds, labels)
        return AccuracyProfile(
            task="text",
            model=model,
            variant=variant,
            device=device,
            accuracy=accuracy,
            coverage=1.0,
        )

    def to_dataframe(self, profiles: Iterable[AccuracyProfile]) -> pd.DataFrame:
        return pd.DataFrame([profile.__dict__ for profile in profiles])
