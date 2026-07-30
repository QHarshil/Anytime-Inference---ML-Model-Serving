"""Model loading utilities for the Anytime Inference Planner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils.logger import get_logger

if TYPE_CHECKING:
    import PIL.Image
    import torch
    import transformers

LOGGER = get_logger("models.model_zoo")


@dataclass
class LoadedTextModel:
    model: torch.nn.Module
    tokenizer: transformers.PreTrainedTokenizer


@dataclass
class LoadedImageModel:
    model: torch.nn.Module
    transform: Callable


class ModelZoo:
    """Unified loader for text and image models with basic quantisation."""

    TEXT_MODELS: dict[str, str] = {
        "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
        "minilm": "philschmid/MiniLM-L6-H384-uncased-sst2",
    }

    IMAGE_MODELS: dict[str, str] = {
        "mobilenetv2": "mobilenet_v2",
        "resnet18": "resnet18",
    }

    def __init__(self) -> None:
        self._cache: dict[str, tuple[object, object]] = {}

    @staticmethod
    def _require_torch() -> ModuleType:
        import importlib

        return importlib.import_module("torch")

    @staticmethod
    def _require_transformers() -> tuple[Any, Any]:
        import importlib

        transformers = importlib.import_module("transformers")
        return (
            transformers.AutoModelForSequenceClassification,
            transformers.AutoTokenizer,
        )

    @staticmethod
    def _require_torchvision() -> tuple[ModuleType, ModuleType]:
        import importlib

        torchvision = importlib.import_module("torchvision")
        transforms = importlib.import_module("torchvision.transforms")
        return torchvision.models, transforms

    def _cache_key(self, prefix: str, *parts: str) -> str:
        return f"{prefix}-{'-'.join(parts)}"

    # ------------------------------------------------------------------
    # Text models
    # ------------------------------------------------------------------
    def load_text_model(self, model_name: str, variant: str, device: str) -> LoadedTextModel:
        """Load a text classification model and tokenizer."""

        if model_name not in self.TEXT_MODELS:
            raise KeyError(f"Unknown text model: {model_name}")

        key = self._cache_key("text", model_name, variant, device)
        if key in self._cache:
            model, tokenizer = self._cache[key]
            return LoadedTextModel(model=model, tokenizer=tokenizer)  # type: ignore[arg-type]

        torch = self._require_torch()
        AutoModelForSequenceClassification, AutoTokenizer = self._require_transformers()
        hf_name = self.TEXT_MODELS[model_name]

        LOGGER.info("Loading text model %s (%s) on %s", hf_name, variant, device)
        model = AutoModelForSequenceClassification.from_pretrained(hf_name)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)

        if variant == "fp16" and device == "cuda":
            model = model.half()
            LOGGER.info("Converted %s to FP16", model_name)
        elif variant == "int8":
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            LOGGER.info("Converted %s to INT8 using dynamic quantisation", model_name)

        model.to(device)
        model.eval()

        self._cache[key] = (model, tokenizer)
        return LoadedTextModel(model=model, tokenizer=tokenizer)

    def predict_text(
        self,
        model_name: str,
        variant: str,
        device: str,
        texts: list[str],
        *,
        batch_size: int = 8,
    ) -> tuple[np.ndarray, float]:
        """Run batched predictions for text inputs."""

        import time

        loaded = self.load_text_model(model_name, variant, device)
        model, tokenizer = loaded.model, loaded.tokenizer
        torch = self._require_torch()

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        if variant == "fp16" and device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        latency_ms = (time.perf_counter() - start) * 1000.0
        return preds, float(latency_ms)

    # ------------------------------------------------------------------
    # Image models
    # ------------------------------------------------------------------
    def load_image_model(self, model_name: str, variant: str, device: str) -> LoadedImageModel:
        """Load an image classification model."""

        if model_name not in self.IMAGE_MODELS:
            raise KeyError(f"Unknown image model: {model_name}")

        key = self._cache_key("image", model_name, variant, device)
        if key in self._cache:
            model, transform = self._cache[key]
            return LoadedImageModel(model=model, transform=transform)  # type: ignore[arg-type]

        torch = self._require_torch()
        models, transforms = self._require_torchvision()
        model_fn = getattr(models, self.IMAGE_MODELS[model_name])

        LOGGER.info("Loading image model %s (%s) on %s", model_name, variant, device)
        model = (
            model_fn(weights="DEFAULT")
            if hasattr(model_fn, "weights")
            else model_fn(pretrained=True)
        )

        if variant == "fp16" and device == "cuda":
            model = model.half()
            LOGGER.info("Converted %s to FP16", model_name)
        elif variant == "int8":
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
            LOGGER.info("Converted %s to INT8 using dynamic quantisation", model_name)

        model.to(device)
        model.eval()

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._cache[key] = (model, transform)
        return LoadedImageModel(model=model, transform=transform)

    def load_model(
        self,
        task: str,
        model_name: str,
        variant: str,
        device: str,
    ):
        """Convenience loader that returns the model and its tokenizer/transform for ``task``."""

        if task == "text":
            text_model = self.load_text_model(model_name, variant, device)
            return text_model.model, text_model.tokenizer
        if task == "vision":
            image_model = self.load_image_model(model_name, variant, device)
            return image_model.model, image_model.transform
        raise ValueError(f"Unknown task: {task}")

    def predict_image(
        self,
        model_name: str,
        variant: str,
        device: str,
        images: list[PIL.Image.Image],
    ) -> tuple[np.ndarray, float]:
        """Run inference on a batch of PIL images."""

        import time

        torch = self._require_torch()

        loaded = self.load_image_model(model_name, variant, device)
        model, transform = loaded.model, loaded.transform

        tensors = torch.stack([transform(img) for img in images]).to(device)
        if variant == "fp16" and device == "cuda":
            tensors = tensors.half()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(tensors)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
        latency_ms = (time.perf_counter() - start) * 1000.0
        return preds, float(latency_ms)


__all__ = ["ModelZoo", "LoadedTextModel", "LoadedImageModel"]
