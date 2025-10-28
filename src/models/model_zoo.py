"""Model loading utilities for the Anytime Inference Planner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..utils.logger import get_logger

LOGGER = get_logger("models.model_zoo")


@dataclass
class LoadedTextModel:
    model: "torch.nn.Module"
    tokenizer: "transformers.PreTrainedTokenizer"


@dataclass
class LoadedImageModel:
    model: "torch.nn.Module"
    transform: "callable"


class ModelZoo:
    """Unified loader for text and image models with basic quantisation."""

    TEXT_MODELS: Dict[str, str] = {
        "distilbert": "distilbert-base-uncased",
        "minilm": "microsoft/MiniLM-L12-H384-uncased",
    }

    IMAGE_MODELS: Dict[str, str] = {
        "mobilenetv2": "mobilenet_v2",
        "resnet18": "resnet18",
    }

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[object, object]] = {}

    @staticmethod
    def _require_torch() -> "torch":
        import importlib

        torch = importlib.import_module("torch")
        return torch  # type: ignore[return-value]

    @staticmethod
    def _require_transformers() -> Tuple[object, object]:
        import importlib

        transformers = importlib.import_module("transformers")
        AutoModelForSequenceClassification = getattr(
            transformers, "AutoModelForSequenceClassification"
        )
        AutoTokenizer = getattr(transformers, "AutoTokenizer")
        return AutoModelForSequenceClassification, AutoTokenizer

    @staticmethod
    def _require_torchvision() -> Tuple[object, object]:
        import importlib

        torchvision = importlib.import_module("torchvision")
        models = getattr(torchvision, "models")
        transforms = importlib.import_module("torchvision.transforms")
        return models, transforms

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
        model = AutoModelForSequenceClassification.from_pretrained(hf_name, num_labels=2)
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
        texts: List[str],
        *,
        batch_size: int = 8,
    ) -> Tuple[np.ndarray, float]:
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
        model = model_fn(weights="DEFAULT") if hasattr(model_fn, "weights") else model_fn(pretrained=True)

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

    def predict_image(
        self,
        model_name: str,
        variant: str,
        device: str,
        images: List["PIL.Image.Image"],
    ) -> Tuple[np.ndarray, float]:
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
