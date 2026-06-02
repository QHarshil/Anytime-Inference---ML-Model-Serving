"""Export FP32 and INT8 variants of the planner models to ONNX.

Usage:
    python scripts/export_onnx.py --task text --output-dir models/onnx
    python scripts/export_onnx.py --task vision --output-dir models/onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.utils.logger import get_logger

LOGGER = get_logger("scripts.export_onnx")


def _export_text(output_dir: Path) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    LOGGER.info("Loading %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    fp32 = AutoModelForSequenceClassification.from_pretrained(model_id).eval()

    sample = tokenizer("the planner exports models to onnx", return_tensors="pt",
                       padding="max_length", truncation=True, max_length=128)
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    fp32_path = output_dir / "text_fp32.onnx"
    LOGGER.info("Exporting FP32 -> %s", fp32_path)
    torch.onnx.export(
        fp32,
        (input_ids, attention_mask),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    int8_path = output_dir / "text_int8.onnx"
    LOGGER.info("Quantising -> %s", int8_path)
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )


def _export_vision(output_dir: Path) -> None:
    from torchvision import models
    from torchvision.models import MobileNet_V2_Weights

    LOGGER.info("Loading MobileNetV2")
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval()
    dummy = torch.randn(1, 3, 224, 224)

    fp32_path = output_dir / "vision_fp32.onnx"
    LOGGER.info("Exporting FP32 -> %s", fp32_path)
    torch.onnx.export(
        model,
        dummy,
        fp32_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
        do_constant_folding=True,
    )

    int8_path = output_dir / "vision_int8.onnx"
    LOGGER.info("Quantising -> %s", int8_path)
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("text", "vision"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/onnx"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "text":
        _export_text(args.output_dir)
    else:
        _export_vision(args.output_dir)
    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
