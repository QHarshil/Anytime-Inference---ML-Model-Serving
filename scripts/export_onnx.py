"""Export FP32 and INT8 variants of the planner models to ONNX.

Usage:
    python scripts/export_onnx.py --task text --output-dir models/onnx
    python scripts/export_onnx.py --task vision --output-dir models/onnx
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import torch

from anytime_serving.utils.logger import get_logger

LOGGER = get_logger("scripts.export_onnx")


# Text candidates, ordered from most to least capacity. Both are genuinely
# fine-tuned on SST-2, so accuracy measured on that split is meaningful.
TEXT_MODELS = {
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "minilm": "philschmid/MiniLM-L6-H384-uncased-sst2",
}


def _quantization_target() -> str:
    """Pick the quantisation target matching the host instruction set.

    Quantising for the wrong architecture is not merely suboptimal: the resulting
    INT8 operators fall back to reference kernels and run *slower* than FP32.
    Quantising DistilBERT for avx512_vnni and serving it on arm64 measured 1.22x
    slower than the FP32 graph it replaced.
    """
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    if machine in {"ppc64le"}:
        return "ppc64le"
    # x86-64. VNNI gives the best INT8 throughput where present; avx2 is the
    # portable fallback that still uses real integer kernels.
    try:
        with open("/proc/cpuinfo") as handle:
            flags = handle.read()
        if "avx512_vnni" in flags:
            return "avx512_vnni"
        if "avx512" in flags:
            return "avx512"
    except OSError:
        pass
    return "avx2"


def _export_text(output_dir: Path, models: dict[str, str]) -> None:
    """Export each text candidate as FP32 and dynamically quantised INT8.

    torch.onnx.export is not used here: its current exporter emits weights as a
    separate external-data file and attaches shape metadata that ONNX Runtime's
    dynamic quantiser rejects. optimum targets ONNX Runtime directly and produces
    a single self-contained graph that quantises cleanly.
    """
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    target = _quantization_target()
    for name, model_id in models.items():
        LOGGER.info("Loading and exporting %s (%s)", name, model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)

        fp32_dir = output_dir / f"text_{name}_fp32"
        model.save_pretrained(fp32_dir)
        tokenizer.save_pretrained(fp32_dir)

        int8_dir = output_dir / f"text_{name}_int8"
        LOGGER.info("  quantising (dynamic INT8, target=%s)", target)
        quantizer = ORTQuantizer.from_pretrained(model)
        quantizer.quantize(
            save_dir=int8_dir,
            quantization_config=getattr(AutoQuantizationConfig, target)(
                is_static=False, per_channel=False
            ),
        )
        tokenizer.save_pretrained(int8_dir)

        for precision, directory in (("fp32", fp32_dir), ("int8", int8_dir)):
            for graph in sorted(directory.glob("*.onnx")):
                LOGGER.info(
                    "  %s/%s: %s (%.1f MB)",
                    name,
                    precision,
                    graph.name,
                    graph.stat().st_size / 1e6,
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
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("text", "vision"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(TEXT_MODELS),
        default=sorted(TEXT_MODELS),
        help="Text candidates to export (ignored for --task vision)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "text":
        _export_text(args.output_dir, {name: TEXT_MODELS[name] for name in args.models})
    else:
        _export_vision(args.output_dir)
    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
