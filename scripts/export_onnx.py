"""Export FP32 and INT8 variants of the planner models to ONNX.

INT8 comes in two flavours and the difference is the point of `--quantization`:

    dynamic  the activation scale is computed at runtime from the tensor actually
             fed, by a `DynamicQuantizeLinear` node per quantised matrix op. A
             request's answer therefore depends on its own padding and on whoever
             shares its batch -- measured at 0.23-0.69% of predictions.
    static   the activation scale is a constant, calibrated once against real data
             before the graph is saved. No `DynamicQuantizeLinear` survives, so
             nothing about the answer can depend on the rest of the tensor.

The two are exported over **the same operator set** (`QUANTISED_OPERATORS`) so the
only thing that differs between them is where the activation scale comes from. That
is not the default: static QDQ would otherwise quantise 24 operator types including
`LayerNormalization` and `Where`, while dynamic quantises `MatMul`, `Gemm` and the
embedding `Gather` and nothing else. Exporting both with their own defaults would
compare two things at once.

Usage:
    python scripts/export_onnx.py --task text --output-dir models/onnx
    python scripts/export_onnx.py --task text --quantization both
    python scripts/export_onnx.py --task vision --output-dir models/onnx
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from anytime_serving.utils.logger import get_logger

LOGGER = get_logger("scripts.export_onnx")


# Text candidates, ordered from most to least capacity. Both are genuinely
# fine-tuned on SST-2, so accuracy measured on that split is meaningful.
TEXT_MODELS = {
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "minilm": "philschmid/MiniLM-L6-H384-uncased-sst2",
}

QUANTIZATION_MODES = ("dynamic", "static", "both")

# The operators both INT8 flavours quantise, so that static and dynamic differ in
# exactly one thing: where the activation scale comes from.
#
# This is what the dynamic exporter picks on its own -- checked against the shipped
# graphs rather than assumed. DistilBERT FP32 has 48 MatMul and 2 Gemm, and the
# dynamic INT8 graph has 50 MatMulInteger and 50 DynamicQuantizeLinear; its word
# embedding table is UINT8, which is the `Gather`. Static QDQ left to itself would
# quantise 24 operator types, which would make it a different experiment.
QUANTISED_OPERATORS = ["MatMul", "Gemm", "Gather"]

# Calibration comes from SST-2 *train*. Calibrating on validation is how a
# quantisation result reports an accuracy it did not earn: the 872 validation
# sentences are what every accuracy on this model is scored against.
CALIBRATION_DATASET = ("glue", "sst2")
CALIBRATION_SPLIT = "train"
DEFAULT_CALIBRATION_SAMPLES = 512

# How the activation range is taken from the calibration run.
#
#   minmax      the widest value seen. Simple, and blunt where activations have
#               outliers -- which transformers do, so this is the one to suspect
#               first if a static export loses accuracy.
#   percentile  a histogram, clipped at PERCENTILE. Trades a little clipping for a
#               much tighter grid over the values that actually occur.
#   entropy     picks the clip that minimises KL against the float distribution.
CALIBRATION_METHODS = ("minmax", "percentile", "entropy")
PERCENTILE = 99.999

# Calibration tokenises the way the benchmarks feed the model, so the activation
# ranges are taken over the tensor distribution that is actually served rather than
# over a tighter one. `run_load_sweep.py`, `profile_variants.py` and
# `count_encoder_batching.py` all use this length, and a test asserts all four agree.
CALIBRATION_SEQUENCE_LENGTH = 128


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


def _calibration_dataset(quantizer, tokenizer, samples: int):
    """Sample SST-2 train and tokenise it the way the benchmarks feed the model.

    Train, never validation -- see CALIBRATION_SPLIT. `get_calibration_dataset`
    shuffles with a fixed seed and then drops every column the graph does not
    declare, so what comes back is `input_ids`, `attention_mask` and, where the
    graph takes it, `token_type_ids`.
    """
    if CALIBRATION_SPLIT != "train":
        raise ValueError(
            f"calibration split is {CALIBRATION_SPLIT!r}. Calibrating on the split "
            f"the accuracy is scored on reports an accuracy the model did not earn."
        )

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=CALIBRATION_SEQUENCE_LENGTH,
            truncation=True,
        )

    dataset_name, config_name = CALIBRATION_DATASET
    return quantizer.get_calibration_dataset(
        dataset_name=dataset_name,
        dataset_config_name=config_name,
        dataset_split=CALIBRATION_SPLIT,
        num_samples=samples,
        preprocess_function=preprocess,
        preprocess_batch=True,
    )


def _export_text(
    output_dir: Path,
    models: dict[str, str],
    quantization: str = "dynamic",
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    calibration_method: str = "minmax",
) -> None:
    """Export each text candidate as FP32 and INT8, dynamically and/or statically.

    torch.onnx.export is not used here: its current exporter emits weights as a
    separate external-data file and attaches shape metadata that ONNX Runtime's
    dynamic quantiser rejects. optimum targets ONNX Runtime directly and produces
    a single self-contained graph that quantises cleanly.
    """
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoCalibrationConfig, AutoQuantizationConfig
    from transformers import AutoTokenizer

    target = _quantization_target()
    wants = ("dynamic", "static") if quantization == "both" else (quantization,)

    for name, model_id in models.items():
        fp32_dir = output_dir / f"text_{name}_fp32"

        # An FP32 graph already on disk is reused rather than rebuilt. Every
        # committed encoder number was measured against the graph that is there, and
        # a re-export that differed by so much as a node would silently make those
        # numbers describe something else. Delete the directory to force a rebuild.
        if sorted(fp32_dir.glob("*.onnx")):
            LOGGER.info("FP32 graph already present at %s; reusing it", fp32_dir)
            tokenizer = AutoTokenizer.from_pretrained(fp32_dir)
            quantizer_source: object = fp32_dir
        else:
            LOGGER.info("Loading and exporting %s (%s)", name, model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
            model.save_pretrained(fp32_dir)
            tokenizer.save_pretrained(fp32_dir)
            quantizer_source = model
        produced = [("fp32", fp32_dir)]

        if "dynamic" in wants:
            int8_dir = output_dir / f"text_{name}_int8"
            LOGGER.info("  quantising (dynamic INT8, target=%s)", target)
            quantizer = ORTQuantizer.from_pretrained(quantizer_source)
            quantizer.quantize(
                save_dir=int8_dir,
                quantization_config=getattr(AutoQuantizationConfig, target)(
                    is_static=False, per_channel=False
                ),
            )
            tokenizer.save_pretrained(int8_dir)
            produced.append(("int8", int8_dir))

        if "static" in wants:
            suffix = "" if calibration_method == "minmax" else f"_{calibration_method}"
            static_dir = output_dir / f"text_{name}_int8_static{suffix}"
            LOGGER.info(
                "  quantising (static INT8, target=%s, %s over %d calibration sample(s) "
                "from %s/%s %s)",
                target,
                calibration_method,
                calibration_samples,
                *CALIBRATION_DATASET,
                CALIBRATION_SPLIT,
            )
            quantizer = ORTQuantizer.from_pretrained(quantizer_source)
            config = getattr(AutoQuantizationConfig, target)(
                is_static=True,
                per_channel=False,
                operators_to_quantize=list(QUANTISED_OPERATORS),
            )
            calibration = _calibration_dataset(quantizer, tokenizer, calibration_samples)
            LOGGER.info("    calibrating over %d row(s)", len(calibration))
            if calibration_method == "minmax":
                calibration_config = AutoCalibrationConfig.minmax(calibration)
            elif calibration_method == "percentile":
                calibration_config = AutoCalibrationConfig.percentiles(
                    calibration, percentile=PERCENTILE
                )
            else:
                calibration_config = AutoCalibrationConfig.entropy(calibration)
            ranges = quantizer.fit(
                dataset=calibration,
                calibration_config=calibration_config,
                operators_to_quantize=config.operators_to_quantize,
                batch_size=8,
            )
            quantizer.quantize(
                save_dir=static_dir,
                quantization_config=config,
                calibration_tensors_range=ranges,
            )
            tokenizer.save_pretrained(static_dir)
            produced.append((f"int8_static{suffix}", static_dir))

        for precision, directory in produced:
            for graph in sorted(directory.glob("*.onnx")):
                LOGGER.info(
                    "  %s/%s: %s (%.1f MB)",
                    name,
                    precision,
                    graph.name,
                    graph.stat().st_size / 1e6,
                )


def _export_vision(output_dir: Path) -> None:
    # torch is imported here rather than at module scope so the constants above can
    # be read without it. `torch` is in the `research` extra and CI's test job
    # installs `bench`, so a module-level import would make every test that reads
    # CALIBRATION_SPLIT pass here and fail there.
    import torch
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
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_MODES,
        default="dynamic",
        help=(
            "Which INT8 flavour to produce alongside FP32. 'dynamic' is what the "
            "shipped graphs are; 'static' calibrates the activation scales so no "
            "DynamicQuantizeLinear survives (ignored for --task vision)"
        ),
    )
    parser.add_argument(
        "--calibration-method",
        choices=CALIBRATION_METHODS,
        default="minmax",
        help=(
            "How static quantisation takes an activation range from the calibration "
            "run. Anything but 'minmax' is written to a suffixed directory so the two "
            "can be measured side by side"
        ),
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help=f"SST-2 train rows to calibrate static quantisation over "
        f"(default {DEFAULT_CALIBRATION_SAMPLES})",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "text":
        _export_text(
            args.output_dir,
            {name: TEXT_MODELS[name] for name in args.models},
            quantization=args.quantization,
            calibration_samples=args.calibration_samples,
            calibration_method=args.calibration_method,
        )
    else:
        _export_vision(args.output_dir)
    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
