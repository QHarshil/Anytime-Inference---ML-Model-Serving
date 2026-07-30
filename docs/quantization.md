# Quantisation and the variant frontier

Which variants are worth serving is a property of the hardware. This project
measures it rather than assuming a precision ladder, because on the reference
host the assumption is wrong.

## The finding

Dynamic INT8 quantisation shrinks both text models about fourfold and costs
little or no accuracy, but does not make either faster on Apple Silicon:

| Variant | p50 | Accuracy | Size |
| --- | --- | --- | --- |
| `distilbert_fp32` | 12.89 ms | 91.06% | 268 MB |
| `distilbert_int8` | 17.17 ms | 90.60% | 67 MB |
| `minilm_fp32` | 5.19 ms | 90.14% | 91 MB |
| `minilm_int8` | 7.16 ms | 90.14% | 23 MB |

Median of three measurement passes through the serving path; see
[`benchmarks.md`](benchmarks.md) for the spread and the method.

Both INT8 variants are **strictly dominated**: slower than an alternative that is
at least as accurate. A dominated variant is never the right choice for a planner
optimising accuracy under a latency bound, so neither is offered.

INT8 remains useful when memory, not latency, is the binding constraint:
`minilm_int8` holds full accuracy at 23 MB, an 11x reduction against
`distilbert_fp32`.

## Consequence for the design

The accuracy/latency axis on this host is **model capacity**, not precision.
MiniLM-L6-H384 is 2.48x faster than DistilBERT-base for 0.92 accuracy points,
which is a genuine tradeoff the planner can exploit. The frontier is therefore
computed from measurements:

```python
# A variant is dominated when another is at least as fast and at least as
# accurate, and strictly better on one of the two.
```

`scripts/profile_variants.py` marks each variant and writes only frontier
variants into `configs/serving.yaml`. On a host where INT8 does accelerate
inference, the same script will place the INT8 variants on the frontier and the
planner will use them with no code change.

## Quantising for the right architecture

Quantising for the wrong instruction set is worse than not quantising: the INT8
operators fall back to reference kernels. DistilBERT quantised for `avx512_vnni`
and served on arm64 measured 16.25 ms against the FP32 graph's 13.31 ms.
Retargeting to `arm64` narrowed but did not close the gap.

`scripts/export_onnx.py` selects the target from the host:

| Host | Target |
| --- | --- |
| arm64 / aarch64 | `arm64` |
| x86-64 with AVX512-VNNI | `avx512_vnni` |
| x86-64 with AVX512 | `avx512` |
| other x86-64 | `avx2` |
| ppc64le | `ppc64le` |

x86 feature detection reads `/proc/cpuinfo` and falls back to `avx2`, which still
uses real integer kernels.

## Export notes

`torch.onnx.export` is not used for the text models. Its current exporter emits
weights as a separate external-data file and attaches shape metadata that ONNX
Runtime's dynamic quantiser rejects with an inference error. `optimum` targets
ONNX Runtime directly and produces a single self-contained graph that quantises
cleanly, so the text path uses `ORTModelForSequenceClassification` and
`ORTQuantizer`.

## Reproducing

```bash
python scripts/export_onnx.py --task text     # both models, FP32 and INT8
python scripts/profile_variants.py            # measure and rank
```

The verdict per variant, the host, and the ONNX Runtime version are all recorded
in `results/variant_profiles.json`.
