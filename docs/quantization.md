# Quantisation and the variant frontier

Which variants are worth serving is a property of the hardware. This project
measures it instead of assuming a precision ladder, because on the reference
host the assumption is wrong.

The two text models are
[DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
([Sanh et al., 2019](https://arxiv.org/abs/1910.01108)) and
[MiniLM-L6-H384](https://huggingface.co/philschmid/MiniLM-L6-H384-uncased-sst2)
([Wang et al., 2020](https://arxiv.org/abs/2002.10957)), both already fine-tuned on
SST-2, and the decoder is [GPT-2 124M](https://huggingface.co/openai-community/gpt2).
Quantisation is [ONNX Runtime's](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html),
applied through [optimum](https://huggingface.co/docs/optimum/index).

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

## Decoders: the same conclusion for INT4, the opposite one for INT8

`scripts/export_decoder.py` exports GPT-2 124M with its KV cache in the graph
signature and measures each precision against
[WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext)
([Merity et al., 2016](https://arxiv.org/abs/1609.07843)). Perplexity is scored
through the serving path, over 32,736 tokens in 32 non-overlapping 1024-token
windows.

| Precision | Size | vs FP32 | Perplexity | Delta | Prefill p50, one thread |
| --- | --- | --- | --- | --- | --- |
| `fp32` | 653 MB | 1.000 | 31.307 | n/a | 450 ms |
| `int8` | 399 MB | 0.611 | 31.371 | +0.063 | 415 ms |
| `int4` | 367 MB | 0.563 | 32.866 | +1.559 | 1837 ms |

The prefill column is measured by `export_decoder.py` in its own single-threaded ONNX
Runtime session, and it is left that way because it is what the export measured. **It
is not the served configuration**, which runs eight intra-op threads and reads very
differently for INT4; [`benchmarks.md`](benchmarks.md) has that. Perplexity and size
are properties of the graph and do not depend on either.

INT8 is nearly free in quality terms, costing 0.06 perplexity for a 39% smaller
graph, and is marginally faster. INT4 costs 1.56 perplexity and, **on one thread**,
runs 4.1x slower than FP32. That is the encoder finding again, and worse: on this host the
[`MatMulNBits`](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantize-to-int4uint4)
operator that 4-bit weights are folded into has no tuned kernel, so they are
unpacked to float on every matrix multiply and the arithmetic happens at full width
anyway. INT4 here buys memory and nothing else.

Prefill is a single 1024-token forward pass with an empty cache, which is the
right shape to compare on for that claim: it is dominated by the weight-bound
matrix multiplies that quantisation changes.

### The precision ordering depends on the phase

A decoder has a second shape to compare on, and it gives a different answer.
`scripts/profile_decode.py` measures a decode step, meaning one token against a filled cache
-- which is a much lower-arithmetic-intensity shape than a 128-token encoder pass, and
there moving a quarter of the weight bytes helps:

| Precision | TPOT at 128 cached | at 512 | at 960 | vs FP32 |
| --- | --- | --- | --- | --- |
| `fp32` | 5.12 ms | 6.66 ms | 8.15 ms | 1.000x |
| `int8` | 3.58 ms | 5.38 ms | 7.22 ms | 0.70x to 0.89x |
| `int4` | 4.53 ms | 6.24 ms | 7.22 ms | 0.88x to 0.94x |

So **INT8 is not dominated on the decoder.** It is faster on prefill and 11-28% faster
on decode, at 0.61x the size for 0.06 perplexity. That is the claim, and it is worth
not inflating: INT8 does not dominate either. INT4 is smaller, at 367 MB against 399,
and FP32 is more accurate, at 31.307 perplexity against 31.371, so all three sit on the
decoder's frontier and what INT8 wins is speed. The earlier finding stands unchanged as
what it always was: a statement about an encoder at sequence length 128, not about a
precision.

It is also not one experiment with one variable. The encoder and decoder measurements
differ in model, in shape, and in the recipe itself: per-tensor across the whole
encoder graph, against per-channel over `MatMul` and `Gemm` with the output projection
excluded for the decoder, for the reason two sections down. So the reversal shows that
the encoder conclusion does not generalise; it does not identify which of the three
differences produced it, and the shape explanation alone cannot, since INT8 also wins
the 256-token chunked prefill. [`benchmarks.md`](benchmarks.md) carries the one split
the measurement does isolate: precision moves the cache-independent part of a decode
step and leaves the per-cached-token part alone.

INT4's speed penalty turned out to be mostly about thread count, not about the
format. Unpacking 4-bit weights is a fixed cost per matrix multiply, so it lands on the
constant part of a step, and it is ordinary arithmetic, which a thread pool divides
well. With the decoder session on eight threads its fitted constant falls from 10.76 ms
to 4.28 ms and it decodes *faster* than FP32 at every cache occupancy, having been 1.7x
to 2.4x slower on one. It still loses on prefill and still costs 1.559 perplexity.

That is worth stating carefully, because it is the second time on this page that a
quantisation "result" turned out to be a statement about something other than the
format: first the shape, now the thread count.

The lesson is the one this page opens with, one level further in. A quantisation
result is a statement about a configuration; it is also a statement about a
*shape*, and a decoder runs two shapes that disagree.

### Two decisions that decide whether the numbers mean anything

Both were found by disbelieving the first result instead of recording it.

Both are ratios from an earlier, smaller scoring configuration whose unquantised
baseline read 26.8 against the 31.307 in the table above. They are comparable with
each other and not with that table; what they establish is the size of the effect, not
a perplexity for any variant that ships.

**Leave the output projection in float.** It maps the hidden state onto 50257
vocabulary entries, so its error lands directly on the distribution being scored.
Quantising it to symmetric 4-bit and nothing else measured perplexity of **1265**
against that 26.8 baseline. Both quantisers here exclude it.

**GPT-2's linear layers export as `Gemm`, not `MatMul`.** PyTorch implements them
as `Conv1D`, and `MatMulNBitsQuantizer` only rewrites `MatMul`. Left alone, INT4
reached exactly one node in the whole graph. That node was the output projection, the
one node that must not be touched, which is how both failures above arrived together.
`export_decoder.py` rewrites `Gemm(A, B, C)` as `Add(MatMul(A, B), C)` first, which
is exact at alpha = beta = 1 with no transpose and is asserted bitwise lossless in
`tests/test_decoder_export.py`. Models built from `nn.Linear`, the Llama family
among them, export as `MatMul` and need no rewrite.

The first INT8 attempt was per-tensor and quantised the output projection: 44.4
perplexity against the same 26.8 baseline. Per-channel scales with the projection left
in float cost 0.06 against the 32-window baseline. A quantisation result is a statement
about a configuration and not about a precision, which is also why the INT8 recipe here
and the encoder's are not the same experiment run twice.

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

## Static quantisation: it removes the determinism problem, and it ships nowhere

Both dynamically quantised encoder graphs carry 50 `DynamicQuantizeLinear` nodes, so
the activation scale is computed at runtime from the tensor actually fed. A request's
logits therefore depend on its own padding and on whoever shares its batch. The
question `benchmarks.md` carried as untested was whether that is the export's doing or
the runtime's. It is the export's, and calibrating the scales away removes it.

`python scripts/export_onnx.py --task text --quantization static` calibrates over 512
SST-2 **train** sentences, tokenised at 128 tokens, the length the benchmarks feed.
Never validation, which is what all 872 accuracies here are scored on.

**The comparison is controlled, and that took one deliberate choice.** Static QDQ left
to its defaults quantises 24 operator types; dynamic quantises three. Exported that way
the two would differ in how much of the graph is 8-bit *as well as* in where the scale
comes from, and neither result would be attributable. `QUANTISED_OPERATORS` pins both to
`MatMul`, `Gemm` and the embedding `Gather`. That is what the dynamic exporter picks on
its own, checked against the shipped graph. The two then hold the
same number of 8-bit weight elements to within the 100 that are the scales themselves:
66,892,840 against 66,892,940 for DistilBERT.

### What it removes, and what it does not

| Variant | `DynamicQuantizeLinear` | Predictions flipped, w2-w32 | Requests whose logits move | Median move of those |
| --- | --- | --- | --- | --- |
| `distilbert_fp32` | 0 | 0 | 614 / 872 | 9.5e-07 |
| `distilbert_int8` | 50 | 4-6 | **872 / 872** | 2.5e-02 |
| `distilbert_int8_static` | **0** | **0** | **1 / 872** | 2.9e-01 |
| `minilm_fp32` | 0 | 0 | 600 / 872 | 4.8e-07 |
| `minilm_int8` | 50 | 2-5 | **872 / 872** | 1.5e-02 |
| `minilm_int8_static` | **0** | **0** | **5 / 872** | 2.7e-02 |

**Half the prediction was right and half was wrong, and the wrong half is the more
useful.** Flips go to zero at every width, as expected. `max_abs_logit_delta` was
predicted to fall to the FP32 control's 1.1e-05 and it does not. It falls from 1.18 to
0.29, four orders of magnitude short.

What collapses is not the size of the movement but **how many requests move at all**:
872 of 872 to 1 of 872. Those are two different mechanisms with two different
signatures, and the maximum alone cannot tell them apart:

- **Dynamic quantisation** gives every request its own scale, so every request moves,
  each by a hundredth of a logit or so.
- **A static grid** cannot move unless a float perturbation crosses a rounding
  boundary. Almost none do. The one that does moves by a whole quantisation step,
  which is why the maximum stays large while the incidence collapses.
- **The FP32 control is the third signature** and it is what makes the other two
  legible: 614 of 872 requests move, by a median of 9.5e-07. Everything wobbles
  infinitesimally. That is float reassociation, and it does not go away at any
  precision.

Counting `rows_moved` beside the maximum is what separated these. A maximum over 872
requests cannot distinguish one sentence near a rounding boundary from every sentence
wobbling.

### What it costs, and why it ships nowhere

| Model | FP32 | INT8 dynamic | INT8 static, minmax | INT8 static, percentile |
| --- | --- | --- | --- | --- |
| DistilBERT | 91.06% | 90.37% | 89.45% | **89.91%** |
| MiniLM | 90.14% | 89.91% | **90.02%** | 89.56% |

**Neither calibration method wins on both models.** Percentile clipping at 99.999 halves
DistilBERT's loss and costs MiniLM 0.46pp. Taking the better of the two per model, the
price of determinism is **0.46pp on DistilBERT and nothing on MiniLM**, which is a
genuinely small price.

**It is declined anyway, and not on accuracy.** Both INT8 encoder variants were already
*dominated* before determinism was ever the question. `distilbert_int8` serves in
17.17 ms against `distilbert_fp32`'s 12.89 ms, and `minilm_int8` in 7.16 ms against
5.19 ms. They are slower **and** no more accurate, which is why `configs/serving.yaml`
carries only the two FP32 variants. Static quantisation improves an axis that was not
the reason INT8 lost. A graph that nothing selects does not become selectable by
answering a question about it.

So this is filed as an answered question, not a shipped variant. The exports are
reproducible from one flag and the counts are committed in
`results/encoder_batching.json`; nothing in the serving path changes.

**What would change the answer**: an execution provider where INT8 is actually faster
than FP32 on this graph, which would put INT8 back on the frontier and make its
determinism a property worth paying 0.46pp for.

## Export notes

`torch.onnx.export` is not used for the text models. Its current exporter emits
weights as a separate external-data file and attaches shape metadata that ONNX
Runtime's dynamic quantiser rejects with an inference error. `optimum` targets
ONNX Runtime directly and produces a single self-contained graph that quantises
cleanly, so the text path uses `ORTModelForSequenceClassification` and
`ORTQuantizer`.

## Reproducing

```bash
python scripts/export_onnx.py --task text     # both models, FP32 and dynamic INT8
python scripts/export_onnx.py --task text --quantization static
python scripts/export_onnx.py --task text --quantization static \
    --calibration-method percentile
python scripts/profile_variants.py            # measure and rank
python scripts/count_encoder_batching.py      # the determinism counts above
```

An FP32 graph already on disk is reused instead of rebuilt, because every committed
encoder number was measured against the one that is there. Delete the directory to
force a rebuild.

The verdict per variant, the host, and the ONNX Runtime version are all recorded
in `results/variant_profiles.json`.
