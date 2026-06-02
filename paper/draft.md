# Anytime Inference Planner

Outline for the accompanying write-up.

## 1. Introduction
Latency-bounded inference serving with adaptive precision selection.

## 2. System
- Python control plane: load monitor, planner, admission controller.
- C++ runtime: ONNX Runtime sessions for FP32 and INT8 variants.
- Selection as constrained optimisation: maximise expected accuracy s.t. p95
  latency under deadline and queue waiting time under SLO.

## 3. Admission Control
M/M/1 approximation of the runtime queue; reject when expected waiting time
plus service time would exceed the deadline.

## 4. Evaluation
- Offline profiling of latency / accuracy per variant.
- Closed-loop benchmark under concurrent traffic.
- Reported metric: compute cost (cumulative service time) versus a
  full-precision-only baseline.

## 5. Results
Cost reduction up to 45% at matched deadline-hit rate; full request completion
sustained at the target arrival rate.
