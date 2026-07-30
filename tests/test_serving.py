"""End-to-end tests for the serving stack using a tiny on-the-fly ONNX model."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import onnx
    import onnxruntime  # noqa: F401
    from onnx import TensorProto, helper
except ImportError:  # pragma: no cover - dependency check
    onnx = None


def _build_identity_model(path: Path) -> None:
    """Build a tiny ONNX graph that returns its input unchanged."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    output_tensor = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 4])
    node = helper.make_node("Identity", inputs=["input"], outputs=["logits"])
    graph = helper.make_graph([node], "identity", [input_tensor], [output_tensor])
    opset = helper.make_opsetid("", 14)
    model = helper.make_model(graph, opset_imports=[opset], ir_version=8)
    onnx.save(model, str(path))


@unittest.skipIf(onnx is None, "onnx/onnxruntime not installed")
class TestRuntimeClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.fp32_path = Path(cls._tmp.name) / "identity_fp32.onnx"
        cls.int8_path = Path(cls._tmp.name) / "identity_int8.onnx"
        _build_identity_model(cls.fp32_path)
        _build_identity_model(cls.int8_path)
        cls.model_paths: dict[str, Path] = {
            "fp32": cls.fp32_path,
            "int8": cls.int8_path,
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_python_backend_round_trip(self):
        from anytime_serving.serving.onnx_runtime import InferenceRequest, RuntimeClient

        with RuntimeClient(self.model_paths) as client:
            data = np.arange(8, dtype=np.float32).reshape(2, 4)
            response = client.infer(InferenceRequest(variant="fp32", data=data))
            np.testing.assert_array_almost_equal(response.logits, data)
            self.assertGreaterEqual(response.runtime_latency_ms, 0.0)
            self.assertGreaterEqual(response.wall_latency_ms, response.runtime_latency_ms - 1e-6)

    def test_pool_dispatches_concurrent_requests(self):
        from concurrent.futures import ThreadPoolExecutor

        from anytime_serving.serving.onnx_runtime import InferenceRequest, RuntimePool

        with RuntimePool(size=3, model_paths=self.model_paths) as pool:
            rng = np.random.default_rng(0)
            requests = [
                InferenceRequest(
                    variant="int8" if i % 2 else "fp32",
                    data=rng.standard_normal((1, 4)).astype(np.float32),
                )
                for i in range(8)
            ]
            with ThreadPoolExecutor(max_workers=4) as executor:
                responses = list(executor.map(pool.infer, requests))
            for req, resp in zip(requests, responses, strict=True):
                np.testing.assert_array_almost_equal(resp.logits, req.data)


@unittest.skipIf(onnx is None, "onnx/onnxruntime not installed")
class TestAdaptiveServer(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.fp32_path = Path(self._tmp.name) / "identity_fp32.onnx"
        self.int8_path = Path(self._tmp.name) / "identity_int8.onnx"
        _build_identity_model(self.fp32_path)
        _build_identity_model(self.int8_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_concurrent_workload_admits_majority(self):
        from anytime_serving.serving.load_monitor import LoadMonitor
        from anytime_serving.serving.onnx_runtime import InferenceRequest, RuntimePool
        from anytime_serving.serving.selector import AdaptiveSelector, VariantProfile
        from anytime_serving.serving.server import AdaptiveServer, drive_workload, poisson_arrivals

        variants = [
            VariantProfile(
                "fp32", service_time_ms=5.0, accuracy=0.91, compute_cost_per_request=1.0
            ),
            VariantProfile(
                "int8", service_time_ms=2.0, accuracy=0.89, compute_cost_per_request=0.4
            ),
        ]
        selector = AdaptiveSelector(variants, servers=2)
        monitor = LoadMonitor(interval_s=0.05)
        monitor.start()
        try:
            with RuntimePool(2, {"fp32": self.fp32_path, "int8": self.int8_path}) as pool:
                server = AdaptiveServer(pool, selector, monitor)
                try:
                    rng = np.random.default_rng(1)

                    def factory(_i: int) -> InferenceRequest:
                        return InferenceRequest(
                            variant="fp32",
                            data=rng.standard_normal((1, 4)).astype(np.float32),
                        )

                    arrivals = poisson_arrivals(
                        duration_s=0.5, rate_rps=20.0, rng=np.random.default_rng(2)
                    )
                    drive_workload(server, factory, arrival_times=arrivals, deadline_ms=200.0)
                finally:
                    server.shutdown()
                stats = server.stats
        finally:
            monitor.stop()

        self.assertEqual(stats.total, len(arrivals))
        self.assertGreater(stats.accepted, 0)
        # With identity models and a generous deadline, every accepted request
        # should complete in time.
        self.assertEqual(stats.deadline_misses, 0)

    def test_rejects_selector_pool_size_mismatch(self):
        """A selector modelling the wrong worker count must not be accepted.

        Modelling a c-worker pool as a single server understates capacity by a
        factor of c, so the server refuses the pairing outright rather than
        silently shedding load the pool could serve.
        """
        from anytime_serving.serving.load_monitor import LoadMonitor
        from anytime_serving.serving.onnx_runtime import RuntimePool
        from anytime_serving.serving.selector import AdaptiveSelector, VariantProfile
        from anytime_serving.serving.server import AdaptiveServer

        variants = [
            VariantProfile("fp32", service_time_ms=5.0, accuracy=0.91, compute_cost_per_request=1.0)
        ]
        selector = AdaptiveSelector(variants, servers=1)
        monitor = LoadMonitor(interval_s=0.05)
        with RuntimePool(4, {"fp32": self.fp32_path}) as pool:
            with self.assertRaises(ValueError) as ctx:
                AdaptiveServer(pool, selector, monitor)
        self.assertIn("selector models 1 server", str(ctx.exception))

    def test_pool_reports_its_size(self):
        from anytime_serving.serving.onnx_runtime import RuntimePool

        with RuntimePool(3, {"fp32": self.fp32_path}) as pool:
            self.assertEqual(pool.size, 3)


if __name__ == "__main__":
    unittest.main()
