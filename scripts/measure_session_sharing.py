"""Measure what sharing one set of sessions across pool workers saves in memory.

`RuntimePool` loads every variant once per worker by default, so N workers hold N copies
of the same read-only weights. `share_sessions=True` loads them once. This measures the
difference, in resident bytes, on the real exported models.

Counting, not timing. Resident set size does not depend on how busy the machine is, so
this is one of the few measurements in this repository that a contended host cannot
corrupt, unlike anything under `results/ab_copy_threads/`, which needs a gated arm.

Each arm runs in its own subprocess. Measuring both in one would charge the second arm
for pages the first already faulted in and read the saving as far smaller than it is.

What this does not measure is latency. Sharing puts every worker on one per-session CPU
arena, and whether that contends under concurrency is open; `share_sessions` is off by
default until it has been measured.

Usage:
    python scripts/measure_session_sharing.py
    python scripts/measure_session_sharing.py --workers 1 2 4 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anytime_serving.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("scripts.measure_session_sharing")

# Run inside the child. Kept as source instead of a module so the parent does not import
# onnxruntime itself and charge the baseline for it.
CHILD = """
import json, os, sys
sys.path.insert(0, {src!r})
import numpy as np, psutil
from pathlib import Path
from anytime_serving.serving.onnx_runtime import InferenceRequest, RuntimePool

share = sys.argv[1] == "shared"
workers = int(sys.argv[2])
paths = {{name: Path(p) for name, p in json.loads(sys.argv[3]).items()}}
process = psutil.Process(os.getpid())
before = process.memory_info().rss
pool = RuntimePool(workers, paths, share_sessions=share)
# Drive every worker once so no session is left lazily unmapped.
data = np.zeros((1, 128), dtype=np.int64)
for _ in range(workers * 2):
    try:
        pool.infer(InferenceRequest(variant=sorted(paths)[0], data=data))
    except Exception:
        break
after = process.memory_info().rss
print(json.dumps({{"share": share, "workers": workers,
                  "backends": pool.loaded_backends,
                  "rss_mb": (after - before) / 2**20}}))
pool.close()
"""


def _arm(src: Path, share: bool, workers: int, paths: dict[str, str]) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD.format(src=str(src)),
            "shared" if share else "unshared",
            str(workers),
            json.dumps(paths),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"arm failed (share={share}, workers={workers}):\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", type=Path, default=Path("results/session_sharing.json"))
    args = parser.parse_args()

    paths = {
        "distilbert_fp32": args.model_dir / "text_distilbert_fp32" / "model.onnx",
        "minilm_fp32": args.model_dir / "text_minilm_fp32" / "model.onnx",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit(
            "these exported models are missing:\n    "
            + "\n    ".join(missing)
            + "\nExport them first:\n    python scripts/export_onnx.py --task text"
        )

    src = Path(__file__).resolve().parents[1] / "src"
    as_str = {name: str(path) for name, path in paths.items()}
    rows = []
    for workers in args.workers:
        unshared = _arm(src, False, workers, as_str)
        shared = _arm(src, True, workers, as_str)
        saved = unshared["rss_mb"] - shared["rss_mb"]
        rows.append(
            {
                "workers": workers,
                "unshared_backends": unshared["backends"],
                "shared_backends": shared["backends"],
                "unshared_rss_mb": round(unshared["rss_mb"], 1),
                "shared_rss_mb": round(shared["rss_mb"], 1),
                "saved_mb": round(saved, 1),
                "saved_fraction": round(saved / unshared["rss_mb"], 4)
                if unshared["rss_mb"]
                else 0.0,
            }
        )
        LOGGER.info(
            "%d worker(s): %.0f MB unshared, %.0f MB shared, %.0f MB saved (%.0f%%)",
            workers,
            unshared["rss_mb"],
            shared["rss_mb"],
            saved,
            100.0 * saved / unshared["rss_mb"] if unshared["rss_mb"] else 0.0,
        )

    payload = {
        "measurement": "resident set size of a RuntimePool, shared against per-worker",
        "variants": sorted(paths),
        "note": "memory only; the latency cost of a shared CPU arena is not measured",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
