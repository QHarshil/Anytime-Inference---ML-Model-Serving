"""Utility to download and cache datasets required by the experiments.

The script relies on the Hugging Face `datasets` library so that data
artifacts are stored in a consistent format across machines. Downloaded
splits are cached under `data/<dataset_name>`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

from datasets import load_dataset

# Mapping from logical dataset identifier to the tuple of arguments passed to
# `datasets.load_dataset`. The second element of the tuple is optional and can
# be `None` if the dataset has no configuration name.
_DATASETS: Dict[str, Tuple[str, str | None]] = {
    "sst2": ("glue", "sst2"),
    "cifar10": ("cifar10", None),
}


def download_dataset(name: str, root: Path) -> None:
    """Download a dataset and store it under ``root / name``.

    Parameters
    ----------
    name:
        Logical identifier from :data:`_DATASETS`.
    root:
        Root directory where datasets are cached.
    """

    if name not in _DATASETS:
        available = ", ".join(sorted(_DATASETS))
        raise ValueError(f"Unknown dataset '{name}'. Available options: {available}")

    dataset_name, config_name = _DATASETS[name]
    target_dir = root / name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading {name} -> {target_dir}")
    dataset_kwargs = {}
    if config_name is not None:
        dataset_kwargs["name"] = config_name

    ds = load_dataset(dataset_name, **dataset_kwargs)
    ds.save_to_disk(str(target_dir))
    print(f"[OK] Finished downloading {name}")


def download_all(datasets: Iterable[str], root: Path) -> None:
    """Download each dataset listed in ``datasets``."""

    for dataset in datasets:
        download_dataset(dataset, root)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(_DATASETS),
        help="Datasets to download (defaults to all supported datasets).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where datasets should be cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_all(args.datasets, args.output_dir)


if __name__ == "__main__":
    main()
