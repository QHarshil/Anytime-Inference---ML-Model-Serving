# Datasets

Cached datasets for the offline experiment pipeline. Nothing here is versioned;
fetch them locally instead:

```bash
python data/download_datasets.py
```

Downloads GLUE SST-2 and CIFAR-10 by default. Run with `--help` for the available
options.

The serving benchmarks in [`../docs/benchmarks.md`](../docs/benchmarks.md) use
SST-2 validation, which this script fetches through the `datasets` library.
