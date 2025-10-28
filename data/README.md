# Data Assets

This directory stores cached datasets used by the anytime inference planner experiments. Datasets are not versioned directly; instead, run `download_datasets.py` to fetch them locally under this folder.

```
python data/download_datasets.py
```

By default the script downloads the SST-2 text classification benchmark and the CIFAR-10 image dataset. You can request additional datasets with the `--datasets` flag.
