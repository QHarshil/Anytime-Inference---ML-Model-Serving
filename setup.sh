#!/bin/bash
# Bootstrap a Python venv, install dependencies, and create result directories.

set -e

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but was not found on PATH." >&2
    exit 1
fi

python3 --version

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

mkdir -p results/ablation results/plots data

venv/bin/python -c "import torch; print(f'cuda_available={torch.cuda.is_available()}')"

echo
echo "Setup complete. Activate the environment with:  source venv/bin/activate"
