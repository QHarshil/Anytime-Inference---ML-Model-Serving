#!/bin/bash
# Bootstrap a Python venv and install the package with its development extras.

set -euo pipefail

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but was not found on PATH." >&2
    exit 1
fi

python3 --version

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

venv/bin/pip install --upgrade pip
venv/bin/pip install -e ".[dev]"

echo
echo "Setup complete. Activate the environment with:  source venv/bin/activate"
echo "Next: python scripts/demo_serving.py"
