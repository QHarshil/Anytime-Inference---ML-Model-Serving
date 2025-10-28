#!/bin/bash
# Setup script for Anytime Inference Planner
# Run this first to set up the environment

set -e  # Exit on error

echo "================================================================"
echo "ANYTIME INFERENCE PLANNER - SETUP"
echo "================================================================"

# Check Python version
echo "Checking Python version..."
python3 --version

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment (optional but recommended)
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⊘ Existing virtual environment found, removing..."
    rm -rf venv
fi
python3 -m venv venv
echo "✓ Virtual environment created"

PYTHON_BIN="venv/bin/python"
PIP_BIN="venv/bin/pip"

if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$PIP_BIN" ]; then
    echo "Warning: virtual environment executables not found; falling back to system Python."
    PYTHON_BIN=$(command -v python3 || true)
    PIP_BIN=$(command -v pip3 || true)
    if [ -z "$PYTHON_BIN" ] || [ -z "$PIP_BIN" ]; then
        echo "Error: Unable to locate python3/pip3."
        exit 1
    fi
fi

echo ""
echo "To activate virtual environment, run:"
echo "  source venv/bin/activate  # Linux/Mac"
echo "  venv\\Scripts\\activate     # Windows"
echo ""

# Install dependencies
echo "Installing dependencies..."
"$PIP_BIN" install --upgrade pip
"$PIP_BIN" install -r requirements.txt

echo ""
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating result directories..."
mkdir -p results/ablation
mkdir -p results/plots
mkdir -p data
echo "✓ Directories created"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
"$PYTHON_BIN" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
echo "================================================================"
echo "SETUP COMPLETED"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment (if created):"
echo "     source venv/bin/activate"
echo ""
echo "  2. Download datasets:"
echo "     python data/download_datasets.py"
echo ""
echo "  3. Run full pipeline:"
echo "     python run_all.py"
echo ""
echo "  4. Or run individual experiments:"
echo "     python experiments/01_profile_latency.py"
echo "     python experiments/02_profile_accuracy.py"
echo "     ..."
echo ""
echo "For quick testing (minimal data):"
echo "  python run_all.py --quick-test"
echo ""
echo "================================================================"
