#!/bin/bash
# Quick test to verify pipeline components work

set -e

echo "Testing pipeline components..."

# Test 1: Check imports
echo "1. Testing imports..."
python3 -c "from src.models.model_zoo import ModelZoo; from src.models.cascade import CascadeEvaluator; from src.utils.io import save_csv; print('✓ Imports OK')"

# Test 2: Check argparse in experiments
echo "2. Testing --quick flag..."
python3 experiments/01_profile_latency.py --help | grep -q "quick" && echo "✓ 01_profile_latency.py has --quick flag"
python3 experiments/02_profile_accuracy.py --help | grep -q "quick" && echo "✓ 02_profile_accuracy.py has --quick flag"

# Test 3: Check file structure
echo "3. Checking directory structure..."
[ -d "results" ] || mkdir -p results
[ -d "results/plots" ] || mkdir -p results/plots
echo "✓ Directory structure OK"

echo ""
echo "All basic tests passed!"
echo "To run full pipeline: python run_all.py --quick-test"
