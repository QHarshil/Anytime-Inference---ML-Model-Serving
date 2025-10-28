"""Integration tests for the profiling and evaluation pipeline."""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import save_csv, load_csv


class TestProfilingPipeline(unittest.TestCase):
    """Test profiling pipeline integration."""
    
    def setUp(self):
        """Set up test directory."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)
    
    def test_latency_profile_schema(self):
        """Test latency profile CSV schema."""
        # Create mock latency profile
        data = {
            'task': ['text', 'text'],
            'model': ['distilbert', 'minilm'],
            'variant': ['fp32', 'fp32'],
            'device': ['cpu', 'cpu'],
            'batch_size': [1, 1],
            'lat_p50_ms': [50.0, 30.0],
            'lat_p95_ms': [80.0, 50.0],
            'items_per_sec': [20.0, 33.3]
        }
        
        df = pd.DataFrame(data)
        output_path = self.test_dir / "latency_profile.csv"
        save_csv(df, output_path)
        
        # Verify file exists and can be loaded
        self.assertTrue(output_path.exists())
        loaded_df = load_csv(output_path)
        self.assertEqual(len(loaded_df), 2)
        self.assertIn('lat_p50_ms', loaded_df.columns)
        self.assertIn('lat_p95_ms', loaded_df.columns)
    
    def test_accuracy_profile_schema(self):
        """Test accuracy profile CSV schema."""
        data = {
            'task': ['text', 'vision'],
            'model': ['distilbert', 'mobilenet'],
            'variant': ['fp32', 'fp32'],
            'accuracy': [0.85, 0.88],
            'num_samples': [872, 10000]
        }
        
        df = pd.DataFrame(data)
        output_path = self.test_dir / "accuracy_profile.csv"
        save_csv(df, output_path)
        
        self.assertTrue(output_path.exists())
        loaded_df = load_csv(output_path)
        self.assertEqual(len(loaded_df), 2)
        self.assertIn('accuracy', loaded_df.columns)
    
    def test_baseline_results_schema(self):
        """Test baseline results CSV schema."""
        data = {
            'task': ['text', 'text'],
            'method': ['StaticSmall', 'StaticLarge'],
            'deadline_ms': [100, 100],
            'seed': [42, 42],
            'deadline_hit_rate': [0.95, 0.65],
            'accuracy': [0.82, 0.89],
            'lat_p50_ms': [50.0, 120.0],
            'lat_p95_ms': [80.0, 180.0]
        }
        
        df = pd.DataFrame(data)
        output_path = self.test_dir / "baseline_results.csv"
        save_csv(df, output_path)
        
        self.assertTrue(output_path.exists())
        loaded_df = load_csv(output_path)
        self.assertEqual(len(loaded_df), 2)
        self.assertIn('deadline_hit_rate', loaded_df.columns)
    
    def test_planner_results_schema(self):
        """Test planner results CSV schema."""
        data = {
            'task': ['text', 'text'],
            'deadline_ms': [100, 100],
            'threshold': [0.8, 0.9],
            'seed': [42, 42],
            'deadline_hit_rate': [0.92, 0.88],
            'accuracy': [0.87, 0.89],
            'coverage': [0.75, 0.60],
            'lat_p50_ms': [60.0, 70.0],
            'lat_p95_ms': [95.0, 110.0]
        }
        
        df = pd.DataFrame(data)
        output_path = self.test_dir / "planner_results.csv"
        save_csv(df, output_path)
        
        self.assertTrue(output_path.exists())
        loaded_df = load_csv(output_path)
        self.assertEqual(len(loaded_df), 2)
        self.assertIn('threshold', loaded_df.columns)
        self.assertIn('coverage', loaded_df.columns)


class TestStatisticalPipeline(unittest.TestCase):
    """Test statistical analysis pipeline."""
    
    def setUp(self):
        """Set up test directory and mock data."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create mock baseline results
        baseline_data = []
        for seed in [42, 43, 44]:
            for method in ['StaticSmall', 'StaticLarge']:
                for deadline in [100, 150]:
                    baseline_data.append({
                        'task': 'text',
                        'method': method,
                        'deadline_ms': deadline,
                        'seed': seed,
                        'deadline_hit_rate': np.random.uniform(0.6, 0.95),
                        'accuracy': np.random.uniform(0.80, 0.90),
                        'lat_p50_ms': 50.0 if method == 'StaticSmall' else 120.0,
                        'lat_p95_ms': 80.0 if method == 'StaticSmall' else 180.0
                    })
        
        self.baseline_df = pd.DataFrame(baseline_data)
        save_csv(self.baseline_df, self.test_dir / "baseline_results.csv")
        
        # Create mock planner results
        planner_data = []
        for seed in [42, 43, 44]:
            for threshold in [0.7, 0.8, 0.9]:
                for deadline in [100, 150]:
                    planner_data.append({
                        'task': 'text',
                        'deadline_ms': deadline,
                        'threshold': threshold,
                        'seed': seed,
                        'deadline_hit_rate': np.random.uniform(0.85, 0.95),
                        'accuracy': np.random.uniform(0.85, 0.90),
                        'coverage': threshold * 0.8,
                        'lat_p50_ms': 60.0,
                        'lat_p95_ms': 95.0
                    })
        
        self.planner_df = pd.DataFrame(planner_data)
        save_csv(self.planner_df, self.test_dir / "planner_results.csv")
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)
    
    def test_paired_data_preparation(self):
        """Test paired data preparation for statistical tests."""
        # Filter for one baseline method
        baseline_subset = self.baseline_df[self.baseline_df['method'] == 'StaticSmall']
        
        # For planner, select best threshold per (deadline, seed)
        planner_best = self.planner_df.loc[
            self.planner_df.groupby(['deadline_ms', 'seed'])['deadline_hit_rate'].idxmax()
        ]
        
        # Merge on matching keys
        paired = baseline_subset.merge(
            planner_best,
            on=['task', 'deadline_ms', 'seed'],
            suffixes=('_baseline', '_planner')
        )
        
        # Should have paired data
        self.assertGreater(len(paired), 0)
        self.assertEqual(len(paired), len(paired['seed'].unique()) * len(paired['deadline_ms'].unique()))


class TestEndToEndPipeline(unittest.TestCase):
    """Test end-to-end pipeline flow."""
    
    def test_pipeline_file_dependencies(self):
        """Test that pipeline files have correct dependencies."""
        # Define expected pipeline order
        pipeline_order = [
            '01_profile_latency.py',
            '02_profile_accuracy.py',
            '03_run_baselines.py',
            '04_run_planner.py',
            '05_ablation.py',
            '06_statistical_tests.py',
            '07_pareto_analysis.py',
            '08_workload.py',
            '09_failure_analysis.py',
            '10_make_figures.py'
        ]
        
        # Verify all files exist
        experiments_dir = Path(__file__).parent.parent / "experiments"
        for filename in pipeline_order:
            filepath = experiments_dir / filename
            self.assertTrue(filepath.exists(), f"Missing: {filename}")


if __name__ == '__main__':
    unittest.main()

