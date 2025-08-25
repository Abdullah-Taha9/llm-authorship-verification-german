#!/usr/bin/env python3
"""
Batch runner for multiple authorship verification experiments.
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml

def run_experiment(config_path: str):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {config_path}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, "authorship_verification/authorship_verification.py", 
            "--config", config_path
        ], check=True, capture_output=True, text=True)
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment {config_path}:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    """Run all experiments in configs/examples/ directory."""
    examples_dir = Path("configs/examples")
    
    if not examples_dir.exists():
        print(f"Examples directory {examples_dir} not found!")
        return
    
    config_files = list(examples_dir.glob("*.yaml"))
    
    if not config_files:
        print(f"No YAML configuration files found in {examples_dir}")
        return
    
    print(f"Found {len(config_files)} configuration files:")
    for config_file in config_files:
        print(f"  - {config_file}")
    
    successful = 0
    failed = 0
    
    for config_file in config_files:
        if run_experiment(str(config_file)):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print("BATCH EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(config_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
