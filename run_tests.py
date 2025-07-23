#!/usr/bin/env python3
"""
Test runner script for ECG simulator tests.
Provides different test execution modes and reporting options.
"""
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def find_python_executable():
    """Find the correct Python executable."""
    # Try python3 first, then python
    for python_cmd in ["python3", "python"]:
        if shutil.which(python_cmd):
            return python_cmd
    
    # If neither found, use sys.executable
    return sys.executable

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} completed successfully")
        return True

def main():
    parser = argparse.ArgumentParser(description="Run ECG simulator tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (no slow/performance tests)")
    parser.add_argument("--medical", action="store_true", help="Run medical accuracy tests only")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Find the correct Python executable
    python_cmd = find_python_executable()
    print(f"Using Python executable: {python_cmd}")
    
    # Base pytest command
    cmd = [python_cmd, "-m", "pytest"]
    
    if args.verbose:
        cmd.append("-v")
    
    # Add markers based on arguments
    markers = []
    if args.quick:
        markers.append("not slow and not performance")
    elif args.medical:
        markers.append("medical")
    elif args.unit:
        markers.append("unit")
    elif args.integration:
        markers.append("integration")
    elif args.performance:
        markers.append("performance")
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    # Coverage options
    if args.coverage or args.html:
        cmd.extend(["--cov=ecg_simulator", "--cov-report=term-missing"])
        if args.html:
            cmd.extend(["--cov-report=html"])
    
    # Benchmark options
    if args.benchmark:
        cmd.extend(["--benchmark-only", "--benchmark-sort=mean"])
    
    # Add test directory
    cmd.append("tests/")
    
    success = run_command(cmd, "ECG Simulator Tests")
    
    if args.html and (args.coverage or args.html):
        print(f"\n📊 HTML coverage report generated: file://{Path.cwd()}/htmlcov/index.html")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())