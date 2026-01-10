import sys
import os
import numpy as np
from pathlib import Path

# Add project root to system path for module imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.aerodynamics.cst import CSTShapeGenerator
from src.aerodynamics.xfoil_runner import XFoilRunner
from src.aerodynamics.evaluator import evaluate_airfoil

def test_xfoil_integration():
    print("=== XFOIL INTEGRATION TEST (PART B) ===")
    
    # Path to compiled XFOIL binary.
    XFOIL_EXE = "/Users/paulbrocvielle/Downloads/Xfoil-for-Mac-main/bin/xfoil"
    
    # Generate test airfoil using 6 CST variables.
    # Configuration: 3 weights per surface (lower/negative, upper/positive).
    w_lower = [-0.15, -0.15, -0.15]
    w_upper = [0.15, 0.15, 0.15]
    
    cst = CSTShapeGenerator()
    coords = cst.generate_airfoil(w_lower, w_upper)
    print(f"[OK] Airfoil generated with {len(coords)} points.")

    # Execute XFOIL analysis.
    # Use 3.0° angle of attack as specified for Task B.3.
    runner = XFoilRunner(xfoil_path=XFOIL_EXE, timeout=15.0)
    print(f"[...] Launching XFOIL at Alpha = 3.0°...")
    
    results = runner.analyze(coords, reynolds=500000, alpha=3.0)
    
    # Validation of results.
    if results['CL'] is not None:
        print("\n=== SUCCESSFUL RESULTS ===")
        print(f"CL (Lift) : {results['CL']}")
        print(f"CD (Drag) : {results['CD']}")
        # Monitor CM extraction status; update runner if necessary.
    else:
        print("\n[ERROR] XFOIL didn't return any data.")
        print("Verify XFOIL binary path and XQuartz availability.")

def test_evaluator():
    print("\n=== EVALUATOR FUNCTION TEST ===")
    
    # Path to compiled XFOIL binary.
    XFOIL_EXE = "/Users/paulbrocvielle/Downloads/Xfoil-for-Mac-main/bin/xfoil"
    
    # Test weights: 3 lower (negative) and 3 upper (positive).
    weights = [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1]
    
    print(f"Testing weights: {weights}")
    
    results = evaluate_airfoil(
        weights=weights,
        reynolds=500000,
        alpha=3.0,
        xfoil_path=XFOIL_EXE
    )
    
    if results['CL'] is not None:
        print("\n[OK] Evaluator returned results:")
        print(f"  CL: {results['CL']}")
        print(f"  CD: {results['CD']}")
        print(f"  CM: {results['CM']}")
    else:
        print("\n[FAIL] Evaluator returned None (Constraint violation or XFOIL error).")

if __name__ == "__main__":
    test_xfoil_integration()
    test_evaluator()