import sys
import os
import numpy as np
from pathlib import Path

# Adding the root directory to the systematic path so I can import my source modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.aerodynamics.cst import CSTShapeGenerator
from src.aerodynamics.xfoil_runner import XFoilRunner
from src.aerodynamics.evaluator import evaluate_airfoil

def test_xfoil_integration():
    print("=== XFOIL INTEGRATION TEST (PART B) ===")
    
    # Path to my compiled XFOIL binary
    XFOIL_EXE = "/Users/paulbrocvielle/Downloads/Xfoil-for-Mac-main/bin/xfoil"
    
    # 1. Generating a test airfoil (CST with 6 variables)
    # 3 weights for the lower surface (negative) and 3 for the upper surface (positive)
    w_lower = [-0.15, -0.15, -0.15]
    w_upper = [0.15, 0.15, 0.15]
    
    cst = CSTShapeGenerator()
    coords = cst.generate_airfoil(w_lower, w_upper)
    print(f"[OK] Airfoil generated with {len(coords)} points.")

    # 2. Running XFOIL analysis
    # I'm using the 3° angle specified in the assignment (B.3)
    runner = XFoilRunner(xfoil_path=XFOIL_EXE, timeout=15.0)
    print(f"[...] Launching XFOIL at Alpha = 3.0°...")
    
    results = runner.analyze(coords, reynolds=500000, alpha=3.0)
    
    # 3. Checking the results
    if results['CL'] is not None:
        print("\n=== SUCCESSFUL RESULTS ===")
        print(f"CL (Lift) : {results['CL']}")
        print(f"CD (Drag) : {results['CD']}")
        # Note: If the runner doesn't extract CM yet, I'll need to update it
    else:
        print("\n[ERROR] XFOIL didn't return any data.")
        print("I should check if the binary path is correct and if XQuartz is open.")

def test_evaluator():
    print("\n=== EVALUATOR FUNCTION TEST ===")
    
    # Path to my compiled XFOIL binary
    XFOIL_EXE = "/Users/paulbrocvielle/Downloads/Xfoil-for-Mac-main/bin/xfoil"
    
    # Weights for the test: 3 lower (negative), 3 upper (positive)
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