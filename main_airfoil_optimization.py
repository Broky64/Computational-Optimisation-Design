
"""
Task B.2 & B.3: Airfoil Shape Optimization using PSO.
Author: Optimization Engineer
Description: 
    Optimizes the CST weights of an airfoil to maximize the Lift-to-Drag ratio (L/D)
    under a Moment Coefficient (CM) constraint.
"""

import sys
import os
import numpy as np

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.algorithms.pso import PSO
from src.aerodynamics.evaluator import evaluate_airfoil
from src.aerodynamics.cst import CSTShapeGenerator

# Configuration
XFOIL_PATH = "/Users/paulbrocvielle/Downloads/Xfoil-for-Mac-main/bin/xfoil"
REYNOLDS = 500000
ALPHA = 3.0

def airfoil_objective_function(weights):
    """
    Objective function for the PSO optimizer.
    Minimize: -(CL / CD)  [equivalent to maximizing L/D]
    Subject to: CM >= -0.1
    """
    
    # 1. Evaluate Aerodynamics
    results = evaluate_airfoil(weights, REYNOLDS, ALPHA, XFOIL_PATH)
    
    # 2. Handle Failures (Geometric or XFOIL convergence)
    if results['CL'] is None:
        return 1000.0  # High penalty for failure
    
    cl = results['CL']
    cd = results['CD']
    cm = results['CM']
    
    # Avoid division by zero or negative drag (unlikely but safe)
    if cd <= 1e-6:
        return 1000.0
        
    # 3. Calculate Base Fitness (Minimize Negative L/D)
    # L/D is typically 50-100. So fitness will be -50 to -100.
    fitness = -(cl / cd)
    
    # 4. Apply Constraints
    # Constraint: CM should not be too negative (CM >= -0.1)
    if cm < -0.1:
        # Penalty proportional to violation + fixed amount to ensure it's worse than valid solutions
        violation = abs(cm - (-0.1))
        penalty = 100.0 + (violation * 1000.0)
        fitness += penalty
        
    return fitness

def main():
    print("=== Airfoil Optimization (Task B.2 & B.3) ===")
    print("Optimization Engineer: Initialize PSO Setup...")
    
    # 1. Define Design Variable Bounds
    # Lower Surface Weights (3 variables): [-0.6, 0.1]
    # Upper Surface Weights (3 variables): [0.1, 0.6]
    lower_bounds_range = (-0.6, 0.1)
    upper_bounds_range = (0.1, 0.6)
    
    bounds = [
        lower_bounds_range, lower_bounds_range, lower_bounds_range, # w_lower 1, 2, 3
        upper_bounds_range, upper_bounds_range, upper_bounds_range  # w_upper 1, 2, 3
    ]
    
    # 2. Initialize PSO
    print("Configuration:")
    print(f"  - Particles: 15")
    print(f"  - Iterations: 20")
    print(f"  - Objective: Maximize L/D (minimize -L/D)")
    print(f"  - Constraint: CM >= -0.1")
    
    optimizer = PSO(
        objective_func=airfoil_objective_function,
        bounds=bounds,
        num_particles=15,
        max_iter=20,
        w=0.7,
        c1=1.4,
        c2=1.4
    )
    
    # 3. Run Optimization
    print("\nStarting Optimization Run...")
    best_weights, best_score = optimizer.optimize()
    
    # 4. Post-Process Results
    print("\n=== Optimization Complete ===")
    print(f"Best Weights: {np.round(best_weights, 4)}")
    print(f"Best Objective Score: {best_score:.4f}")
    
    # Re-evaluate to get coefficients
    final_res = evaluate_airfoil(best_weights, REYNOLDS, ALPHA, XFOIL_PATH)
    
    if final_res['CL'] is not None:
        cl, cd, cm = final_res['CL'], final_res['CD'], final_res['CM']
        print("\nFinal Aerodynamic Performance:")
        print(f"  CL : {cl:.4f}")
        print(f"  CD : {cd:.5f}")
        print(f"  CM : {cm:.4f}")
        print(f"  L/D: {(cl/cd):.2f}")
        
        # Check constraint satisfaction
        if cm < -0.1:
            print("  [WARNING] CM Constraint violated!")
        else:
            print("  [OK] CM Constraint satisfied.")
            
        # 5. Save Coordinates
        print("\nSaving best airfoil geometry to 'best_airfoil_B3.dat'...")
        cst = CSTShapeGenerator()
        # Parse weights again for generation
        w_lower = best_weights[:3]
        w_upper = best_weights[3:]
        coords = cst.generate_airfoil(w_lower, w_upper, n_points=120)
        
        output_file = "best_airfoil_B3.dat"
        with open(output_file, "w") as f:
            f.write(f"Optimization Result (L/D={(cl/cd):.2f})\n")
            for x, y in coords:
                f.write(f" {x:.6f}  {y:.6f}\n")
        print(f"File saved: {os.path.abspath(output_file)}")
        
    else:
        print("\n[ERROR] Failed to re-evaluate the best solution. It might be unstable.")
    
if __name__ == "__main__":
    main()
