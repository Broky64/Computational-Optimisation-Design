import numpy as np
import joblib
import pandas as pd
import os
import time
from src.algorithms.pso import PSO
from src.aerodynamics.evaluator import evaluate_airfoil

def run_surrogate_optimization(xfoil_path):
    """
    Task C.4: Optimization using a surrogate model to demonstrate significant computational speedup compared to XFOIL.
    """
    model_path = "results/surrogate_model.pkl"
    print(f"\n[ANALYSIS] Starting Optimization using Surrogate Model...")
    
    # Load surrogate model
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Run Option 7 first!")
        return
        
    model = joblib.load(model_path)
    print("  Model loaded successfully.")
    
    # Objective function definition using the surrogate model for rapid evaluation.
    def surrogate_objective(weights):
        # Format input weights as a DataFrame for model prediction.
        # Weights vector: [w1, w2, w3, w4, w5, w6]
        X = np.array(weights).reshape(1, -1)
        X_df = pd.DataFrame(X, columns=['w1','w2','w3','w4','w5','w6'])
        
        # Surrogate model prediction
        predicted_ld = model.predict(X_df)[0]
        
        # Convert L/D maximization to minimization.
        return -predicted_ld

    # PSO configuration
    # High iteration count is feasible due to the speed of the surrogate model.
    dim = 6
    bounds = [(-0.6, 0.0)]*3 + [(0.0, 0.6)]*3
    
    pso_params = {
        'num_particles': 50, 
        'max_iter': 100,      
        'w': 0.7, 'c1': 1.4, 'c2': 1.4,
        'dim': dim,
        'n_jobs': 1 # Sequential execution is sufficient given the rapid evaluation time.
    }
    
    print("  Launching PSO on Surrogate Model (50 particles, 100 iter)...")
    
    # Optimization execution and timing
    start_time = time.time()
    
    solver = PSO(surrogate_objective, bounds, **pso_params)
    best_weights, best_score_pred = solver.optimize()
    
    elapsed = time.time() - start_time
    print(f"  Optimization finished in {elapsed:.4f} seconds!")
    
    # Validation against XFOIL ground truth
    print("\n[VERIFICATION] Validating result with real XFOIL...")
    real_res = evaluate_airfoil(best_weights, 500_000, 3.0, xfoil_path)
    
    real_ld = 0.0
    if real_res['CL'] and real_res['CD'] and real_res['CD'] > 0:
        real_ld = real_res['CL'] / real_res['CD']
    
    predicted_ld = -best_score_pred
    error = abs(predicted_ld - real_ld)
    
    # Report generation
    report = f"""===========================================================
TASK C.4 REPORT: Surrogate-Based Optimization
===========================================================
PERFORMANCE:
  Optimization Time:   {elapsed:.4f} seconds
  Iterations:          100
  Particles:           50

RESULTS COMPARISON:
  AI Predicted L/D:    {predicted_ld:.2f}
  Real XFOIL L/D:      {real_ld:.2f}
  
  Prediction Error:    {error:.2f}
  Accuracy:            {100 - (error/real_ld*100):.1f}% (approx)

CONCLUSION:
  The surrogate model allows optimization in real-time.
===========================================================
"""
    os.makedirs('results', exist_ok=True)
    with open("results/task_c4_results.txt", 'w') as f:
        f.write(report)
    print("\n" + report)
    print("[REPORT SAVED] -> results/task_c4_results.txt")