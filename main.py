import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Path configuration
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Imports
from src.algorithms.pso import PSO
from src.benchmarks.griewank import griewank_function
from src.aerodynamics.evaluator import evaluate_airfoil 
from src.aerodynamics.cst import CSTShapeGenerator

# Modular imports
from src.analysis.robustness import run_robustness_analysis
from src.analysis.surrogate import generate_training_data
from src.analysis.model_training import train_surrogate_model
from src.analysis.surrogate_optimization import run_surrogate_optimization

# Import modularisé (Visualisation) <--- NOUVEAU
from src.visualization.plotter import plot_airfoil_from_dat

# Configuration
XFOIL_PATH = "/Users/paulbrocvielle/Downloads/Xfoil-for-Mac-main/bin/xfoil"

# Helper functions
def save_report(filename, content):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f: f.write(content)
    print(f"\n[REPORT SAVED] -> {filename}")

def airfoil_objective_function(weights):
    """Objective function for airfoil optimization (Task B.3)"""
    res = evaluate_airfoil(weights, 500_000, 3.0, XFOIL_PATH)
    
    if res['CL'] is None or res['CD'] is None or res['CD'] < 0.001: return 1000.0
    
    ld = res['CL'] / res['CD']
    if ld > 200: return 1000.0
    
    fitness = -ld
    if res['CM'] < -0.1:
        fitness += 500.0 + (abs(res['CM'] - (-0.1)) * 1000.0)
    return fitness

# Task implementations

def run_task_a():
    print("\n=== TASK A: PSO Benchmark (Griewank) ===")
    dim = 5
    bounds = (-600, 600)
    runs = 10
    params = {'num_particles': 100, 'max_iter': 300, 'w': 0.9, 'c1': 1.4, 'c2': 1.4, 'dim': dim, 'n_jobs': -1}

    best_scores = []
    histories = []
    best_pos_overall = None

    for r in range(runs):
        print(f"Run {r+1}/{runs}...", end='\r')
        solver = PSO(griewank_function, bounds, **params)
        pos, score = solver.optimize()
        best_scores.append(score)
        histories.append(solver.history)
        if score == np.min(best_scores): best_pos_overall = pos
        
    print(f"Runs completed.                                ")
    
    mean, std = np.mean(best_scores), np.std(best_scores)
    best_val = np.min(best_scores)
    pos_str = ", ".join([f"{x:.8f}" for x in best_pos_overall]) if best_pos_overall is not None else "N/A"

    report = f"""===========================================================
TASK A REPORT: Griewank Function Benchmark
===========================================================
CONFIGURATION:
  Runs:           {runs}
  Particles:      {params['num_particles']}
  Iterations:     {params['max_iter']}
  Dimensions:     {dim}

RESULTS STATISTICS:
  Mean Best Score:       {mean:.10f}
  Std Dev Best Score:    {std:.10f}

BEST SOLUTION FOUND:
  Score:                 {best_val:.10f}
  Position:              [{pos_str}]
===========================================================
"""
    save_report("results/task_a_results.txt", report)
    
    plt.figure()
    plt.plot(histories[np.argmin(best_scores)])
    plt.yscale('log'); plt.title('Task A: Griewank Convergence'); plt.grid(True)
    plt.savefig('results/task_a_convergence.png'); plt.close()

def run_task_b1():
    print("\n=== TASK B.1: Pipeline Test ===")
    test_weights = [-0.15, -0.2, -0.1, 0.2, 0.25, 0.2]
    print("Running single evaluation...")
    res = evaluate_airfoil(test_weights, 500000, 3.0, XFOIL_PATH)
    
    report = f"""===========================================================
TASK B.1 REPORT: Pipeline Unit Test
===========================================================
INPUT:
  Test Weights:   {test_weights}
  Reynolds:       500,000
  Alpha:          3.0 deg

RESULTS:
"""
    if res['CL'] is not None:
        report += f"""  Status:         [SUCCESS]
  CL (Lift):      {res['CL']:.4f}
  CD (Drag):      {res['CD']:.5f}
  CM (Moment):    {res['CM']:.4f}
  ---------------------------
  L/D Ratio:      {(res['CL']/res['CD']):.2f}
===========================================================
"""
        print(f"[SUCCESS] L/D: {res['CL']/res['CD']:.2f}")
    else:
        report += """  Status:         [FAIL]
  Reason:         XFOIL Error or Geometric Constraint Violation.
===========================================================
"""
        print("[FAIL] XFOIL Error")
        
    save_report("results/task_b1_results.txt", report)

def run_task_b3():
    print("\n=== TASK B.3: Airfoil Optimization ===")
    bounds = [(-0.6, 0.0)]*3 + [(0.0, 0.6)]*3
    runs = 5 
    params = {'num_particles': 20, 'max_iter': 30, 'w': 0.9, 'c1': 1.4, 'c2': 1.4, 'n_jobs': -1}
    
    all_scores, all_weights, all_histories = [], [], []
    
    for r in range(runs):
        print(f"Run {r+1}/{runs}...")
        solver = PSO(airfoil_objective_function, bounds, **params)
        w, s = solver.optimize()
        all_scores.append(s); all_weights.append(w); all_histories.append(solver.history)
        print(f"  -> Score: {s:.4f}")

    best_idx = np.argmin(all_scores)
    best_w = all_weights[best_idx]
    
    final = evaluate_airfoil(best_w, 500_000, 3.0, XFOIL_PATH)
    ld = final['CL']/final['CD'] if final['CL'] else 0
    
    # Save airfoil geometry
    cst = CSTShapeGenerator()
    coords = cst.generate_airfoil(best_w[:3], best_w[3:], 150)
    with open("results/optimized_airfoil_B3.dat", 'w') as f:
        f.write(f"Optimized (L/D={ld:.2f})\n")
        for x,y in coords: f.write(f" {x:.6f}  {y:.6f}\n")
    
    weights_formatted = ", ".join([f"{x:.6f}" for x in best_w])

    report = f"""===========================================================
TASK B.3 REPORT: Airfoil Shape Optimization
===========================================================
CONFIGURATION:
  Runs:           {runs}
  Particles:      {params['num_particles']}
  Iterations:     {params['max_iter']}
  Constraint:     CM >= -0.1

STATISTICAL RESULTS (on {runs} runs):
  Mean Best Score:       {np.mean(all_scores):.4f}
  Std Dev Score:         {np.std(all_scores):.4f}
  Best Overall Score:    {all_scores[best_idx]:.4f} (Run {best_idx+1})

BEST GEOMETRY ANALYSIS:
  Final L/D Ratio:       {ld:.2f}
  Moment Coeff (CM):     {final['CM']:.4f}
  
  Best Weights (Copy for C.1):
  [{weights_formatted}]
===========================================================
"""
    save_report("results/task_b3_results.txt", report)
    
    # Comprehensive plot of all runs with intelligent scaling
    plt.figure(figsize=(10, 6))
    
    # Plot all optimization runs in light gray for context
    for i, h in enumerate(all_histories):
        if i != best_idx:
            plt.plot(h, color='gray', alpha=0.3, linewidth=1, label='_nolegend_')
    
    # Plot the best optimization run in blue for focus
    best_history = all_histories[best_idx]
    plt.plot(best_history, color='blue', linewidth=2, label=f'Best Run ({best_idx+1})')
    
    # Adjust Y-axis scale to ignore high penalty values from failed calculations
    # Limit the maximum Y-value to 50 if the best run has negative values
    valid_vals = [v for v in best_history if v < 500]  # Filter extreme penalties for axis calculation
    if valid_vals:
        # Provide margin around the best run results
        y_min = min(valid_vals) - 10
        y_max = max(valid_vals) + 10
        # Ensure Y-axis maximum does not exceed 50 to omit high-penalty runs
        y_max = min(y_max, 50) 
        plt.ylim(y_min, y_max)
    
    plt.title(f'Airfoil Optimization Convergence ({runs} runs)')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function (-L/D)')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/task_b3_convergence.png')
    plt.close()

def run_task_b4():
    print("\n=== TASK B.4: Sensitivity Analysis ===")
    sizes = [10, 20, 40]
    bounds = [(-0.6, 0.0)]*3 + [(0.0, 0.6)]*3
    
    plt.figure()
    results_str = ""
    for n in sizes:
        print(f"Testing Pop={n}...")
        params = {'num_particles': n, 'max_iter': 20, 'w': 0.9, 'c1': 1.4, 'c2': 1.4, 'n_jobs': -1}
        solver = PSO(airfoil_objective_function, bounds, **params)
        _, s = solver.optimize()
        
        results_str += f"  Swarm Size {n:<2}:      Best Score = {s:.4f}\n"
        plt.plot(solver.history, label=f'Population = {n}', marker='o')
        
    report = f"""===========================================================
TASK B.4 REPORT: Sensitivity Analysis
===========================================================
OBJECTIVE:
  Compare convergence for different swarm sizes.

RESULTS:
{results_str}
===========================================================
"""
    save_report("results/task_b4_results.txt", report)
    plt.legend(); plt.grid(True); plt.title('Swarm Size Impact')
    plt.savefig('results/sensitivity_analysis_B4.png'); plt.close()

# Module execution wrappers

def launch_c1():
    # Predefined optimal weights for robustness analysis
    best_weights = [0.000000, 0.000000, 0.000000, 0.145422, 0.150759, 0.293384]
    run_robustness_analysis(best_weights, XFOIL_PATH, n_samples=100)

def launch_c3():
    generate_training_data(XFOIL_PATH, n_samples=1000)

def launch_c3_train():
    train_surrogate_model()

def launch_c4():
    run_surrogate_optimization(XFOIL_PATH)

def launch_visualization():
    # Appelle la fonction importée du fichier src/visualization/plotter.py
    dat_path = "results/optimized_airfoil_B3.dat"
    img_path = "results/optimized_airfoil_view.png"
    plot_airfoil_from_dat(dat_path, img_path)

def main():
    while True:
        print("\n--- OPTIMIZATION SUITE ---")
        print("1. Task A (Griewank)")
        print("2. Task B.1 (Test Pipeline)")
        print("3. Task B.3 (Airfoil Opt)")
        print("4. Task B.4 (Sensitivity)")
        print("5. Task C.1 (Robustness) [Via Module]")
        print("6. Task C.3 (Data Gen)   [Via Module]")
        print("7. Task C.3 (Train Model)[Via Module]")
        print("8. Task C.4 (Surrogate Opt)")
        print("9. Task C.5 (Visualization)")
        print("0. Exit")
        
        c = input("Choice: ")
        if c == '1': run_task_a()
        elif c == '2': run_task_b1()
        elif c == '3': run_task_b3()
        elif c == '4': run_task_b4()
        elif c == '5': launch_c1()
        elif c == '6': launch_c3()
        elif c == '7': launch_c3_train()
        elif c == '8': launch_c4()
        elif c == '9': launch_visualization()
        elif c == '0': break

if __name__ == "__main__":
    main()