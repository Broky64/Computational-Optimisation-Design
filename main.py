import numpy as np
import matplotlib.pyplot as plt
import os
from src.algorithms.pso import PSO
from src.benchmarks.griewank import griewank_function

def main():
    print("Optimization of Griewank Function using PSO - Robustness Test")
    print("-----------------------------------------------------------")
    
    # 1. Configuration
    runs = 10
    dim = 5
    bounds = (-600, 600)
    
    # PSO Hyperparameters
    pso_params = {
        'num_particles': 100,
        'max_iter': 500,
        'w': 0.7,
        'c1': 1.4,
        'c2': 1.4,
        'dim': dim
    }

    # Storage for statistics
    best_scores = []
    best_positions = []
    convergence_histories = []

    # 2. Execution Loop
    for run in range(runs):
        print(f"\nRun {run + 1}/{runs}...")
        
        # Initialize PSO
        pso_solver = PSO(
            objective_func=griewank_function,
            bounds=bounds,
            **pso_params
        )
        
        # Optimize
        best_pos, best_score = pso_solver.optimize()
        
        # Store results
        best_scores.append(best_score)
        best_positions.append(best_pos)
        convergence_histories.append(pso_solver.history)
        
        print(f"  -> Run {run + 1} Best Score: {best_score:.10f}")

    # 3. Statistical Analysis
    best_scores_np = np.array(best_scores)
    mean_score = np.mean(best_scores_np)
    std_score = np.std(best_scores_np)
    
    best_idx = np.argmin(best_scores_np)
    best_of_best_score = best_scores_np[best_idx]
    best_of_best_pos = best_positions[best_idx]
    best_history = convergence_histories[best_idx]

    print("\n-----------------------------------------------------------")
    print("Statistical Results (over 10 runs):")
    print(f"Mean Best Score:       {mean_score:.10f}")
    print(f"Std Dev of Best Score: {std_score:.10f}")
    print("-----------------------------------------------------------")
    print(f"'Best of the Best' (Run {best_idx + 1}):")
    print(f"  Score: {best_of_best_score:.10f}")
    print(f"  Position: {best_of_best_pos}")
    print("-----------------------------------------------------------")

    # 4. Visualization
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(best_history, label=f'Best Run (Run {best_idx + 1})', color='blue', linewidth=2)
    plt.yscale('log')
    plt.title('PSO Convergence Curve (Best Run)')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value (log scale)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    output_path = os.path.join('results', 'convergence_best_run.png')
    plt.savefig(output_path)
    print(f"\nConvergence plot saved to: {output_path}")

if __name__ == "__main__":
    main()
