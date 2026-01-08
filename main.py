import numpy as np
from src.algorithms.pso import PSO
from src.benchmarks.griewank import griewank_function

def main():
    print("Optimization of Griewank Function using PSO")
    print("-------------------------------------------")
    
    # 1. Configuration
    dim = 5
    bounds = (-600, 600)  # Common search space for Griewank
    
    # 2. Initialize PSO
    # We pass dim=5 explicitly, though the default in pso.py is now also 5.
    pso_solver = PSO(
        objective_func=griewank_function,
        bounds=bounds,
        num_particles=100,
        max_iter=500,
        dim=dim,
        w=0.7,
        c1=1.4,
        c2=1.4
    )
    
    # 3. Optimize
    print(f"Running PSO with {pso_solver.n_particles} particles for {pso_solver.max_iter} iterations...")
    best_pos, best_score = pso_solver.optimize()
    
    # 4. Results
    print("\nOptimization Complete!")
    print(f"Global Best Score (Minimum found): {best_score:.10f}")
    print("Global Best Position:")
    print(best_pos)
    
    # Known optimum is at [0, 0, ..., 0] with value 0
    print(f"\nDistance from known optimum (origin): {np.linalg.norm(best_pos):.10f}")

if __name__ == "__main__":
    main()
