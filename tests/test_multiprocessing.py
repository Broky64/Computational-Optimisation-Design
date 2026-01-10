import time
import multiprocessing
import os
import sys

# Get absolute path to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.algorithms.pso import PSO

def expensive_function(x):
    """
    A dummy function that takes some time to compute
    to make parallelization visible/necessary.
    """
    # Simulate heavy computation
    _ = [i**2 for i in range(100000)]
    return sum(x**2)

if __name__ == "__main__":
    print(f"CPU count: {multiprocessing.cpu_count()}")
    
    # 1. Sequential Run
    print("\n--- Sequential Run (n_jobs=1) ---")
    start_time = time.time()
    pso_seq = PSO(
        objective_func=expensive_function,
        bounds=(-10, 10),
        num_particles=20,
        max_iter=5,
        dim=5,
        n_jobs=1
    )
    pso_seq.optimize()
    seq_duration = time.time() - start_time
    print(f"Sequential Duration: {seq_duration:.4f} seconds")

    # 2. Parallel Run
    print("\n--- Parallel Run (n_jobs=-1) ---")
    start_time = time.time()
    pso_par = PSO(
        objective_func=expensive_function,
        bounds=(-10, 10),
        num_particles=20,
        max_iter=5,
        dim=5,
        n_jobs=-1
    )
    pso_par.optimize()
    par_duration = time.time() - start_time
    print(f"Parallel Duration: {par_duration:.4f} seconds")
    
    print("\n--- Summary ---")
    print(f"Speedup: {seq_duration / par_duration:.2f}x")
    if par_duration < seq_duration:
        print("PASS: Parallel execution was faster.")
    else:
        print("FAIL: Parallel execution was slower (overhead might dominate for small tasks).")
