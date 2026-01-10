import matplotlib
matplotlib.use('Agg')  # Mode sans fenêtre graphique (Fichiers uniquement)
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Ajout du dossier src au path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# --- IMPORTS ---
from src.algorithms.pso import PSO
from src.benchmarks.griewank import griewank_function
from src.aerodynamics.evaluator import evaluate_airfoil 
from src.aerodynamics.cst import CSTShapeGenerator

# --- CONFIGURATION ---
XFOIL_PATH = "/Users/paulbrocvielle/Downloads/Xfoil-for-Mac-main/bin/xfoil"

# --- HELPER POUR SAUVEGARDER LES RESULTATS ---
def save_report(filename, content):
    """Ecrit le contenu dans un fichier (écrase l'ancien)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(content)
    print(f"\n[REPORT SAVED] Résultats finaux enregistrés dans : {filename}")

# --- TACHE B.2 : Fonction Objectif Aile ---
def airfoil_objective_function(weights):
    REYNOLDS = 500_000
    ALPHA = 3.0
    
    results = evaluate_airfoil(weights, REYNOLDS, ALPHA, XFOIL_PATH)
    
    if results['CL'] is None: return 1000.0 
    
    cl, cd, cm = results['CL'], results['CD'], results['CM']
    
    if cd < 0.001: return 1000.0 # Garde-fou physique
    
    ld_ratio = cl / cd
    if ld_ratio > 200: return 1000.0 # Garde-fou "Too Good To Be True"

    fitness = -ld_ratio
    
    if cm < -0.1:
        violation = abs(cm - (-0.1))
        penalty = 500.0 + (violation * 1000.0)
        fitness += penalty
        
    return fitness

# --- TACHES ---

def run_task_a():
    print("\n===========================================================")
    print("   TASK A: PSO on Griewank Function (Robustness Test)    ")
    print("===========================================================")
    
    dim = 5
    bounds = (-600, 600)
    runs = 10
    pso_params = {'num_particles': 50, 'max_iter': 300, 'w': 0.9, 'c1': 1.4, 'c2': 1.4, 'dim': dim, 'n_jobs': -1}

    best_scores = []
    best_positions = [] # Stockage des coordonnées
    histories = []

    for r in range(runs):
        print(f"Run {r+1}/{runs}...", end='\r')
        solver = PSO(griewank_function, bounds, **pso_params)
        best_pos, best_score = solver.optimize()
        
        best_scores.append(best_score)
        best_positions.append(best_pos)
        histories.append(solver.history)
        
    print(f"Runs completed.                                      ")
    
    # Calcul des stats
    mean_score = np.mean(best_scores)
    std_score = np.std(best_scores)
    
    # Identification du meilleur run absolu
    best_idx = np.argmin(best_scores)
    best_val = best_scores[best_idx]
    best_pos_overall = best_positions[best_idx] # Les coordonnées gagnantes
    
    # Affichage Terminal
    print(f"[RESULTS A] Mean: {mean_score:.6f} | Std Dev: {std_score:.6f}")
    print(f"Best Position: {np.round(best_pos_overall, 6)}")
    
    # Creation du Rapport Texte
    report = f"""===========================================================
TASK A REPORT: Griewank Function Benchmark
===========================================================
Configuration:
  Runs: {runs}
  Particles: {pso_params['num_particles']}
  Iterations: {pso_params['max_iter']}
  Dimensions: {dim}

STATISTICAL RESULTS:
  Mean Best Score:       {mean_score:.10f}
  Std Dev of Best Score: {std_score:.10f}
  
BEST SOLUTION (Run {best_idx+1}):
  Score:    {best_val:.10f}
  Position: {list(best_pos_overall)}
===========================================================
"""
    save_report("results/task_a_results.txt", report)
    
    # Plot
    plt.figure()
    plt.plot(histories[best_idx])
    plt.yscale('log')
    plt.title('Task A: Griewank Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.grid(True)
    plt.savefig('results/task_a_convergence.png')
    plt.close()

def run_task_b1():
    print("\n=== TASK B.1: Pipeline Test ===")
    test_weights = [-0.15, -0.2, -0.1, 0.2, 0.25, 0.2]
    print("Running single evaluation...")
    res = evaluate_airfoil(test_weights, 500000, 3.0, XFOIL_PATH)
    
    report = f"""=== TASK B.1 REPORT ===
Test Weights: {test_weights}
Reynolds: 500,000 | Alpha: 3.0 deg

RESULTS:
"""
    if res['CL'] is not None:
        msg = f"""  [SUCCESS]
  CL (Lift): {res['CL']:.4f}
  CD (Drag): {res['CD']:.5f}
  CM (Moment): {res['CM']:.4f}
  L/D Ratio: {(res['CL']/res['CD']):.2f}
"""
        print(f"[SUCCESS] L/D: {res['CL']/res['CD']:.2f}")
    else:
        msg = "  [FAIL] XFOIL Error or Geometric Constraint Violation.\n"
        print("[FAIL] XFOIL Error")
        
    report += msg
    save_report("results/task_b1_results.txt", report)

def run_task_b3():
    print("\n===========================================================")
    print("   TASK B.3: Airfoil Shape Optimization (PSO)            ")
    print("===========================================================")
    
    bounds = [(-0.6, 0.0), (-0.6, 0.0), (-0.6, 0.0), (0.0, 0.6), (0.0, 0.6), (0.0, 0.6)]
    runs = 5 
    pso_params = {'num_particles': 20, 'max_iter': 30, 'w': 0.9, 'c1': 1.4, 'c2': 1.4, 'n_jobs': -1}
    
    print(f"Config: {runs} Runs | {pso_params['num_particles']} Particles")
    
    all_scores = []
    all_weights = []
    all_histories = []
    
    for r in range(runs):
        print(f"\n--- Run {r+1}/{runs} ---")
        solver = PSO(airfoil_objective_function, bounds, **pso_params)
        w_opt, s_opt = solver.optimize()
        
        all_scores.append(s_opt)
        all_weights.append(w_opt)
        all_histories.append(solver.history)
        print(f"-> Run {r+1} Finished. Score: {s_opt:.4f}")

    best_idx = np.argmin(all_scores)
    best_weights = all_weights[best_idx]
    best_score = all_scores[best_idx]
    
    # Analyse de la meilleure géométrie
    final_res = evaluate_airfoil(best_weights, 500000, 3.0, XFOIL_PATH)
    ld = 0.0
    cm = 0.0
    if final_res['CL']:
        ld = final_res['CL'] / final_res['CD']
        cm = final_res['CM']
        
        # Sauvegarde DAT
        cst = CSTShapeGenerator()
        coords = cst.generate_airfoil(best_weights[:3], best_weights[3:], n_points=150)
        fname = "results/optimized_airfoil_B3.dat"
        with open(fname, 'w') as f:
            f.write(f"Optimized Airfoil (Run {best_idx+1}) L/D={ld:.2f}\n")
            for x, y in coords: f.write(f" {x:.6f}  {y:.6f}\n")
        
        # Plot Convergence
        plt.figure(figsize=(10, 6))
        plt.plot(all_histories[best_idx], label=f'Best Run ({best_idx+1})', linewidth=2)
        for i, h in enumerate(all_histories):
            if i != best_idx: plt.plot(h, color='gray', alpha=0.3)
        plt.xlabel('Iteration'); plt.ylabel('Negative L/D')
        plt.title(f'Airfoil Optimization Convergence ({runs} runs)')
        plt.legend(); plt.grid(True)
        plt.savefig('results/task_b3_convergence.png')
        plt.close()

    # --- CONSTRUCTION DU RAPPORT FINAL (Clean) ---
    report = f"""===========================================================
TASK B.3 REPORT: Airfoil Shape Optimization
===========================================================
Configuration:
  Runs: {runs}
  Particles: {pso_params['num_particles']}
  Iterations: {pso_params['max_iter']}
  Constraint: CM >= -0.1

STATISTICAL RESULTS (on {runs} runs):
  Mean Best Score:       {np.mean(all_scores):.4f}
  Std Dev of Score:      {np.std(all_scores):.4f}
  Best Overall Score:    {best_score:.4f} (Run {best_idx+1})

BEST SOLUTION DETAILS:
  Weights (Lower, Upper): {list(best_weights)}
  Final L/D Ratio:        {ld:.2f}
  Moment Coeff (CM):      {cm:.4f}

NOTE: Use these weights for Task C.1 (Robustness Analysis).
===========================================================
"""
    save_report("results/task_b3_results.txt", report)
    print("Scores finaux:", [f"{s:.2f}" for s in all_scores])

def run_task_b4():
    print("\n===========================================================")
    print("   TASK B.4: Sensitivity Analysis (Swarm Size)           ")
    print("===========================================================")
    
    swarm_sizes = [10, 20, 40]
    colors = ['blue', 'green', 'red']
    bounds = [(-0.6, 0.0), (-0.6, 0.0), (-0.6, 0.0), (0.0, 0.6), (0.0, 0.6), (0.0, 0.6)]
    
    report_lines = []
    
    plt.figure(figsize=(10, 6))
    
    for i, n_particles in enumerate(swarm_sizes):
        print(f"Testing Swarm Size: {n_particles}...")
        pso_params = {'num_particles': n_particles, 'max_iter': 20, 'w': 0.9, 'c1': 1.4, 'c2': 1.4, 'n_jobs': -1}
        
        solver = PSO(airfoil_objective_function, bounds, **pso_params)
        _, best_score = solver.optimize()
        
        res_str = f"Swarm Size {n_particles}: Best Score = {best_score:.4f}"
        print(f"-> {res_str}")
        report_lines.append(res_str)
        
        plt.plot(solver.history, label=f'Population = {n_particles}', color=colors[i], marker='o')

    plt.xlabel('Iteration'); plt.ylabel('Best Fitness')
    plt.title('Population Size vs Convergence')
    plt.grid(True); plt.legend()
    plt.savefig('results/task_b4_sensitivity_analysis.png')
    plt.close()
    
    # Sauvegarde Rapport
    report = """===========================================================
TASK B.4 REPORT: Sensitivity Analysis
===========================================================
Objective: Compare convergence for different swarm sizes.

RESULTS:
""" + "\n".join(report_lines) + "\n===========================================================\n"
    
    save_report("results/task_b4_results.txt", report)

def run_task_c1():
    print("\n===========================================================")
    print("   TASK C.1: Robustness Analysis (Uncertainty on Alpha)  ")
    print("===========================================================")
    
    # --- A REMPLIR AVEC VOS RESULTATS DE B.3 ---
    # (J'ai remis vos poids ici pour vous faire gagner du temps)
    best_weights = [0.0, 0.0, 0.0, 0.1448442576084017, 0.15637738799845433, 0.2898776828706623]
    
    if len(best_weights) != 6:
        print("[ERROR] 'best_weights' est vide ! Veuillez copier les poids depuis 'results/task_b3_results.txt'")
        return

    alpha_mean = 3.0
    alpha_std = 0.1
    n_samples = 100
    
    print(f"Simulating {n_samples} flights (Alpha ~ N({alpha_mean}, {alpha_std}))...")
    alphas = np.random.normal(alpha_mean, alpha_std, n_samples)
    
    cl_res = []
    cd_res = []
    valid_count = 0
    
    for i, alpha_val in enumerate(alphas):
        res = evaluate_airfoil(best_weights, 500_000, alpha_val, XFOIL_PATH)
        
        # --- CORRECTION DU BUG DIV/0 ---
        # On vérifie que CL et CD existent ET que CD n'est pas nul
        if res['CL'] is not None and res['CD'] is not None and res['CD'] > 1e-6:
            cl_res.append(res['CL'])
            cd_res.append(res['CD'])
            valid_count += 1
            
        if i % 10 == 0: print(f"Sample {i}/{n_samples}...", end='\r')
        
    print(f"\nSimulation done. Valid runs: {valid_count}/{n_samples}")
    
    if valid_count > 0:
        # Conversion en numpy array pour calcul vectoriel
        cl_arr = np.array(cl_res)
        cd_arr = np.array(cd_res)
        ld_vals = cl_arr / cd_arr
        
        # Double sécurité : on filtre les infinis s'il en reste
        mask = np.isfinite(ld_vals)
        ld_vals = ld_vals[mask]
        cl_arr = cl_arr[mask]
        cd_arr = cd_arr[mask]
        
        # Stats
        ld_mean, ld_std = np.mean(ld_vals), np.std(ld_vals)
        cl_mean, cl_std = np.mean(cl_arr), np.std(cl_arr)
        cd_mean, cd_std = np.mean(cd_arr), np.std(cd_arr)
        
        # Rapport
        report = f"""===========================================================
TASK C.1 REPORT: Robustness Analysis (Monte Carlo)
===========================================================
Input Weights: {best_weights}
Parameters: Alpha ~ N({alpha_mean}, {alpha_std}) | Samples: {n_samples}
Valid Samples: {len(ld_vals)}/{n_samples}

ROBUSTNESS STATISTICS:
  L/D RATIO:
    Mean:    {ld_mean:.4f}
    Std Dev: {ld_std:.4f}  <-- Key Metric for Robustness

  LIFT (CL):
    Mean:    {cl_mean:.4f}
    Std Dev: {cl_std:.4f}

  DRAG (CD):
    Mean:    {cd_mean:.6f}
    Std Dev: {cd_std:.6f}
===========================================================
"""
        save_report("results/task_c1_results.txt", report)
        
        # Plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.hist(cl_arr, bins=15); plt.title('CL Dist')
        plt.subplot(1, 3, 2); plt.hist(cd_arr, bins=15); plt.title('CD Dist')
        plt.subplot(1, 3, 3); plt.hist(ld_vals, bins=15); plt.title('L/D Dist')
        plt.tight_layout()
        plt.savefig('results/task_c1_robustness.png')
        plt.close()
    else:
        print("No valid runs.")
def main():
    while True:
        print("\n--- MAIN MENU ---")
        print("1. Run Task A (Griewank)")
        print("2. Run Task B.1 (Test Pipeline)")
        print("3. Run Task B.3 (Full Optimization)")
        print("4. Run Task B.4 (Sensitivity Analysis)")
        print("5. Run Task C.1 (Robustness Check)")
        print("0. Exit")
        
        c = input("Choice: ")
        if c == '1': run_task_a()
        elif c == '2': run_task_b1()
        elif c == '3': run_task_b3()
        elif c == '4': run_task_b4()
        elif c == '5': run_task_c1()
        elif c == '0': break

if __name__ == "__main__":
    main()