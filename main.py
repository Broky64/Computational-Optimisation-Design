import matplotlib
matplotlib.use('Agg')  # Empêche l'affichage de fenêtres bloquantes (Mode fichier uniquement)
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
# Chemin XFOIL (Vérifiez qu'il est correct pour votre machine)
XFOIL_PATH = "/Users/paulbrocvielle/Downloads/Xfoil-for-Mac-main/bin/xfoil"

# --- TACHE B.2 : Fonction Objectif Aile ---
def airfoil_objective_function(weights):
    """
    Minimiser (-L/D) sous contrainte CM >= -0.1
    """
    REYNOLDS = 500_000
    ALPHA = 3.0
    
    # 1. Appel XFOIL
    results = evaluate_airfoil(weights, REYNOLDS, ALPHA, XFOIL_PATH)
    
    # 2. Gestion échecs XFOIL
    if results['CL'] is None:
        return 1000.0 
    
    cl = results['CL']
    cd = results['CD']
    cm = results['CM']
    
    # --- GARDE-FOU PHYSIQUE (Correction du bug L/D infini) ---
    # Une traînée inférieure à 0.001 est impossible à ce Reynolds.
    # Si XFOIL renvoie ça, c'est un artefact numérique -> On rejette.
    if cd < 0.001:
        return 1000.0
        
    # 3. Fitness (Maximiser L/D => Minimiser -L/D)
    ld_ratio = cl / cd
    
    # Second garde-fou : Si L/D > 200, c'est "trop beau pour être vrai"
    if ld_ratio > 200:
        return 1000.0

    fitness = -ld_ratio
    
    # 4. Contrainte de Moment (CM >= -0.1)
    if cm < -0.1:
        violation = abs(cm - (-0.1))
        penalty = 500.0 + (violation * 1000.0)
        fitness += penalty
        
    return fitness

# --- TACHES ---

def run_task_a():
    """Tâche A.3 : Optimisation Griewank (10 runs)"""
    print("\n===========================================================")
    print("   TASK A: PSO on Griewank Function (Robustness Test)    ")
    print("===========================================================")
    
    dim = 5
    bounds = (-600, 600)
    runs = 10
    
    # Paramètres (rapides pour Griewank)
    pso_params = {
        'num_particles': 50, 
        'max_iter': 300, 
        'w': 0.9, 
        'c1': 1.4, 
        'c2': 1.4, 
        'dim': dim,
        'n_jobs': -1 # Parallélisme activé
    }

    best_scores = []
    histories = []

    for r in range(runs):
        print(f"Run {r+1}/{runs}...")
        solver = PSO(griewank_function, bounds, **pso_params)
        _, best_score = solver.optimize()
        best_scores.append(best_score)
        histories.append(solver.history)
    
    # Stats
    mean_score = np.mean(best_scores)
    std_score = np.std(best_scores)
    best_idx = np.argmin(best_scores)
    
    print(f"\n[RESULTS A] Mean: {mean_score:.6f} | Std Dev: {std_score:.6f}")
    print(f"Best Run: {best_scores[best_idx]:.6f}")
    
    # Plot
    plt.figure()
    plt.plot(histories[best_idx])
    plt.yscale('log')
    plt.title('Task A: Griewank Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.grid(True)
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/task_a_convergence.png')
    plt.close() # Ferme la figure pour libérer la mémoire
    print("Graph saved to results/task_a_convergence.png")

def run_task_b1():
    """Tâche B.1 : Test Unitaire Pipeline"""
    print("\n=== TASK B.1: Pipeline Test ===")
    test_weights = [-0.15, -0.2, -0.1, 0.2, 0.25, 0.2]
    print("Running single evaluation...")
    res = evaluate_airfoil(test_weights, 500000, 3.0, XFOIL_PATH)
    
    if res['CL'] is not None:
        print(f"[SUCCESS] L/D: {res['CL']/res['CD']:.2f}")
    else:
        print("[FAIL] XFOIL Error")

def run_task_b3():
    """
    Tâche B.3 : Optimisation Complète Aile (Multi-runs)
    """
    print("\n===========================================================")
    print("   TASK B.3: Airfoil Shape Optimization (PSO)            ")
    print("===========================================================")
    
    # 1. Paramètres
    bounds = [
        (-0.6, 0.0), (-0.6, 0.0), (-0.6, 0.0), # Lower
        (0.0, 0.6),  (0.0, 0.6),  (0.0, 0.6)   # Upper
    ]
    
    runs = 5  # Nombre de répétitions pour les stats

    # Configuration PSO pour la Tâche B.3
    pso_params = {
        'num_particles': 20, 
        'max_iter': 30,
        'w': 0.9,
        'c1': 1.4,
        'c2': 1.4,
        'n_jobs': -1  # <--- C'EST CETTE LIGNE QUI ACTIVE LE PARALLÉLISME
    }
    
    print(f"Config: {runs} Runs | {pso_params['num_particles']} Particles | {pso_params['max_iter']} Iterations")
    print("Starting optimization loop...")
    
    all_scores = []
    all_weights = []
    all_histories = []
    
    # 2. Boucle de runs
    for r in range(runs):
        print(f"\n--- Run {r+1}/{runs} ---")
        solver = PSO(
            objective_func=airfoil_objective_function,
            bounds=bounds,
            **pso_params
        )
        
        w_opt, s_opt = solver.optimize()
        
        all_scores.append(s_opt)
        all_weights.append(w_opt)
        all_histories.append(solver.history)
        print(f"-> Run {r+1} Finished. Score: {s_opt:.4f}")

    # 3. Statistiques & Meilleur Résultat
    all_scores_np = np.array(all_scores)
    best_idx = np.argmin(all_scores_np)
    best_weights = all_weights[best_idx]
    best_score = all_scores_np[best_idx]
    
    print("\n===========================================================")
    print("   TASK B.3 STATISTICAL RESULTS")
    print("===========================================================")
    print(f"Mean Best Score:       {np.mean(all_scores_np):.4f}")
    print(f"Std Dev of Score:      {np.std(all_scores_np):.4f}")
    print(f"Best Overall Score:    {best_score:.4f} (Run {best_idx+1})")
    
    # 4. Sauvegarde du Meilleur Profil
    final_res = evaluate_airfoil(best_weights, 500000, 3.0, XFOIL_PATH)
    
    if final_res['CL']:
        ld = final_res['CL'] / final_res['CD']
        print(f"\n[BEST GEOMETRY ANALYSIS]")
        print(f"  Final L/D Ratio: {ld:.2f}")
        print(f"  Moment Coeff:    {final_res['CM']:.4f}")
        
        # Fichier DAT
        cst = CSTShapeGenerator()
        coords = cst.generate_airfoil(best_weights[:3], best_weights[3:], n_points=150)
        
        os.makedirs('results', exist_ok=True)
        fname = "results/optimized_airfoil_B3.dat"
        with open(fname, 'w') as f:
            f.write(f"Optimized Airfoil (Run {best_idx+1}) L/D={ld:.2f}\n")
            for x, y in coords:
                f.write(f" {x:.6f}  {y:.6f}\n")
        print(f"  Geometry saved to: {fname}")
        
        # Plot de convergence (Meilleur Run)
        plt.figure(figsize=(10, 6))
        plt.plot(all_histories[best_idx], label=f'Best Run ({best_idx+1})', linewidth=2)
        # On peut aussi afficher les autres en transparence légère
        for i, h in enumerate(all_histories):
            if i != best_idx:
                plt.plot(h, color='gray', alpha=0.3)
                
        plt.xlabel('Iteration')
        plt.ylabel('Negative L/D')
        plt.title(f'Airfoil Optimization Convergence ({runs} runs)')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/task_b3_convergence.png')
        plt.close()
        print("  Convergence plot saved to: results/task_b3_convergence.png")
        
    else:
        print("Error: Could not validate the best solution.")

def main():
    while True:
        print("\n--- MAIN MENU ---")
        print("1. Run Task A (Griewank)")
        print("2. Run Task B.1 (Test Pipeline)")
        print("3. Run Task B.3 (Full Optimization - 5 Runs)")
        print("0. Exit")
        
        c = input("Choice: ")
        if c == '1': run_task_a()
        elif c == '2': run_task_b1()
        elif c == '3': run_task_b3()
        elif c == '0': break

if __name__ == "__main__":
    main()