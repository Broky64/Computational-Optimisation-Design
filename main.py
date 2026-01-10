import numpy as np
import matplotlib.pyplot as plt
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
# Mettez votre chemin XFOIL exact ici
XFOIL_PATH = "/Users/paulbrocvielle/Downloads/Xfoil-for-Mac-main/bin/xfoil"

# --- TACHE B.2 : Formulation du problème ---
def airfoil_objective_function(weights):
    """
    Fonction objectif pour l'optimisation du profil.
    Objectif : Maximiser L/D (donc Minimiser -L/D)
    Contrainte : CM >= -0.1
    """
    # Paramètres imposés
    REYNOLDS = 500_000
    ALPHA = 3.0
    
    # 1. Évaluation XFOIL
    # On passe XFOIL_PATH via la variable globale pour simplifier l'appel PSO
    results = evaluate_airfoil(weights, REYNOLDS, ALPHA, XFOIL_PATH)
    
    # 2. Gestion des échecs (Pénalité très forte)
    if results['CL'] is None:
        return 1000.0 
    
    cl = results['CL']
    cd = results['CD']
    cm = results['CM']
    
    # Sécurité anti-division par zéro
    if cd <= 1e-7:
        return 1000.0
        
    # 3. Calcul du score (Fitness)
    # On veut maximiser L/D, donc on minimise l'opposé.
    # L/D varie souvent entre 20 et 100. Donc fitness entre -20 et -100.
    fitness = -(cl / cd)
    
    # 4. Application des contraintes (Penalty Method)
    # Contrainte : Le moment ne doit pas être trop piqueur (CM >= -0.1)
    if cm < -0.1:
        # Pénalité = Valeur fixe + proportionnelle à la violation
        violation = abs(cm - (-0.1))
        penalty = 500.0 + (violation * 1000.0)
        fitness += penalty
        
    return fitness

# --- TACHES ---

def run_task_a():
    """Tâche A.3 : Optimisation Griewank"""
    print("\n=== TASK A: PSO on Griewank Function ===")
    
    dim = 5
    bounds = (-600, 600)
    pso_params = {'num_particles': 50, 'max_iter': 300, 'w': 0.9, 'c1': 1.4, 'c2': 1.4, 'dim': dim}

    solver = PSO(griewank_function, bounds, **pso_params)
    best_pos, best_score = solver.optimize()
    
    print(f"\n[RESULT] Best Score: {best_score:.6f}")
    
    # Plot rapide
    plt.figure()
    plt.plot(solver.history)
    plt.yscale('log')
    plt.title('Griewank Convergence')
    plt.savefig('results/task_a_plot.png')
    print("Graph saved to results/task_a_plot.png")

def run_task_b1():
    """Tâche B.1 : Test Unitaire Pipeline"""
    print("\n=== TASK B.1: Pipeline Test ===")
    test_weights = [-0.15, -0.2, -0.1, 0.2, 0.25, 0.2]
    res = evaluate_airfoil(test_weights, 500000, 3.0, XFOIL_PATH)
    
    if res['CL'] is not None:
        print(f"[SUCCESS] L/D: {res['CL']/res['CD']:.2f}")
    else:
        print("[FAIL] XFOIL Error")

def run_task_b3():
    """
    Tâche B.3 : Exécution de l'optimisation aérodynamique.
    """
    print("\n===========================================================")
    print("   TASK B.3: Airfoil Shape Optimization (PSO)            ")
    print("===========================================================")
    
    # 1. Définition des bornes (6 variables)
    # Poids Intrados (Lower) : négatifs [-0.6, 0.0]
    # Poids Extrados (Upper) : positifs [0.0, 0.6]
    bounds = [
        (-0.6, 0.0), (-0.6, 0.0), (-0.6, 0.0), # Lower
        (0.0, 0.6),  (0.0, 0.6),  (0.0, 0.6)   # Upper
    ]
    
    # 2. Configuration PSO
    # NOTE : XFOIL est lent. On réduit la population et les itérations par rapport à Griewank.
    # 20 particules * 30 itérations = 600 appels XFOIL (environ 10-15 minutes selon PC)
    pso_params = {
        'num_particles': 20, 
        'max_iter': 3,
        'w': 0.9,    # Inertie adaptative gérée dans pso.py
        'c1': 1.4,
        'c2': 1.4
    }
    
    print(f"Configuration: {pso_params['num_particles']} particles, {pso_params['max_iter']} iterations.")
    print("Starting optimization (this may take time)...")
    
    # 3. Lancement
    solver = PSO(
        objective_func=airfoil_objective_function,
        bounds=bounds,
        **pso_params
    )
    
    best_weights, best_score = solver.optimize()
    
    # 4. Résultats
    print("\n=== OPTIMIZATION FINISHED ===")
    print(f"Best Objective Score: {best_score:.4f}")
    print(f"Best Weights: {np.round(best_weights, 4)}")
    
    # Ré-évaluation finale pour affichage propre
    final_res = evaluate_airfoil(best_weights, 500000, 3.0, XFOIL_PATH)
    if final_res['CL']:
        ld = final_res['CL'] / final_res['CD']
        print(f"Final L/D Ratio: {ld:.2f}")
        print(f"Moment Coeff:    {final_res['CM']:.4f}")
        
        # Sauvegarde du profil
        cst = CSTShapeGenerator()
        coords = cst.generate_airfoil(best_weights[:3], best_weights[3:], n_points=150)
        
        fname = "results/optimized_airfoil.dat"
        os.makedirs('results', exist_ok=True)
        with open(fname, 'w') as f:
            f.write(f"Optimized Airfoil L/D={ld:.2f}\n")
            for x, y in coords:
                f.write(f" {x:.6f}  {y:.6f}\n")
        print(f"Geometry saved to {fname}")
        
        # Plot Convergence
        plt.figure()
        plt.plot(solver.history)
        plt.xlabel('Iteration')
        plt.ylabel('Negative L/D')
        plt.title('Airfoil Optimization Convergence')
        plt.grid(True)
        plt.savefig('results/task_b3_convergence.png')
        print("Convergence plot saved.")
        
    else:
        print("Error: Could not validate the best solution.")

def main():
    while True:
        print("\n--- MAIN MENU ---")
        print("1. Run Task A (Griewank)")
        print("2. Run Task B.1 (Test Pipeline)")
        print("3. Run Task B.3 (Full Optimization)")
        print("0. Exit")
        
        c = input("Choice: ")
        if c == '1': run_task_a()
        elif c == '2': run_task_b1()
        elif c == '3': run_task_b3()
        elif c == '0': break

if __name__ == "__main__":
    main()