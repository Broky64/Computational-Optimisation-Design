import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ajout du dossier src au path si nécessaire
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# --- IMPORTS ---
# Partie A : Algorithmes & Benchmarks
from src.algorithms.pso import PSO
from src.benchmarks.griewank import griewank_function

# Partie B : Aérodynamique
# Note: evaluate_airfoil EST la fonction demandée en B.1
from src.aerodynamics.evaluator import evaluate_airfoil 

# --- CONFIGURATION GLOBALE ---
# Modifiez ce chemin selon votre installation (c'est celui vu dans vos fichiers précédents)
XFOIL_PATH = "/Users/paulbrocvielle/Downloads/Xfoil-for-Mac-main/bin/xfoil"


def run_task_a():
    """
    Tâche A.3 : Optimisation de la fonction de Griewank (Test de robustesse)
    """
    print("\n===========================================================")
    print("   TASK A: Optimisation Solver Benchmark (Griewank)      ")
    print("===========================================================")
    
    # 1. Configuration
    runs = 10
    dim = 5
    bounds = (-600, 600)
    
    # Hyperparamètres PSO (W est géré dynamiquement dans votre classe PSO, mais on passe une valeur init)
    pso_params = {
        'num_particles': 100, # Population suffisante pour Griewank
        'max_iter': 500,
        'w': 0.9,
        'c1': 1.4,
        'c2': 1.4,
        'dim': dim
    }

    best_scores = []
    best_positions = []
    convergence_histories = []

    # 2. Boucle d'exécution
    for run in range(runs):
        print(f"-> Run {run + 1}/{runs}...")
        
        # Initialisation
        pso_solver = PSO(
            objective_func=griewank_function,
            bounds=bounds,
            **pso_params
        )
        
        # Optimisation
        best_pos, best_score = pso_solver.optimize()
        
        # Stockage
        best_scores.append(best_score)
        best_positions.append(best_pos)
        convergence_histories.append(pso_solver.history)

    # 3. Analyse Statistique
    best_scores_np = np.array(best_scores)
    mean_score = np.mean(best_scores_np)
    std_score = np.std(best_scores_np)
    
    best_idx = np.argmin(best_scores_np)
    
    print("\n--- A.3 Results ---")
    print(f"Mean Score: {mean_score:.6f}")
    print(f"Std Dev:    {std_score:.6f}")
    print(f"Best Run ({best_idx+1}): {best_scores_np[best_idx]:.8f}")

    # 4. Visualisation
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_histories[best_idx], label=f'Best Run', color='blue')
    plt.yscale('log')
    plt.title('Task A: PSO Convergence on Griewank')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value (log)')
    plt.grid(True, which="both", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join('results', 'task_a_convergence.png'))
    print("Graph saved to results/task_a_convergence.png")


def run_task_b1():
    """
    Tâche B.1 : Fonction d'automatisation (CST -> XFOIL -> Paramètres).
    Cette fonction teste le pipeline complet sur un profil arbitraire.
    """
    print("\n===========================================================")
    print("   TASK B.1: Airfoil Automation Pipeline Test            ")
    print("===========================================================")

    # 1. Définition d'un profil test (Poids CST)
    # 6 variables : 3 intrados (négatifs), 3 extrados (positifs)
    # Exemple : Un profil symétrique simple aurait des poids nuls ou symétriques.
    # Ici on met un profil un peu cambré pour tester.
    test_weights = [
        -0.15, -0.20, -0.10,  # Lower surface weights (w_lower 1, 2, 3)
         0.20,  0.25,  0.20   # Upper surface weights (w_upper 1, 2, 3)
    ]
    
    reynolds = 500_000
    alpha = 3.0 # Angle d'attaque imposé par le devoir (B.3)

    print(f"Testing inputs:")
    print(f" - Weights: {test_weights}")
    print(f" - Reynolds: {reynolds}")
    print(f" - Alpha: {alpha}°")
    print(f" - XFOIL Path: {XFOIL_PATH}")

    # 2. Appel de la fonction d'automatisation (B.1 logic is inside evaluate_airfoil)
    print("\nRunning XFOIL automation...")
    
    try:
        results = evaluate_airfoil(
            weights=test_weights,
            reynolds=reynolds,
            alpha=alpha,
            xfoil_path=XFOIL_PATH
        )
        
        # 3. Affichage des résultats extraits
        if results['CL'] is not None:
            print("\n[SUCCESS] XFOIL Results Extracted:")
            print(f"  CL (Lift)   : {results['CL']:.4f}")
            print(f"  CD (Drag)   : {results['CD']:.5f}")
            print(f"  CM (Moment) : {results['CM']:.4f}")
            print(f"  L/D Ratio   : {(results['CL']/results['CD']):.2f}")
        else:
            print("\n[FAILURE] XFOIL failed to converge or geometry was invalid.")
            print("Check if XFOIL path is correct and if geometry is not self-intersecting.")
            
    except Exception as e:
        print(f"\n[ERROR] An exception occurred: {e}")
        print("Make sure XFOIL is installed and the path is correct.")


def main():
    while True:
        print("\n--- MAIN MENU ---")
        print("1. Run Task A (Griewank Optimization)")
        print("2. Run Task B.1 (Airfoil Automation Test)")
        print("0. Exit")
        
        choice = input("Select an option: ")
        
        if choice == '1':
            run_task_a()
        elif choice == '2':
            run_task_b1()
        elif choice == '0':
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()