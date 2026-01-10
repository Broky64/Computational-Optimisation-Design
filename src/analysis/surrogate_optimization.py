import numpy as np
import joblib
import pandas as pd
import os
import time
from src.algorithms.pso import PSO
from src.aerodynamics.evaluator import evaluate_airfoil

def run_surrogate_optimization(xfoil_path):
    """
    Tâche C.4 : Optimisation utilisant le Surrogate Model (IA) au lieu de XFOIL.
    Objectif : Montrer l'accélération drastique du temps de calcul.
    """
    model_path = "results/surrogate_model.pkl"
    print(f"\n[ANALYSIS] Starting Optimization using Surrogate Model...")
    
    # 1. Chargement du modèle
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Run Option 7 first!")
        return
        
    model = joblib.load(model_path)
    print("  Model loaded successfully.")
    
    # 2. Définition de la fonction objectif "Virtuelle" (Ultra rapide)
    def surrogate_objective(weights):
        # Le modèle attend un DataFrame avec des noms de colonnes précis
        # weights est une liste [w1, w2, w3, w4, w5, w6]
        X = np.array(weights).reshape(1, -1)
        X_df = pd.DataFrame(X, columns=['w1','w2','w3','w4','w5','w6'])
        
        # Prédiction instantanée
        predicted_ld = model.predict(X_df)[0]
        
        # On maximise L/D => on minimise l'opposé
        return -predicted_ld

    # 3. Configuration du PSO
    # On peut se permettre beaucoup d'itérations car c'est gratuit en temps de calcul
    dim = 6
    bounds = [(-0.6, 0.0)]*3 + [(0.0, 0.6)]*3
    
    pso_params = {
        'num_particles': 50, 
        'max_iter': 100,      
        'w': 0.7, 'c1': 1.4, 'c2': 1.4,
        'dim': dim,
        'n_jobs': 1 # Pas de parallélisme nécessaire, le modèle est trop rapide
    }
    
    print("  Launching PSO on Surrogate Model (50 particles, 100 iter)...")
    
    # 4. Exécution et Chronométrage
    start_time = time.time()
    
    solver = PSO(surrogate_objective, bounds, **pso_params)
    best_weights, best_score_pred = solver.optimize()
    
    elapsed = time.time() - start_time
    print(f"  Optimization finished in {elapsed:.4f} seconds!")
    
    # 5. Vérification avec la "Vérité Terrain" (XFOIL)
    print("\n[VERIFICATION] Validating result with real XFOIL...")
    real_res = evaluate_airfoil(best_weights, 500_000, 3.0, xfoil_path)
    
    real_ld = 0.0
    if real_res['CL'] and real_res['CD'] and real_res['CD'] > 0:
        real_ld = real_res['CL'] / real_res['CD']
    
    predicted_ld = -best_score_pred
    error = abs(predicted_ld - real_ld)
    
    # 6. Rapport
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