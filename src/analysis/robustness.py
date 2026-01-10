import numpy as np
import matplotlib.pyplot as plt
import os
from src.aerodynamics.evaluator import evaluate_airfoil

def run_robustness_analysis(best_weights, xfoil_path, n_samples=100):
    """
    Exécute l'analyse de robustesse (Monte Carlo) pour la Tâche C.1.
    """
    print(f"\n[ANALYSIS] Starting Robustness Check on {n_samples} samples...")
    print(f"Base Weights: {best_weights}")

    alpha_mean = 3.0
    alpha_std = 0.1
    
    # 1. Génération des scénarios
    alphas = np.random.normal(alpha_mean, alpha_std, n_samples)
    
    cl_res = []
    cd_res = []
    valid_count = 0
    
    # 2. Simulation
    for i, alpha_val in enumerate(alphas):
        res = evaluate_airfoil(best_weights, 500_000, alpha_val, xfoil_path)
        
        # Filtre de validité (éviter les crashs ou CD=0)
        if res['CL'] is not None and res['CD'] is not None and res['CD'] > 1e-6:
            cl_res.append(res['CL'])
            cd_res.append(res['CD'])
            valid_count += 1
            
        if i % 10 == 0: 
            print(f"  > Simulating flight condition {i}/{n_samples}...", end='\r')
            
    print(f"\n[DONE] Valid simulations: {valid_count}/{n_samples}")
    
    if valid_count == 0:
        print("[ERROR] Aucune simulation valide.")
        return

    # 3. Calculs Statistiques
    cl_arr = np.array(cl_res)
    cd_arr = np.array(cd_res)
    ld_vals = cl_arr / cd_arr
    
    # Nettoyage des infinis éventuels
    mask = np.isfinite(ld_vals)
    ld_vals = ld_vals[mask]
    cl_arr = cl_arr[mask]
    cd_arr = cd_arr[mask]
    
    ld_mean, ld_std = np.mean(ld_vals), np.std(ld_vals)
    
    # 4. Génération du Rapport
    report = f"""===========================================================
TASK C.1 REPORT: Robustness Analysis (Monte Carlo)
===========================================================
Input Weights: {best_weights}
Parameters: Alpha ~ N({alpha_mean}, {alpha_std}) | Samples: {n_samples}
Valid Samples: {len(ld_vals)}/{n_samples}

ROBUSTNESS STATISTICS:
  L/D RATIO:
    Mean:    {ld_mean:.4f}
    Std Dev: {ld_std:.4f}  <-- Key Metric
  LIFT (CL):
    Mean:    {np.mean(cl_arr):.4f} | Std: {np.std(cl_arr):.4f}
  DRAG (CD):
    Mean:    {np.mean(cd_arr):.6f} | Std: {np.std(cd_arr):.6f}
===========================================================
"""
    # Sauvegarde Rapport Texte
    os.makedirs('results', exist_ok=True)
    with open("results/task_c1_results.txt", 'w') as f:
        f.write(report)
    print("\n" + report)
    print("[SAVED] Report -> results/task_c1_results.txt")

    # 5. Génération du Graphique
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.hist(cl_arr, bins=15, color='skyblue', edgecolor='black'); plt.title('Lift (CL)')
    plt.subplot(1, 3, 2); plt.hist(cd_arr, bins=15, color='salmon', edgecolor='black'); plt.title('Drag (CD)')
    plt.subplot(1, 3, 3); plt.hist(ld_vals, bins=15, color='lightgreen', edgecolor='black'); plt.title('L/D Ratio')
    plt.tight_layout()
    plt.savefig('results/task_c1_robustness.png')
    plt.close()
    print("[SAVED] Plot -> results/task_c1_robustness.png")