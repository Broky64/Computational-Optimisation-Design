import numpy as np
import os
from src.aerodynamics.evaluator import evaluate_airfoil

def generate_training_data(xfoil_path, n_samples=200):
    """
    Tâche C.3 : Génère un dataset CSV de profils aléatoires pour l'entrainement IA.
    """
    print(f"\n[ANALYSIS] Generating Surrogate Training Data ({n_samples} samples)...")
    
    csv_file = "results/surrogate_dataset.csv"
    os.makedirs('results', exist_ok=True)
    
    # Bornes (Intrados négatif, Extrados positif)
    lower_bounds = np.array([-0.6, -0.6, -0.6, 0.0, 0.0, 0.0])
    upper_bounds = np.array([ 0.0,  0.0,  0.0, 0.6, 0.6, 0.6])
    
    header = "w1,w2,w3,w4,w5,w6,CL,CD,CM"
    
    with open(csv_file, 'w') as f:
        f.write(header + "\n")
        
        valid_count = 0
        total_attempts = 0
        
        while valid_count < n_samples:
            total_attempts += 1
            
            # Sampling aléatoire
            weights = np.random.uniform(lower_bounds, upper_bounds)
            
            # Calcul XFOIL
            res = evaluate_airfoil(weights, 500_000, 3.0, xfoil_path)
            
            # Validation stricte
            if (res['CL'] is not None and 
                res['CD'] is not None and 
                res['CD'] > 1e-6 and 
                res['CD'] < 1.0):
                
                # Ecriture CSV
                w_str = ",".join([f"{w:.6f}" for w in weights])
                line = f"{w_str},{res['CL']:.6f},{res['CD']:.8f},{res['CM']:.6f}\n"
                f.write(line)
                f.flush()
                
                valid_count += 1
                print(f"  > Collected {valid_count}/{n_samples} (Total tries: {total_attempts})", end='\r')
                
    print(f"\n[DONE] Dataset generated: {csv_file}")
    print(f"Efficiency: {valid_count/total_attempts*100:.1f}%")