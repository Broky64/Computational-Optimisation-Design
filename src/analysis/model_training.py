import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib # Pour sauvegarder le modèle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def train_surrogate_model():
    """
    Tâche C.3 (Partie 2) : Entraînement du Modèle de Substitution (Surrogate Model).
    Utilise Random Forest pour prédire L/D à partir des poids CST.
    """
    csv_path = "results/surrogate_dataset.csv"
    model_path = "results/surrogate_model.pkl"
    report_path = "results/task_c3_model_report.txt"
    
    print(f"\n[ANALYSIS] Training Surrogate Model...")
    
    # 1. Chargement des données
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset not found at {csv_path}. Run Option 6 first!")
        return

    df = pd.read_csv(csv_path)
    
    # On calcule la cible (L/D)
    # Attention aux divisions par zéro éventuelles (bien que déjà filtrées)
    df['LD'] = df['CL'] / df['CD']
    
    # Features (Entrées) : w1 à w6
    X = df[['w1', 'w2', 'w3', 'w4', 'w5', 'w6']]
    # Target (Sortie) : LD
    y = df['LD']
    
    # 2. Séparation Train / Test (80% pour apprendre, 20% pour tester)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  Dataset Size:   {len(df)}")
    print(f"  Training Set:   {len(X_train)}")
    print(f"  Test Set:       {len(X_test)}")
    
    # 3. Entraînement (Random Forest)
    # n_estimators=100 : 100 arbres de décision
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Évaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test) # Le score le plus important (sur des données inconnues)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"  Model Trained. R2 Score (Test): {r2_test:.4f}")
    
    # 5. Sauvegarde du modèle (pour utilisation future)
    joblib.dump(model, model_path)
    
    # 6. Rapport Texte
    report = f"""===========================================================
TASK C.3 REPORT: Surrogate Model Training
===========================================================
ALGORITHM: Random Forest Regressor
DATASET:   {csv_path} ({len(df)} samples)

PERFORMANCE METRICS:
  R2 Score (Train):   {r2_train:.4f}  (Should be close to 1.0)
  R2 Score (Test):    {r2_test:.4f}   (> 0.8 is good, > 0.9 is excellent)
  Mean Abs Error:     {mae_test:.4f}  (Average prediction error on L/D)

CONCLUSION:
  The model is saved as '{model_path}'.
  It can now predict L/D instantly without XFOIL.
===========================================================
"""
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"[REPORT SAVED] -> {report_path}")

    # 7. Graphique de Validation (Predicted vs Actual)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Test Data')
    
    # Ligne diagonale parfaite
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title(f'Surrogate Model Accuracy (R2 = {r2_test:.3f})')
    plt.xlabel('Actual L/D (XFOIL)')
    plt.ylabel('Predicted L/D (AI Model)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/surrogate_validation.png')
    plt.close()
    print(f"[PLOT SAVED]   -> results/surrogate_validation.png")