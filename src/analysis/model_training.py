import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def train_surrogate_model():
    csv_path = "results/surrogate_dataset.csv"
    model_path = "results/surrogate_model.pkl"
    report_path = "results/task_c3_model_report.txt"
    
    print(f"\n[ANALYSIS] Training Surrogate Model...")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset not found. Run Option 6 first!")
        return

    df = pd.read_csv(csv_path)
    
    # Calculate Lift-to-Drag ratio
    df['LD'] = df['CL'] / df['CD']
    
    # Data Cleaning: Outlier Removal
    # Remove non-physical or extreme results to improve model accuracy.
    # Retain Lift-to-Drag ratios within the realistic range of 0 to 200.
    original_len = len(df)
    df = df[ (df['LD'] > 0) & (df['LD'] < 200) ]
    cleaned_len = len(df)
    print(f"  Data Cleaning: Removed {original_len - cleaned_len} outliers.")
    
    if cleaned_len < 50:
        print("[ERROR] Insufficient valid data remaining after cleaning.")
        return

    # Data preparation
    X = df[['w1', 'w2', 'w3', 'w4', 'w5', 'w6']]
    y = df['LD']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  Training Set:   {len(X_train)}")
    print(f"  Test Set:       {len(X_test)}")
    
    # Model training with 200 trees
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Performance evaluation
    y_pred_test = model.predict(X_test)
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"  Model Trained. R2 Score (Test): {r2_test:.4f}")
    
    joblib.dump(model, model_path)
    
    # Report generation
    report = f"""===========================================================
TASK C.3 REPORT: Surrogate Model Training
===========================================================
ALGORITHM: Random Forest Regressor (200 trees)
DATASET:   {csv_path}
CLEANING:  Removed {original_len - cleaned_len} outliers (kept 0 < L/D < 200)

PERFORMANCE METRICS:
  R2 Score (Train):   {r2_train:.4f}
  R2 Score (Test):    {r2_test:.4f}   (Target: > 0.85)
  Mean Abs Error:     {mae_test:.4f}

CONCLUSION:
  Model saved to '{model_path}'.
===========================================================
"""
    with open(report_path, 'w') as f: f.write(report)
    print(f"[REPORT SAVED] -> {report_path}")

    # Result visualization
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Test Data')
    
    # Reference diagonal representing perfect prediction
    limit_min = min(y_test.min(), y_pred_test.min())
    limit_max = max(y_test.max(), y_pred_test.max())
    plt.plot([limit_min, limit_max], [limit_min, limit_max], 'r--', label='Perfect Prediction')
    
    plt.title(f'Surrogate Model Accuracy (R2 = {r2_test:.3f})')
    plt.xlabel('Actual L/D'); plt.ylabel('Predicted L/D')
    plt.legend(); plt.grid(True)
    plt.savefig('results/surrogate_validation.png'); plt.close()
    print(f"[PLOT SAVED]   -> results/surrogate_validation.png")