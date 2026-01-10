import numpy as np
import matplotlib.pyplot as plt
import os

def plot_airfoil_from_dat(dat_file, output_file):
    """
    Lit un fichier .dat (format XFOIL) et génère une image PNG propre.
    """
    print(f"\n[VISUALIZATION] Processing {dat_file}...")
    
    # 1. Vérification du fichier
    if not os.path.exists(dat_file):
        print(f"[ERROR] Le fichier '{dat_file}' n'existe pas.")
        print("Avez-vous lancé la Tâche B.3 ?")
        return

    try:
        # 2. Chargement des données
        # skiprows=1 pour ignorer le titre "Optimized (L/D=...)"
        data = np.loadtxt(dat_file, skiprows=1)
        x = data[:, 0]
        y = data[:, 1]
        
        # 3. Création du Graphique
        plt.figure(figsize=(12, 4)) # Format panoramique pour l'aile
        
        # Tracé principal
        plt.plot(x, y, color='#0055A4', linewidth=2, label='Optimized Shape')
        
        # Remplissage de l'aile
        plt.fill(x, y, color='#0055A4', alpha=0.2)
        
        # Points de contrôle (discrets)
        plt.scatter(x, y, color='black', s=10, marker='.', alpha=0.5, label='Coordinates')
        
        # Ligne de corde (Corde = ligne entre 0,0 et 1,0)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # 4. Mise en forme technique
        plt.title('Optimized Airfoil Geometry', fontsize=14)
        plt.xlabel('Position x/c (Chord)', fontsize=12)
        plt.ylabel('Thickness y/c', fontsize=12)
        
        # CRUCIAL : Force les axes à avoir la même échelle (sinon l'aile paraît grosse)
        plt.axis('equal') 
        
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='upper right')
        
        # 5. Sauvegarde
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"[SUCCESS] Image saved to: {output_file}")
        
    except Exception as e:
        print(f"[ERROR] Impossible de tracer le profil : {e}")