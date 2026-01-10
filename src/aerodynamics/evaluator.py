
import numpy as np
import sys
import os

# Ensure system path includes project root for module imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.aerodynamics.cst import CSTShapeGenerator
from src.aerodynamics.xfoil_runner import XFoilRunner

def evaluate_airfoil(weights, reynolds, alpha, xfoil_path):
    """
    Evaluates an airfoil defined by CST weights using XFOIL.
    
    Parameters
    ----------
    weights : list or np.ndarray
        List of 6 weights: first 3 for lower surface, last 3 for upper surface.
    reynolds : float
        Reynolds number.
    alpha : float
        Angle of attack in degrees.
    xfoil_path : str
        Path to the XFOIL executable.
        
    Returns
    -------
    dict
        Dictionary with keys 'CL', 'CD', 'CM'. Values are None if failed or invalid geometry.
    """
    
    # 1. Parse weights
    # Weights partitioned into lower (first 3) and upper (last 3) surfaces.
    w_lower = weights[:3]
    w_upper = weights[3:]
    
    cst = CSTShapeGenerator()
    
    # 2. Geometric Constraint Check (Thickness)
    # Verify that the upper surface is consistently above or equal to the lower surface.
    # Access internal surface calculation for thickness verification.
    n_points = 100
    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2
    
    y_upper = cst._calculate_surface(w_upper, x)
    y_lower = cst._calculate_surface(w_lower, x)
    
    # Invalid geometry if lower surface exceeds upper surface.
    if np.any(y_lower > y_upper):
        return {'CL': None, 'CD': None, 'CM': None}
        
    # 3. Generate Coordinates
    coords = cst.generate_airfoil(w_lower, w_upper, n_points=n_points)
    
    # 4. Run XFOIL
    runner = XFoilRunner(xfoil_path=xfoil_path, timeout=15.0)
    results = runner.analyze(coords, reynolds=reynolds, alpha=alpha)
    
    return results
