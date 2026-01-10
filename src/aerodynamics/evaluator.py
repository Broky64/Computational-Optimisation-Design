
import numpy as np
import sys
import os

# Ensure we can import modules if running directly, though usually this is handled by package structure
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
    # Assuming standard order: first 3 are lower, last 3 are upper
    w_lower = weights[:3]
    w_upper = weights[3:]
    
    cst = CSTShapeGenerator()
    
    # 2. Geometric Constraint Check (Thickness)
    # I'll check that the upper surface is strictly above (or equal to) the lower surface.
    # accessing the internal method for a precise grid check.
    n_points = 100
    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2
    
    y_upper = cst._calculate_surface(w_upper, x)
    y_lower = cst._calculate_surface(w_lower, x)
    
    # If the lower surface is above the upper surface at any point, it's invalid.
    if np.any(y_lower > y_upper):
        return {'CL': None, 'CD': None, 'CM': None}
        
    # 3. Generate Coordinates
    coords = cst.generate_airfoil(w_lower, w_upper, n_points=n_points)
    
    # 4. Run XFOIL
    runner = XFoilRunner(xfoil_path=xfoil_path, timeout=15.0)
    results = runner.analyze(coords, reynolds=reynolds, alpha=alpha)
    
    return results
