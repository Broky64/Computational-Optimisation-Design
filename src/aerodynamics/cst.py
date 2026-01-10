"""
CST (Class Shape Transformation) Airfoil Geometry Generator.
"""

import numpy as np
import math

class CSTShapeGenerator:
    """
    Generates airfoil coordinates using the Class Shape Transformation (CST) method.
    
    The method uses Bernstein polynomials to define the shape function S(x), and a 
    class function C(x) to define the general class of the geometry (e.g., airfoil with round nose 
    and sharp trailing edge).
    """

    def __init__(self):
        pass

    def _bernstein_poly(self, r: int, n: int, x: np.ndarray) -> np.ndarray:
        """
        Calculate the r-th Bernstein polynomial of degree n at x.
        
        B_{r,n}(x) = K_{r,n} * x^r * (1-x)^(n-r)
        where K_{r,n} is the binomial coefficient.
        """
        k = math.factorial(n) / (math.factorial(r) * math.factorial(n - r))
        return k * (x**r) * ((1 - x)**(n - r))

    def _calculate_surface(self, weights: list, x: np.ndarray) -> np.ndarray:
        """
        Calculate y-coordinates for a surface given weights.
        
        y(x) = C(x) * S(x)
        C(x) = x^0.5 * (1-x)^1.0  (Class function for airfoils)
        S(x) = Sum(w_i * B_{i,n}(x))
        """
        n = len(weights) - 1
        class_function = (x**0.5) * (1 - x)
        
        shape_function = np.zeros_like(x)
        for i, w in enumerate(weights):
            shape_function += w * self._bernstein_poly(i, n, x)
            
        return class_function * shape_function

    def generate_airfoil(self, w_lower: list, w_upper: list, n_points: int = 100) -> np.ndarray:
        """
        Generate airfoil coordinates from CST weights.

        Parameters
        ----------
        w_lower : list
            Weights for the lower surface shape function.
        w_upper : list
            Weights for the upper surface shape function.
        n_points : int, optional
            Number of points per surface. Total points will be roughly 2*n_points.
            Default is 100.

        Returns
        -------
        np.ndarray
            (N, 2) array of X, Y coordinates, ordered Counter-Clockwise 
            from Trailing Edge (Upper) -> Leading Edge -> Trailing Edge (Lower).
            Format fits XFOIL requirements.
        """
        # Distribute points using cosine spacing for better resolution at LE/TE
        beta = np.linspace(0, np.pi, n_points)
        x = (1 - np.cos(beta)) / 2  # Points from 0.0 to 1.0

        # Calculate y coordinates
        y_upper = self._calculate_surface(w_upper, x)
        y_lower = self._calculate_surface(w_lower, x)
        
        # Upper surface: Trailing Edge (1.0) to Leading Edge (0.0)
        # x array goes 0->1. We want 1->0 for upper in the final list?
        # Standard XFOIL: Top surface from TE to LE, then Bottom surface from LE to TE.
        
        # Prepare Upper Segment: (1.0 -> 0.0)
        # Currently x is 0 -> 1. So y_upper corresponds to x 0->1.
        # We need to reverse both for the "Top" segment of the file.
        x_top = np.flip(x)
        y_top = np.flip(y_upper)
        
        # Prepare Lower Segment: (0.0 -> 1.0)
        # We skip the first point (0,0) to avoid duplication with x_top's last point
        x_bot = x[1:]
        y_bot = y_lower[1:]
        
        # For the lower surface, y is typically negative if I provide negative weights.
        # I'm assuming the weights I pass already account for the sign.
        # Finally, I'll combine the top and bottom coordinates into one array for XFOIL.
        
        # Combine
        coords_x = np.concatenate([x_top, x_bot])
        coords_y = np.concatenate([y_top, y_bot])
        
        coordinates = np.column_stack((coords_x, coords_y))
        
        return coordinates
