import numpy as np

def griewank_function(x):
    """
    Computes the Griewank function value for an input vector x.
    
    The global minimum is at x = [0, ..., 0] with f(x) = 0.
    Search range is usually [-600, 600].
    
    Parameters:
    -----------
    x : np.array or list
        Input vector of design variables (n dimensions).
        
    Returns:
    --------
    float
        The function value.
    """
    x = np.array(x)
    
    # Term 1: Summation component
    sum_term = np.sum(x**2) / 4000
    
    # Term 2: Product component
    # Warning: Indices in math start at 1, but in Python at 0.
    # We need to create an index array [1, 2, ..., n]
    indices = np.arange(1, len(x) + 1)
    prod_term = np.prod(np.cos(x / np.sqrt(indices)))
    
    # Final Formula
    result = sum_term - prod_term + 1
    
    return result

# Petit test rapide si on exécute ce fichier directement
if __name__ == "__main__":
    # Test au point optimal (devrait donner 0.0)
    x_opt = [0, 0, 0, 0, 0]
    print(f"Test at origin {x_opt}: f(x) = {griewank_function(x_opt)}")
    
    # Test avec un point aléatoire
    x_rand = [100, 200, 300, 400, 500]
    print(f"Test at random {x_rand}: f(x) = {griewank_function(x_rand)}")