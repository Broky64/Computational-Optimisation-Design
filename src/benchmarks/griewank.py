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
    # Mathematical indices begin at 1; adjust for Python's 0-based indexing.
    # Generate an index array starting from 1 for the product term.
    indices = np.arange(1, len(x) + 1)
    prod_term = np.prod(np.cos(x / np.sqrt(indices)))
    
    # Griewank function calculation.
    result = sum_term - prod_term + 1
    
    return result

# Script execution test block.
if __name__ == "__main__":
    # Validation at global minimum (expected value: 0.0).
    x_opt = [0, 0, 0, 0, 0]
    print(f"Test at origin {x_opt}: f(x) = {griewank_function(x_opt)}")
    
    # Validation with an arbitrary test point.
    x_rand = [100, 200, 300, 400, 500]
    print(f"Test at random {x_rand}: f(x) = {griewank_function(x_rand)}")