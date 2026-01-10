import numpy as np
import copy

class Particle:
    def __init__(self, dim, lower_bound, upper_bound):
        self.position = np.random.uniform(lower_bound, upper_bound, dim)
        self.velocity = np.zeros(dim)
        self.best_position = copy.deepcopy(self.position)
        self.best_score = float('inf')
        self.current_score = float('inf')

class PSO:
    def __init__(self, objective_func, bounds, num_particles=30, max_iter=100, w=0.7, c1=1.4, c2=1.4, dim=5):
        """
        Particle Swarm Optimization Solver.
        
        Parameters:
        -----------
        objective_func : function
            The function to minimize.
        bounds : tuple
            (lower_bound, upper_bound) e.g., (-600, 600)
        num_particles : int
            Number of particles in the swarm.
        max_iter : int
            Maximum number of iterations.
        w : float
            Initial inertia weight (will be overridden by adaptive logic).
        c1 : float
            Cognitive coefficient (personal learning).
        c2 : float
            Social coefficient (global learning).
        dim : int
            Dimensionality of the problem (if bounds is a simple tuple).
        """
        self.func = objective_func
        self.bounds = bounds
        self.n_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Determine dimensionality
        if isinstance(bounds[0], (list, np.ndarray, tuple)) and len(bounds) > 2:
             # e.g. bounds = [(-600, 600), (-600, 600), ...]
            self.dim = len(bounds)
            self.lower_bound = np.array([b[0] for b in bounds])
            self.upper_bound = np.array([b[1] for b in bounds])
        else:
            # e.g. bounds = (-600, 600)
            self.dim = dim
            self.lower_bound = bounds[0]
            self.upper_bound = bounds[1]
        
        # Swarm State
        self.particles = []
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.history = [] # I'll use this to store the convergence data

    def optimize(self):
        # 1. First, I'll initialize the swarm
        self._initialize_swarm()
        
        # --- MODIFICATION: Velocity Clamping ---
        # Limit velocity to 20% of the total search range to prevent explosion
        range_width = self.upper_bound - self.lower_bound
        v_max = 0.2 * range_width
        
        # --- MODIFICATION: Adaptive Inertia settings ---
        w_max = 0.9  # High exploration at start
        w_min = 0.4  # High exploitation at end
        
        # 2. This is the main optimization loop
        for i in range(self.max_iter):
            
            # Update Inertia Weight linearly
            current_w = w_max - i * ((w_max - w_min) / self.max_iter)
            
            for particle in self.particles:
                
                # --- UPDATE VELOCITY ---
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # v_new = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)
                
                # Use adaptive 'current_w' instead of fixed 'self.w'
                particle.velocity = (current_w * particle.velocity) + cognitive + social
                
                # --- APPLY VELOCITY CLAMPING ---
                # Clip velocity between -v_max and v_max
                particle.velocity = np.clip(particle.velocity, -v_max, v_max)
                
                # --- UPDATE POSITION ---
                particle.position = particle.position + particle.velocity
                
                # --- HANDLE BOUNDS ---
                particle.position = np.clip(particle.position, self.lower_bound, self.upper_bound)
                
                # --- EVALUATE ---
                score = self.func(particle.position)
                particle.current_score = score
                
                # I'll update the personal best if the new score is lower
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = copy.deepcopy(particle.position)
                    
                # And here I update the global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = copy.deepcopy(particle.position)
            
            # Logging the history for each iteration
            self.history.append(self.global_best_score)
            # print(f"Iter {i+1}/{self.max_iter} | Best Score: {self.global_best_score:.6f}")

        return self.global_best_position, self.global_best_score

    def _initialize_swarm(self):
        self.particles = []
        self.global_best_score = float('inf') # Reset global best for new run
        
        for _ in range(self.n_particles):
            p = Particle(self.dim, self.lower_bound, self.upper_bound)
            
            score = self.func(p.position)
            p.best_score = score
            p.current_score = score
            
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = copy.deepcopy(p.position)
            
            self.particles.append(p)

if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to python path to allow imports of other modules
    # Current file is in src/algorithms/pso.py, so we go up 3 levels to get project root?
    # No, usually src is at root or 'Computational Optimisation Design' is root.
    # User's main.py is in 'Computational Optimisation Design'.
    # src is in 'Computational Optimisation Design/src'.
    # So we need to add 'Computational Optimisation Design' to path.
    # relative path from pso.py: ../..
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.append(project_root)
    
    try:
        from src.benchmarks.griewank import griewank_function
        func_name = "Griewank"
        func = griewank_function
        bounds = (-600, 600)
    except ImportError:
        print("Could not import Griewank function, using Sphere function instead.")
        def sphere_function(x):
            return np.sum(np.array(x)**2)
        func_name = "Sphere"
        func = sphere_function
        bounds = (-10, 10)

    print(f"Running PSO directly on {func_name} function...")
    
    pso = PSO(
        objective_func=func,
        bounds=bounds,
        num_particles=50,
        max_iter=100,
        dim=5
    )
    
    best_pos, best_score = pso.optimize()
    
    print("\nOptimization Results:")
    print(f"  Best Score: {best_score}")
    print(f"  Best Position: {best_pos}")