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
            Inertia weight.
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
        
        # 2. This is the main optimization loop
        for i in range(self.max_iter):
            for particle in self.particles:
                
                # --- UPDATE VELOCITY ---
                # I'm using r1 and r2 as random vectors between 0 and 1
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # v_new = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)
                
                particle.velocity = (self.w * particle.velocity) + cognitive + social
                
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
            print(f"Iter {i+1}/{self.max_iter} | Best Score: {self.global_best_score:.6f}")

        return self.global_best_position, self.global_best_score

    def _initialize_swarm(self):
        self.particles = []
        for _ in range(self.n_particles):
            p = Particle(self.dim, self.lower_bound, self.upper_bound)
            
            score = self.func(p.position)
            p.best_score = score
            p.current_score = score
            
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = copy.deepcopy(p.position)
            
            self.particles.append(p)