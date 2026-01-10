import numpy as np
import copy
import multiprocessing

class Particle:
    def __init__(self, dim, lower_bound, upper_bound):
        self.position = np.random.uniform(lower_bound, upper_bound, dim)
        self.velocity = np.zeros(dim)
        self.best_position = copy.deepcopy(self.position)
        self.best_score = float('inf')
        self.current_score = float('inf')

class PSO:
    def __init__(self, objective_func, bounds, num_particles=30, max_iter=100, w=0.7, c1=1.4, c2=1.4, dim=5, n_jobs=-1):
        """
        n_jobs : int
            Number of parallel processes. -1 means use all available cores.
            1 means sequential (standard mode).
        """
        self.func = objective_func
        self.bounds = bounds
        self.n_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        
        # Determine dimensionality
        if isinstance(bounds[0], (list, np.ndarray, tuple)) and len(bounds) > 2:
            self.dim = len(bounds)
            self.lower_bound = np.array([b[0] for b in bounds])
            self.upper_bound = np.array([b[1] for b in bounds])
        else:
            self.dim = dim
            self.lower_bound = bounds[0]
            self.upper_bound = bounds[1]
        
        # Swarm State
        self.particles = []
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.history = []

    def optimize(self):
        # Initialization sequence: Instantiate swarm and evaluate initial positions.
        
        # Manage pool creation and pass to optimization loop.
        
        if self.n_jobs > 1:
            print(f"Starting Parallel Optimization on {self.n_jobs} cores...")
            with multiprocessing.Pool(self.n_jobs) as pool:
                self._run_optimization_loop(pool)
        else:
            print("Starting Sequential Optimization...")
            self._run_optimization_loop(pool=None)

        return self.global_best_position, self.global_best_score

    def _run_optimization_loop(self, pool):
        """
        Internal loop to handle both sequential and parallel execution
        without duplicating logic.
        """
        # Initialization
        self.particles = [Particle(self.dim, self.lower_bound, self.upper_bound) for _ in range(self.n_particles)]
        self.global_best_score = float('inf')
        
        positions = [p.position for p in self.particles]
        
        # Initial assessment of swarm performance
        if pool:
            scores = pool.map(self.func, positions)
        else:
            scores = [self.func(p) for p in positions]
            
        for idx, p in enumerate(self.particles):
            score = scores[idx]
            p.best_score = score
            p.current_score = score
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = copy.deepcopy(p.position)
        
        # Main optimization loop
        range_width = self.upper_bound - self.lower_bound
        v_max = 0.2 * range_width
        w_max = 0.9
        w_min = 0.4
        
        for i in range(self.max_iter):
            current_w = w_max - i * ((w_max - w_min) / self.max_iter)
            
            positions_to_evaluate = []
            
            # Update velocity and position for each particle
            for particle in self.particles:
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)
                
                particle.velocity = (current_w * particle.velocity) + cognitive + social
                particle.velocity = np.clip(particle.velocity, -v_max, v_max)
                particle.position = particle.position + particle.velocity
                particle.position = np.clip(particle.position, self.lower_bound, self.upper_bound)
                
                positions_to_evaluate.append(particle.position)

            # Evaluate particles (parallel or sequential)
            if pool:
                scores = pool.map(self.func, positions_to_evaluate)
            else:
                scores = [self.func(pos) for pos in positions_to_evaluate]

            # Update personal and global best records
            for idx, particle in enumerate(self.particles):
                score = scores[idx]
                particle.current_score = score
                
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = copy.deepcopy(particle.position)
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = copy.deepcopy(particle.position)
            
            self.history.append(self.global_best_score)
            print(f"Iter {i+1}/{self.max_iter} | Best: {self.global_best_score:.6f}")