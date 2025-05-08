import numpy as np
import pandas as pd
from bot_training import bot_training

class PSO:
    def __init__(self, bot_type, bounds, num_particles=10, max_iter=100, group=None):
        self.bot_type = bot_type.lower()
        if self.bot_type not in ['sma', 'smaema', 'complex']:
            raise ValueError("bot_type must be 'sma', 'smaema', or 'complex'")
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.group = group
        # Integer dimensions: [0, 1] for sma/smaema, [3, 4, 5, 10, 11, 12] for complex
        # Other dimensions (weights and smoothing factors) can be continuous values
        self.integer_dims = [0, 1] if self.bot_type in ['sma', 'smaema'] else [3, 4, 5, 10, 11, 12]
        self.dim = len(bounds)  # Number of dimensions: 2 for sma, 3 for smaema, 14 for complex
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        self.history = []
        self.particles = self._initialize_particles()

    def _initialize_particles(self):
        """Initialize particles with random positions and velocities."""
        particles = []
        for i in range(self.num_particles):
            position = np.zeros(self.dim)
            for j, (low, high) in enumerate(self.bounds):
                if j in self.integer_dims:
                    position[j] = np.random.randint(low, high + 1)
                else:
                    position[j] = np.random.uniform(low, high)
            velocity = np.random.uniform(-1, 1, self.dim)
            particles.append({
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_fitness': -np.inf
            })
            # Evaluate and log initial particle
            params = self._convert_for_fitness(position)
            fitness = self._evaluate_fitness(params)
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = position.copy()
            
            # Add initial position to history
            row = [0, i] + list(position) + [fitness, self.global_best_fitness]
            self.history.append(row)
            
        return particles

    def _convert_for_fitness(self, position):
        """Ensuring integer dims are int"""
        return [int(round(val)) if i in self.integer_dims else float(val) for i, val in enumerate(position)]

    def _evaluate_fitness(self, params):
        """Evaluate fitness using bot_training function."""
        if self.bot_type == 'complex':
            high = params[:7]
            low = params[7:]
            return bot_training(self.bot_type, high, low)
        else:  # sma or smaema
            return bot_training(self.bot_type, *params)

    def optimize(self):
        """
        Run PSO optimization.
        
        Returns:
            (best_position, best_fitness)
        """
        for t in range(self.max_iter):
            for i, particle in enumerate(self.particles):
                # Compute fitness
                params = self._convert_for_fitness(particle['position'])
                fitness = self._evaluate_fitness(params)

                # Update particle's best
                if fitness > particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = particle['position'].copy()

                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle['position'].copy()

                # Log particle state
                row = [t + 1, i] + list(particle['position']) + [fitness, self.global_best_fitness]
                self.history.append(row)

            # Update velocities and positions
            for particle in self.particles:
                r1, r2 = np.random.rand(), np.random.rand()
                particle['velocity'] = (self.w * particle['velocity'] +
                                       self.c1 * r1 * (particle['best_position'] - particle['position']) +
                                       self.c2 * r2 * (self.global_best_position - particle['position']))

                # Update position
                particle['position'] += particle['velocity']

                # Enforce bounds
                for j in range(self.dim):
                    particle['position'][j] = np.clip(particle['position'][j], self.bounds[j][0], self.bounds[j][1])
                    if j in self.integer_dims:
                        particle['position'][j] = round(particle['position'][j])

        # Save log to CSV
        columns = ["Iteration", "Particle_ID"] + [f"Dim_{j}" for j in range(self.dim)] + ["Fitness", "Best_So_far"]
        df = pd.DataFrame(self.history, columns=columns)
        filename = f"pso_{self.bot_type}_{'Group'+self.group if self.group else ''}_log.csv"
        df.to_csv(filename, index=False)

        # Return best position
        final_position = self._convert_for_fitness(self.global_best_position)
        return final_position, self.global_best_fitness


def particle_swarm_optimization(fitness_func, bot_type, dim, bounds, num_agents=10, max_iter=100, integer_dims=None, group=None):
    print("Start PSO")
    # Initialize and run PSO
    pso = PSO(bot_type, bounds, num_agents, max_iter, group)
    
    # Run optimization
    best_position, best_fitness = pso.optimize()
    
    # Format the output
    if bot_type == "complex":
        best_position = [best_position[:7], best_position[7:]]
        
    return best_position, best_fitness