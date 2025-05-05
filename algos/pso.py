from algos.base import BaseAlgo
import numpy as np


class PSOA(BaseAlgo):
    """
    Particle Swarm Optimisation (PSO) algorithm.
    """

    def _init_particles(self):
        particles = []
        for _ in range(self.num_particles):
            position = [
                np.random.randint(self.bounds[0][0], self.bounds[0][1] + 1),  # fast_sma
                np.random.randint(self.bounds[1][0], self.bounds[1][1] + 1),  # slow_sma
            ]
            velocity = [np.random.uniform(-1, 1) for _ in range(2)]
            particles.append(
                {
                    "position": position,
                    "velocity": velocity,
                    "best_position": position.copy(),
                    "best_fitness": -np.inf,
                }
            )
        return particles

    def __init__(self, eval_func, bounds, *, num_particles=30, max_iter=50):
        super().__init__(eval_func, bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.particles = self._init_particles()
        self.global_best_position = None
        self.global_best_fitness = -np.inf

    def optimize(self):
        for _ in range(self.max_iter):
            for particle in self.particles:
                # Evaluate fitness
                fitness_value = self.eval_func(particle["position"])

                # Update particle's best
                if fitness_value > particle["best_fitness"]:
                    particle["best_fitness"] = fitness_value
                    particle["best_position"] = particle["position"].copy()

                # Update global best
                if fitness_value > self.global_best_fitness:
                    self.global_best_fitness = fitness_value
                    self.global_best_position = particle["position"].copy()

            # Update velocities and positions
            for particle in self.particles:
                for i in range(len(particle["position"])):
                    r1, r2 = np.random.rand(), np.random.rand()
                    particle["velocity"][i] = (
                        self.w * particle["velocity"][i]
                        + self.c1
                        * r1
                        * (particle["best_position"][i] - particle["position"][i])
                        + self.c2
                        * r2
                        * (self.global_best_position[i] - particle["position"][i])
                    )

                    # Update position
                    particle["position"][i] += particle["velocity"][i]

                    # Bound the position
                    particle["position"][i] = max(
                        self.bounds[i][0],
                        min(self.bounds[i][1], particle["position"][i]),
                    )
                    if i in [0, 1]:  # fast_sma and slow_sma should be integers
                        particle["position"][i] = round(particle["position"][i])

        return self.global_best_position, self.global_best_fitness
