from algos.base import BaseAlgo
import numpy as np


class PSO(BaseAlgo):
    """
    Particle Swarm Optimisation (PSO) algorithm.
    """

    def __init_particles(self, num_particles):
        """
        Initialize the particles in the swarm.
        :param num_particles: Number of particles in the swarm.
        """
        particles = []
        for _ in range(num_particles):
            position = [np.random.randint(low, high + 1) for low, high in self.bounds]
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

    def _algo_init(self, _, *, num_particles=30):
        """
        Initialize the PSO algorithm with the given parameters.
        :param num_particles: Number of particles in the swarm.
        """
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter

        self.particles = self.__init_particles(num_particles)

    def _algo_iter(self, _):
        for particle in self.particles:
            # Evaluate fitness and update global best
            fitness_value = self._eval_and_update(particle["position"])

            # Update particle's best
            if fitness_value > particle["best_fitness"]:
                particle["best_fitness"] = fitness_value
                particle["best_position"] = particle["position"].copy()

        # Update velocities and positions
        for particle in self.particles:
            for i in range(len(particle["position"])):
                r1, r2 = np.random.rand(), np.random.rand()
                particle["velocity"][i] = (
                    self.w * particle["velocity"][i]
                    + self.c1
                    * r1
                    * (particle["best_position"][i] - particle["position"][i])
                    + self.c2 * r2 * (self.best_params[i] - particle["position"][i])
                )

                # Update position
                particle["position"][i] += particle["velocity"][i]

                # Bound the position
                particle["position"][i] = max(
                    self.bounds[i][0],
                    min(self.bounds[i][1], particle["position"][i]),
                )
                if i in self.integer_dims:  # fast_sma and slow_sma should be integers
                    particle["position"][i] = round(particle["position"][i])
