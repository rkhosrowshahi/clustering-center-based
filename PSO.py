import numpy as np
from tqdm import tqdm


class PSO:
    def __init__(self, fobj, bounds, num_particles=100, maxfes=100, w=0.5, c1=1, c2=1, NC=0, fstar=0):
        self.fobj = fobj
        self.bounds = bounds
        self.num_particles = num_particles
        self.maxfes = maxfes
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.NC = NC
        self.fstar = fstar

    def solve(self):
        bnd = list(self.bounds)
        lb, hb = np.asarray(bnd).T
        dim = len(bnd)
        # Initialize particles and velocities
        particles = np.random.uniform(lb, hb, (self.num_particles, dim))
        velocities = np.zeros((self.num_particles - self.NC, dim))
        max_iter = int(self.maxfes // self.num_particles)
        fes = 0
        # Initialize the best positions and fitness values
        best_positions = np.copy(particles)
        best_fitness = np.array([self.fobj(ind.reshape(1, -1))[0] - self.fstar for ind in particles])
        fes += self.num_particles
        swarm_best_position = best_positions[np.argmin(best_fitness)]
        swarm_best_fitness = np.min(best_fitness)

        if self.NC != 0:
            CS = int((self.num_particles - self.NC) // self.NC)
        w = self.w
        c1 = self.c1
        c2 = self.c2
        all_best_fs, all_mean_fs = np.zeros(max_iter), np.zeros(max_iter)
        all_best_fs[0] = swarm_best_fitness
        all_mean_fs[0] = np.mean(best_fitness)
        # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
        for i in tqdm(range(max_iter)):
            if fes >= self.maxfes:
                break
            # Update velocities
            r1 = np.random.uniform(0, 1, (self.num_particles - self.NC, dim))
            r2 = np.random.uniform(0, 1, (self.num_particles - self.NC, dim))
            velocities = w * velocities + c1 * r1 * (
                    best_positions[:self.num_particles - self.NC] - particles[
                                                                    :self.num_particles - self.NC]) + c2 * r2 * (
                                 swarm_best_position - particles[
                                                       :self.num_particles - self.NC])

            # Update positions
            particles[:self.num_particles - self.NC] += velocities

            # Evaluate fitness of each particle
            fitness_values = np.array(
                [self.fobj(ind.reshape(1, -1))[0] - self.fstar for ind in particles[:self.num_particles - self.NC]])
            fes += self.num_particles - self.NC

            # Update best positions and fitness values
            improved_indices = np.where(fitness_values < best_fitness[:self.num_particles - self.NC])
            best_positions[improved_indices] = particles[improved_indices]
            best_fitness[improved_indices] = fitness_values[improved_indices]
            if np.min(fitness_values) <= swarm_best_fitness:
                swarm_best_position = particles[np.argmin(fitness_values)]
                swarm_best_fitness = np.min(fitness_values)

            if self.NC > 0:
                # Sort population based on fitness values in ascending order
                sorted_indexes = np.argsort(best_fitness)
                particles = particles[sorted_indexes]
                best_fitness = best_fitness[sorted_indexes]
                best_positions = best_positions[sorted_indexes]
                # velocities=velocities[sorted_indexes]
                # Clustering the population in order to calculate the centroids
                # in order to be injected into the end of the population.
                centroids = []
                for cidx in range(self.NC):
                    cluster = np.copy(best_positions[cidx * CS: (cidx + 1) * CS])
                    centroid = np.mean(cluster, axis=0)
                    centroids.append(centroid)
                # Inject centroids
                # inject_centroids(centroids)
                centroids_fitness = np.array([self.fobj(ind.reshape(1, -1))[0] - self.fstar for ind in centroids])
                # particles[self.num_particles - self.NC:, :] = np.copy(centroids)
                best_positions[self.num_particles - self.NC:, :] = np.copy(centroids)
                best_fitness[self.num_particles - self.NC:] = centroids_fitness
                # Increasing function evaluation value so far
                fes += self.NC
                # Sort population based on new centroids
                sorted_indexes = np.argsort(best_fitness)
                # particles = particles[sorted_indexes]
                best_fitness = best_fitness[sorted_indexes]
                best_positions = best_positions[sorted_indexes]
                # velocities=velocities[sorted_indexes]
                if np.min(centroids_fitness) < swarm_best_fitness:
                    swarm_best_position = centroids[np.argmin(centroids_fitness)]
                    swarm_best_fitness = np.min(centroids_fitness)

            all_best_fs[i+1] = swarm_best_fitness
            all_mean_fs[i+1] = np.mean(best_fitness)
        # Return the best solution found by the PSO algorithm
        return swarm_best_position, swarm_best_fitness, all_best_fs, all_mean_fs
# def pso():
