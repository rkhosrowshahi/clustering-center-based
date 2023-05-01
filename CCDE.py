import numpy as np
import base


class DEIterator:
    def __init__(self, de):
        self.de = de
        self.population = de.init()
        self.fitness = np.asarray(de.evaluate(self.population))
        # Sort population based on fitness values in ascending order
        sorted_indexes = np.argsort(self.fitness)
        self.population = self.population[sorted_indexes]
        self.fitness = self.fitness[sorted_indexes]
        self.best_fitness = self.fitness[0]
        self.best_idx = 0
        # Saving best and mean of fitness values
        self.all_best_fs = [self.best_fitness]
        self.all_mean_fs = [np.mean(self.fitness)]
        # F and CR control parameters
        self.f, self.cr = base.dither(de.mutation_bounds, de.crossover_bounds)
        if de.NC != 0:
            self.CS = (de.popsize - de.NC) // de.NC
        # Index of the last target vector used for mutation/crossover
        self.idx_target = 0
        # Current iteration of the algorithm (it is incremented
        # only when all vectors of the population are processed)
        self.fes = de.popsize
        self.iteration = 1

    def __iter__(self):
        de = self.de

        # This is the main DE loop. For each vector (target vector) in the
        # population, a mutant is created by combining different vectors in
        # the population (depending on the strategy selected). If the mutant
        # is better than the target vector, the target vector is replaced.
        yield self

        while self.fes < de.maxfes and self.iteration <= de.maxiters:
            # Compute values for f and cr in each iteration
            self.f, self.cr = self.calculate_params()
            offspring = []
            for self.idx_target in range(de.popsize - de.NC):
                # Create a mutant using a base vector, and the current f and cr values

                # for i in range(de.popsize - de.NC):
                mutant = self.create_mutant(self.idx_target)
                offspring.append(mutant)

            self.replace(offspring)
            self.fes += de.popsize - de.NC
            # Evaluate and replace if better
            # self.replace(self.idx_target, mutant)
            # Yield the current state of the algorithm
            # yield self
            if de.NC > 0:
                # Sort population based on fitness values in ascending order
                sorted_indexes = np.argsort(self.fitness)
                self.population = self.population[sorted_indexes]
                self.fitness = self.fitness[sorted_indexes]
                self.best_idx = 0
                self.best_fitness = self.fitness[self.best_idx]
                # Clustering the population in order to calculate the centroids
                # in order to be injected into the end of the population.
                centroids = []
                for cidx in range(de.NC):
                    cluster = np.copy(self.population[cidx * self.CS: (cidx + 1) * self.CS])
                    centroid = np.mean(cluster, axis=0)
                    centroids.append(centroid)
                # Inject centroids
                self.inject_centroids(centroids)
                # Increasing function evaluation value so far
                self.fes += de.NC
                # Sort population based on new centroids
                sorted_indexes = np.argsort(self.fitness)
                self.population = self.population[sorted_indexes]
                self.fitness = self.fitness[sorted_indexes]
                self.best_idx = 0
                self.best_fitness = self.fitness[self.best_idx]

            if self.fitness.min() <= self.best_fitness:
                self.best_fitness = self.fitness.min()
                self.best_idx = self.fitness.argmin()
            # Saving latest best and mean of fitness values
            self.all_mean_fs.append(np.mean(self.fitness))
            self.all_best_fs.append(self.best_fitness)
            self.iteration += 1
            yield self

    def calculate_params(self):
        return base.dither(self.de.mutation_bounds, self.de.crossover_bounds)

    def create_mutant(self, i):
        # Simple self-adaptive strategy, where the F and CR control
        # parameters are inherited from the base vector.
        return self.de.mutant(i, self.population, self.f, self.cr)

    def replace(self, offspring):
        # de = self.de
        offspring_fitness = self.de.evaluate(offspring)
        replace = self.fitness[:self.de.popsize - self.de.NC] >= offspring_fitness
        self.population[:self.de.popsize - self.de.NC] = np.where(replace[:, np.newaxis], np.copy(offspring),
                                                                  self.population[:self.de.popsize - self.de.NC])
        self.fitness[:self.de.popsize - self.de.NC] = np.where(replace, offspring_fitness,
                                                               self.fitness[:self.de.popsize - self.de.NC])

    def inject_centroids(self, centroids):
        # de = self.de
        centroids_fitness = self.de.evaluate(centroids)
        self.population[self.de.popsize - self.de.NC:, :] = np.copy(centroids)
        self.fitness[self.de.popsize - self.de.NC:] = centroids_fitness
        # mutant_fitness, = self.de.evaluate([mutant])
        # return self.replacement(i, mutant, mutant_fitness)

    def replacement(self, target_idx, mutant, mutant_fitness):
        if mutant_fitness <= self.best_fitness:
            self.best_fitness = mutant_fitness
            self.best_idx = target_idx
        if mutant_fitness <= self.fitness[target_idx]:
            self.population[target_idx] = mutant
            self.fitness[target_idx] = mutant_fitness
            return True
        return False


# class PDEIterator(DEIterator):
#     def __init__(self, de):
#         super().__init__(de)
#         self.mutants = np.zeros((de.popsize, de.dims))
#
#     def create_mutant(self, i):
#         mutant = super().create_mutant(i)
#         # Add to the mutants population for parallel evaluation (later)
#         # self.mutants.append(mutant)
#         self.mutants[i, :] = mutant
#         return mutant
#
#     def replace(self, i, mutant):
#         # Do not analyze after having the whole population (wait until the last individual)
#         if i == self.de.popsize - 1:
#             # Evaluate the whole new population (class PDE implements a parallel version of evaluate)
#             mutant_fitness = self.de.evaluate(self.mutants)
#             for j in range(self.de.popsize):
#                 super().replacement(j, self.mutants[j], mutant_fitness[j])


class CCDE:
    _binomial = {'best1bin': 'best1',
                 'randtobest1bin': 'randtobest1',
                 'currenttobest1bin': 'currenttobest1',
                 'best2bin': 'best2',
                 'rand2bin': 'rand2',
                 'rand1bin': 'rand1'}
    _exponential = {'best1exp': 'best1',
                    'rand1exp': 'rand1',
                    'randtobest1exp': 'randtobest1',
                    'currenttobest1exp': 'currenttobest1',
                    'best2exp': 'best2',
                    'rand2exp': 'rand2'}

    def __init__(self, fobj, bounds, strategy='rand1bin', mutation=0.5, crossover=0.9, maxfes=3e+06, popsize=100,
                 seed=None, NC=None, fstar=0, name="CCDE"):
        # Convert crossover param to an interval, as in mutation. If min/max values in the interval are
        # different, a dither mechanism is used for crossover (although this is not recommended, but still supported)

        if strategy in self._binomial:
            self.mutation_func = getattr(base, self._binomial[strategy])
            self.crossover_func = base.binomial_crossover
        elif strategy in self._exponential:
            self.mutation_func = getattr(base, self._exponential[strategy])
            self.crossover_func = base.binomial_crossover  # change to exponential
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy

        self.crossover_bounds = crossover
        self.mutation_bounds = mutation

        if getattr(crossover, '__len__', None) is None:
            self.crossover_bounds = [crossover, crossover]

        if getattr(mutation, '__len__', None) is None:
            self.mutation_bounds = [mutation, mutation]

        # If self-adaptive, include mutation and crossover as two new variables
        bnd = list(bounds)
        self._MIN, self._MAX = np.asarray(bnd, dtype='f8').T
        self._DIFF = np.fabs(self._MAX - self._MIN)
        self.dims = len(bnd)
        self.fobj = fobj
        self.popsize = popsize
        self.maxfes = maxfes
        self.maxiters = int(maxfes // self.popsize)
        self.NC = NC
        if NC == None:
            self.NC = 0
        # if the fstar is bigger than 0, the error has to be calculated by F(x) - F(x*)
        self.fstar = fstar
        self.initialize_random_state(seed)
        self.name = name

    @staticmethod
    def initialize_random_state(seed):
        np.random.seed(seed)

    @staticmethod
    def crossover(method, target, mutant, probability):
        return method(target, mutant, probability)

    @staticmethod
    def mutate(method, target_idx, population, f):
        return method(target_idx, population, f)

    def init(self):
        return base.random_init(self.popsize, self.dims)

    def mutant(self, target_idx, population, f, cr):
        # Create a mutant using a base vector
        trial = self.mutate(self.mutation_func,
                            target_idx, population, f)
        # Repair the individual if a gene is out of bounds
        mutant = self.crossover(self.crossover_func,
                                population[target_idx], trial, cr)
        return mutant

    def evaluate(self, P):
        # return self.fobj
        return [self.fobj(ind.reshape(1, -1))[0] - self.fstar for ind in P]

    def iterator(self):
        return iter(DEIterator(self))

    def geniterator(self):
        it = self.iterator()
        # iteration = 0
        for step in it:
            """if step.iteration != iteration:
                iteration = step.iteration"""
            yield step

    def solve(self, show_progress=False):
        if show_progress:
            from tqdm.auto import tqdm
            iterator = tqdm(self.geniterator(), total=self.maxiters,  # initial=1,
                            desc=f"Optimizing ({self.name})")
        else:
            iterator = self.geniterator()
        for step in iterator:
            idx = step.best_idx
            P = step.population
            bestfitness = step.fitness[idx]
            all_mean_fs = np.asarray(step.all_mean_fs)
            all_best_fs = np.asarray(step.all_best_fs)
        return P[idx].reshape(1, -1), bestfitness, all_best_fs, all_mean_fs

# class PCCDE(CCDE):
#     def __init__(self, fobj, bounds,  strategy='rand1bin', mutation=0.5, crossover=0.9, maxfes=3e+06, popsize=100, seed=None, NC=None, processes=None, chunksize=None):
#         super().__init__(fobj, bounds, strategy, mutation, crossover,
#                          maxfes, popsize, seed, NC)
#         from multiprocessing import Pool
#         self.processes = processes
#         self.chunksize = chunksize
#         self.name = 'Parallel CCDE'
#         self.pool = None
#         if processes is None or processes > 0:
#             self.pool = Pool(processes=self.processes)
#
#     def iterator(self):
#         it = PDEIterator(self)
#         try:
#             for data in it:
#                 yield data
#         finally:
#             if self.pool is not None:
#                 self.pool.terminate()
#
#     def evaluate(self, P):
#         if self.pool is not None:
#             return list(self.pool.map(self.fobj, P, chunksize=self.chunksize))
#         else:
#             return super().evaluate(P)
