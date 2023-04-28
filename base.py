import numpy as np


def binomial_crossover(target, mutant, cr):
    n = len(target)
    p = np.random.rand(n) < cr
    if not np.any(p):
        p[np.random.randint(0, n)] = True
    return np.where(p, mutant, target)


def random_sample(popsize, exclude, size=5):
    # Optimized version using numpy
    idxs = list(range(popsize))
    idxs.remove(exclude)
    return np.random.choice(idxs, size=size, replace=False)


def rand1(target_idx, population, f):
    samples = random_sample(population.shape[0], target_idx, size=3)
    a, b, c = population[samples]

    pop_0 = a
    pop_r0 = b
    pop_r1 = c

    return pop_0 + f * (pop_r0 - pop_r1)


def best1(target_idx, population, f):
    samples = random_sample(population.shape[0], target_idx, size=2)
    a, b = population[samples]

    pop_0 = population[0]
    pop_r0 = a
    pop_r1 = b

    return pop_0 + f * (pop_r0 - pop_r1)


def random_init(popsize, dimensions):
    return np.random.rand(popsize, dimensions)


def dither_from_interval(interval):
    low, up = min(interval), max(interval)
    if low == up:
        return low
    return np.random.uniform(low, up)


def dither(*intervals):
    return [dither_from_interval(interval) for interval in intervals]


class _FunctionWrapper:
    """
    Object to wrap user cost function, allowing picklability
    """

    def __init__(self, f, args):
        self.f = f
        self.args = [] if args is None else args

    def __call__(self, x):
        return self.f(x, *self.args)


class MapWrapper:
    """
    Parallelisation wrapper for working with map-like callables, such as
    `multiprocessing.Pool.map`.
    Parameters
    ----------
    pool : int or map-like callable
        If `pool` is an integer, then it specifies the number of threads to
        use for parallelization. If ``int(pool) == 1``, then no parallel
        processing is used and the map builtin is used.
        If ``pool == -1``, then the pool will utilize all available CPUs.
        If `pool` is a map-like callable that follows the same
        calling sequence as the built-in map function, then this callable is
        used for parallelization.
    """

    def __init__(self, pool=1):
        self.pool = None
        self._mapfunc = map
        self._own_pool = False

        if callable(pool):
            self.pool = pool
            self._mapfunc = self.pool
        else:
            from multiprocessing import Pool
            # user supplies a number
            if pool == None:
                # use as many processors as possible
                self.pool = Pool()
                self._mapfunc = self.pool.map
                self._own_pool = True
            elif int(pool) == 1:
                pass
            elif int(pool) > 1:
                # use the number of processors requested
                self.pool = Pool(processes=int(pool))
                self._mapfunc = self.pool.map
                self._own_pool = True
            else:
                raise RuntimeError("Number of workers specified must be -1,"
                                   " an int >= 1, or an object with a 'map' "
                                   "method")

    def __enter__(self):
        return self

    def terminate(self):
        if self._own_pool:
            self.pool.terminate()

    def join(self):
        if self._own_pool:
            self.pool.join()

    def close(self):
        if self._own_pool:
            self.pool.close()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._own_pool:
            self.pool.close()
            self.pool.terminate()

    def __call__(self, func, iterable):
        # only accept one iterable because that's all Pool.map accepts
        try:
            return self._mapfunc(func, iterable)
        except TypeError as e:
            # wrong number of arguments
            raise TypeError("The map-like callable must be of the"
                            " form f(func, iterable)") from e