_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}

from contextlib import contextmanager
import functools
import operator
import sys
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
    Optional,
    Union,
    TYPE_CHECKING,
    TypeVar,
)

import numpy as np

IntNumber = Union[int, np.integer]
DecimalNumber = Union[float, np.floating, np.integer]

# Since Generator was introduced in numpy 1.17, the following condition is needed for
# backward compatibility
if TYPE_CHECKING:
    SeedType = Optional[Union[IntNumber, np.random.Generator,
                              np.random.RandomState]]
    GeneratorType = TypeVar("GeneratorType", bound=Union[np.random.Generator,
                                                         np.random.RandomState])

try:
    from numpy.random import Generator as Generator
except ImportError:
    class Generator():  # type: ignore[no-redef]
        pass


def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    """
    np.where(cond, x, fillvalue) always evaluates x even where cond is False.
    This one only evaluates f(arr1[cond], arr2[cond], ...).

    Examples
    --------
    >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
    >>> def f(a, b):
    ...     return a*b
    >>> _lazywhere(a > 2, (a, b), f, np.nan)
    array([ nan,  nan,  21.,  32.])

    Notice, it assumes that all `arrays` are of the same shape, or can be
    broadcasted together.

    """
    cond = np.asarray(cond)
    if fillvalue is None:
        if f2 is None:
            raise ValueError("One of (fillvalue, f2) must be given.")
        else:
            fillvalue = np.nan
    else:
        if f2 is not None:
            raise ValueError("Only one of (fillvalue, f2) can be given.")

    args = np.broadcast_arrays(cond, *arrays)
    cond, arrays = args[0], args[1:]
    temp = tuple(np.extract(cond, arr) for arr in arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=fillvalue, dtype=tcode)
    np.place(out, cond, f(*temp))
    if f2 is not None:
        temp = tuple(np.extract(~cond, arr) for arr in arrays)
        np.place(out, ~cond, f2(*temp))

    return out


def _lazyselect(condlist, choicelist, arrays, default=0):
    """
    Mimic `np.select(condlist, choicelist)`.

    Notice, it assumes that all `arrays` are of the same shape or can be
    broadcasted together.

    All functions in `choicelist` must accept array arguments in the order
    given in `arrays` and must return an array of the same shape as broadcasted
    `arrays`.

    Examples
    --------
    >>> x = np.arange(6)
    >>> np.select([x <3, x > 3], [x**2, x**3], default=0)
    array([  0,   1,   4,   0,  64, 125])

    >>> _lazyselect([x < 3, x > 3], [lambda x: x**2, lambda x: x**3], (x,))
    array([   0.,    1.,    4.,   0.,   64.,  125.])

    >>> a = -np.ones_like(x)
    >>> _lazyselect([x < 3, x > 3],
    ...             [lambda x, a: x**2, lambda x, a: a * x**3],
    ...             (x, a), default=np.nan)
    array([   0.,    1.,    4.,   nan,  -64., -125.])

    """
    arrays = np.broadcast_arrays(*arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=default, dtype=tcode)
    for func, cond in zip(choicelist, condlist):
        if np.all(cond is False):
            continue
        cond, _ = np.broadcast_arrays(cond, arrays[0])
        temp = tuple(np.extract(cond, arr) for arr in arrays)
        np.place(out, cond, func(*temp))
    return out


def _aligned_zeros(shape, dtype=float, order="C", align=None):
    """Allocate a new ndarray with aligned memory.

    Primary use case for this currently is working around a f2py issue
    in NumPy 1.9.1, where dtype.alignment is such that np.zeros() does
    not necessarily create arrays aligned up to it.

    """
    dtype = np.dtype(dtype)
    if align is None:
        align = dtype.alignment
    if not hasattr(shape, '__len__'):
        shape = (shape,)
    size = functools.reduce(operator.mul, shape) * dtype.itemsize
    buf = np.empty(size + align + 1, np.uint8)
    offset = buf.__array_interface__['data'][0] % align
    if offset != 0:
        offset = align - offset
    # Note: slices producing 0-size arrays do not necessarily change
    # data pointer --- so we use and allocate size+1
    buf = buf[offset:offset+size+1][:-1]
    data = np.ndarray(shape, dtype, buf, order=order)
    data.fill(0)
    return data


def _prune_array(array):
    """Return an array equivalent to the input array. If the input
    array is a view of a much larger array, copy its contents to a
    newly allocated array. Otherwise, return the input unchanged.
    """
    if array.base is not None and array.size < array.base.size // 2:
        return array.copy()
    return array


def prod(iterable):
    """
    Product of a sequence of numbers.

    Faster than np.prod for short lists like array shapes, and does
    not overflow if using Python integers.
    """
    product = 1
    for x in iterable:
        product *= x
    return product


def float_factorial(n: int) -> float:
    """Compute the factorial and return as a float

    Returns infinity when result is too large for a double
    """
    return float(math.factorial(n)) if n < 171 else np.inf


class DeprecatedImport:
    """
    Deprecated import with redirection and warning.

    Examples
    --------
    Suppose you previously had in some module::

        from foo import spam

    If this has to be deprecated, do::

        spam = DeprecatedImport("foo.spam", "baz")

    to redirect users to use "baz" module instead.

    """

    def __init__(self, old_module_name, new_module_name):
        self._old_name = old_module_name
        self._new_name = new_module_name
        __import__(self._new_name)
        self._mod = sys.modules[self._new_name]

    def __dir__(self):
        return dir(self._mod)

    def __getattr__(self, name):
        warnings.warn("Module %s is deprecated, use %s instead"
                      % (self._old_name, self._new_name),
                      DeprecationWarning)
        return getattr(self._mod, name)


# copy-pasted from scikit-learn utils/validation.py
# change this to scipy.stats._qmc.check_random_state once numpy 1.16 is dropped
def check_random_state(seed):
    """Turn `seed` into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed

    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _asarray_validated(a, check_finite=True,
                       sparse_ok=False, objects_ok=False, mask_ok=False,
                       as_inexact=False):
    """
    Helper function for SciPy argument validation.

    Many SciPy linear algebra functions do support arbitrary array-like
    input arguments. Examples of commonly unsupported inputs include
    matrices containing inf/nan, sparse matrix representations, and
    matrices with complicated elements.

    Parameters
    ----------
    a : array_like
        The array-like input.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    sparse_ok : bool, optional
        True if scipy sparse matrices are allowed.
    objects_ok : bool, optional
        True if arrays with dype('O') are allowed.
    mask_ok : bool, optional
        True if masked arrays are allowed.
    as_inexact : bool, optional
        True to convert the input array to a np.inexact dtype.

    Returns
    -------
    ret : ndarray
        The converted validated array.

    """
    if not sparse_ok:
        import scipy.sparse
        if scipy.sparse.issparse(a):
            msg = ('Sparse matrices are not supported by this function. '
                   'Perhaps one of the scipy.sparse.linalg functions '
                   'would work instead.')
            raise ValueError(msg)
    if not mask_ok:
        if np.ma.isMaskedArray(a):
            raise ValueError('masked arrays are not supported')
    toarray = np.asarray_chkfinite if check_finite else np.asarray
    a = toarray(a)
    if not objects_ok:
        if a.dtype is np.dtype('O'):
            raise ValueError('object arrays are not supported')
    if as_inexact:
        if not np.issubdtype(a.dtype, np.inexact):
            a = toarray(a, dtype=np.float_)
    return a


def _validate_int(k, name, minimum=None):
    """
    Validate a scalar integer.

    This functon can be used to validate an argument to a function
    that expects the value to be an integer.  It uses `operator.index`
    to validate the value (so, for example, k=2.0 results in a
    TypeError).

    Parameters
    ----------
    k : int
        The value to be validated.
    name : str
        The name of the parameter.
    minimum : int, optional
        An optional lower bound.
    """
    try:
        k = operator.index(k)
    except TypeError:
        raise TypeError(f'{name} must be an integer.') from None
    if minimum is not None and k < minimum:
        raise ValueError(f'{name} must be an integer not less '
                         f'than {minimum}') from None
    return k


# Add a replacement for inspect.getfullargspec()/
# The version below is borrowed from Django,
# https://github.com/django/django/pull/4846.

# Note an inconsistency between inspect.getfullargspec(func) and
# inspect.signature(func). If `func` is a bound method, the latter does *not*
# list `self` as a first argument, while the former *does*.
# Hence, cook up a common ground replacement: `getfullargspec_no_self` which
# mimics `inspect.getfullargspec` but does not list `self`.
#
# This way, the caller code does not need to know whether it uses a legacy
# .getfullargspec or a bright and shiny .signature.

FullArgSpec = namedtuple('FullArgSpec',
                         ['args', 'varargs', 'varkw', 'defaults',
                          'kwonlyargs', 'kwonlydefaults', 'annotations'])


def getfullargspec_no_self(func):
    """inspect.getfullargspec replacement using inspect.signature.

    If func is a bound method, do not list the 'self' parameter.

    Parameters
    ----------
    func : callable
        A callable to inspect

    Returns
    -------
    fullargspec : FullArgSpec(args, varargs, varkw, defaults, kwonlyargs,
                              kwonlydefaults, annotations)

        NOTE: if the first argument of `func` is self, it is *not*, I repeat
        *not*, included in fullargspec.args.
        This is done for consistency between inspect.getargspec() under
        Python 2.x, and inspect.signature() under Python 3.x.

    """
    sig = inspect.signature(func)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY]
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    varkw = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    varkw = varkw[0] if varkw else None
    defaults = tuple(
        p.default for p in sig.parameters.values()
        if (p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
            p.default is not p.empty)
    ) or None
    kwonlyargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    kwdefaults = {p.name: p.default for p in sig.parameters.values()
                  if p.kind == inspect.Parameter.KEYWORD_ONLY and
                  p.default is not p.empty}
    annotations = {p.name: p.annotation for p in sig.parameters.values()
                   if p.annotation is not p.empty}
    return FullArgSpec(args, varargs, varkw, defaults, kwonlyargs,
                       kwdefaults or None, annotations)


class _FunctionWrapper:
    """
    Object to wrap user's function, allowing picklability
    """
    def __init__(self, f, args):
        self.f = f
        self.args = [] if args is None else args

    def __call__(self, x):
        #print(x.reshape(1,-1).shape)
        return self.f(x.reshape(1, -1), *self.args)


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
            if int(pool) == -1:
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


def rng_integers(gen, low, high=None, size=None, dtype='int64',
                 endpoint=False):
    """
    Return random integers from low (inclusive) to high (exclusive), or if
    endpoint=True, low (inclusive) to high (inclusive). Replaces
    `RandomState.randint` (with endpoint=False) and
    `RandomState.random_integers` (with endpoint=True).

    Return random integers from the "discrete uniform" distribution of the
    specified dtype. If high is None (the default), then results are from
    0 to low.

    Parameters
    ----------
    gen : {None, np.random.RandomState, np.random.Generator}
        Random number generator. If None, then the np.random.RandomState
        singleton is used.
    low : int or array-like of ints
        Lowest (signed) integers to be drawn from the distribution (unless
        high=None, in which case this parameter is 0 and this value is used
        for high).
    high : int or array-like of ints
        If provided, one above the largest (signed) integer to be drawn from
        the distribution (see above for behavior if high=None). If array-like,
        must contain integer values.
    size : array-like of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
        samples are drawn. Default is None, in which case a single value is
        returned.
    dtype : {str, dtype}, optional
        Desired dtype of the result. All dtypes are determined by their name,
        i.e., 'int64', 'int', etc, so byteorder is not available and a specific
        precision may have different C types depending on the platform.
        The default value is np.int_.
    endpoint : bool, optional
        If True, sample from the interval [low, high] instead of the default
        [low, high) Defaults to False.

    Returns
    -------
    out: int or ndarray of ints
        size-shaped array of random integers from the appropriate distribution,
        or a single such random int if size not provided.
    """
    if isinstance(gen, Generator):
        return gen.integers(low, high=high, size=size, dtype=dtype,
                            endpoint=endpoint)
    else:
        if gen is None:
            # default is RandomState singleton used by np.random.
            gen = np.random.mtrand._rand
        if endpoint:
            # inclusive of endpoint
            # remember that low and high can be arrays, so don't modify in
            # place
            if high is None:
                return gen.randint(low + 1, size=size, dtype=dtype)
            if high is not None:
                return gen.randint(low, high=high + 1, size=size, dtype=dtype)

        # exclusive
        return gen.randint(low, high=high, size=size, dtype=dtype)


@contextmanager
def _fixed_default_rng(seed=1638083107694713882823079058616272161):
    """Context with a fixed np.random.default_rng seed."""
    orig_fun = np.random.default_rng
    np.random.default_rng = lambda seed=seed: orig_fun(seed)
    try:
        yield
    finally:
        np.random.default_rng = orig_fun


def _argmin(a, keepdims=False, axis=None):
    """
    argmin with a `keepdims` parameter.

    See https://github.com/numpy/numpy/issues/8710

    If axis is not None, a.shape[axis] must be greater than 0.
    """
    res = np.argmin(a, axis=axis)
    if keepdims and axis is not None:
        res = np.expand_dims(res, axis=axis)
    return res


def _first_nonnan(a, axis):
    """
    Return the first non-nan value along the given axis.

    If a slice is all nan, nan is returned for that slice.

    The shape of the return value corresponds to ``keepdims=True``.

    Examples
    --------
    >>> nan = np.nan
    >>> a = np.array([[ 3.,  3., nan,  3.],
                      [ 1., nan,  2.,  4.],
                      [nan, nan,  9., -1.],
                      [nan,  5.,  4.,  3.],
                      [ 2.,  2.,  2.,  2.],
                      [nan, nan, nan, nan]])
    >>> _first_nonnan(a, axis=0)
    array([[3., 3., 2., 3.]])
    >>> _first_nonnan(a, axis=1)
    array([[ 3.],
           [ 1.],
           [ 9.],
           [ 5.],
           [ 2.],
           [nan]])
    """
    k = _argmin(np.isnan(a), axis=axis, keepdims=True)
    return np.take_along_axis(a, k, axis=axis)


def _nan_allsame(a, axis, keepdims=False):
    """
    Determine if the values along an axis are all the same.

    nan values are ignored.

    `a` must be a numpy array.

    `axis` is assumed to be normalized; that is, 0 <= axis < a.ndim.

    For an axis of length 0, the result is True.  That is, we adopt the
    convention that ``allsame([])`` is True. (There are no values in the
    input that are different.)

    `True` is returned for slices that are all nan--not because all the
    values are the same, but because this is equivalent to ``allsame([])``.

    Examples
    --------
    >>> a
    array([[ 3.,  3., nan,  3.],
           [ 1., nan,  2.,  4.],
           [nan, nan,  9., -1.],
           [nan,  5.,  4.,  3.],
           [ 2.,  2.,  2.,  2.],
           [nan, nan, nan, nan]])
    >>> _nan_allsame(a, axis=1, keepdims=True)
    array([[ True],
           [False],
           [False],
           [False],
           [ True],
           [ True]])
    """
    if axis is None:
        if a.size == 0:
            return True
        a = a.ravel()
        axis = 0
    else:
        shp = a.shape
        if shp[axis] == 0:
            shp = shp[:axis] + (1,)*keepdims + shp[axis + 1:]
            return np.full(shp, fill_value=True, dtype=bool)
    a0 = _first_nonnan(a, axis=axis)
    return ((a0 == a) | np.isnan(a)).all(axis=axis, keepdims=keepdims)


def _rename_parameter(old_name, new_name, dep_version=None):
    """
    Generate decorator for backward-compatible keyword renaming.

    Apply the decorator generated by `_rename_parameter` to functions with a
    recently renamed parameter to maintain backward-compatibility.

    After decoration, the function behaves as follows:
    If only the new parameter is passed into the function, behave as usual.
    If only the old parameter is passed into the function (as a keyword), raise
    a DeprecationWarning if `dep_version` is provided, and behave as usual
    otherwise.
    If both old and new parameters are passed into the function, raise a
    DeprecationWarning if `dep_version` is provided, and raise the appropriate
    TypeError (function got multiple values for argument).

    Parameters
    ----------
    old_name : str
        Old name of parameter
    new_name : str
        New name of parameter
    dep_version : str, optional
        Version of SciPy in which old parameter was deprecated in the format
        'X.Y.Z'. If supplied, the deprecation message will indicate that
        support for the old parameter will be removed in version 'X.Y+2.Z'

    Notes
    -----
    Untested with functions that accept *args. Probably won't work as written.

    """
    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            if old_name in kwargs:
                if dep_version:
                    end_version = dep_version.split('.')
                    end_version[1] = str(int(end_version[1]) + 2)
                    end_version = '.'.join(end_version)
                    message = (f"Use of keyword argument `{old_name}` is "
                               f"deprecated and replaced by `{new_name}`.  "
                               f"Support for `{old_name}` will be removed "
                               f"in SciPy {end_version}.")
                    warnings.warn(message, DeprecationWarning, stacklevel=2)
                if new_name in kwargs:
                    message = (f"{fun.__name__}() got multiple values for "
                               f"argument now known as `{new_name}`")
                    raise TypeError(message)
                kwargs[new_name] = kwargs.pop(old_name)
            return fun(*args, **kwargs)
        return wrapper
    return decorator


"""
differential_evolution: The differential evolution global optimization algorithm
Added by Andrew Nelson 2014
"""
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize
# from scipy.optimize._optimize import _status_message
# from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper

from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
                                         NonlinearConstraint, LinearConstraint)
from scipy.sparse import issparse

__all__ = ['differential_evolution']

_MACHEPS = np.finfo(np.float64).eps


def differential_evolution(func, bounds, args=(), strategy='rand1bin',
                           maxiter=30000, popsize=100, tol=0,
                           mutation=0.5, recombination=0.9, seed=None,
                           callback=None, disp=True, polish=False,
                           init='random', atol=0, updating='deferred',
                           workers=1, constraints=(), x0=None, *,
                           integrality=None, vectorized=False, block_size=0, save_link=None):
    # using a context manager means that any created Pool objects are
    # cleared up.
    with DifferentialEvolutionSolver(func, bounds, args=args,
                                     strategy=strategy,
                                     maxiter=maxiter,
                                     popsize=popsize, tol=tol,
                                     mutation=mutation,
                                     recombination=recombination,
                                     seed=seed, polish=polish,
                                     callback=callback,
                                     disp=disp, init=init, atol=atol,
                                     updating=updating,
                                     workers=workers,
                                     constraints=constraints,
                                     x0=x0,
                                     integrality=integrality,
                                     vectorized=vectorized,
                                     block_size=block_size,
                                     save_link=save_link) as solver:
        ret = solver.solve()

    return ret


class DifferentialEvolutionSolver:
    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'currenttobest1bin': '_currenttobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'currenttobest1exp': '_currenttobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    __init_error_msg = ("The population initialization method must be one of "
                        "'latinhypercube' or 'random', or an array of shape "
                        "(S, N) where N is the number of parameters and S>5")

    def __init__(self, func, bounds, args=(),
                 strategy='rand1bin', maxiter=1000, popsize=100,
                 tol=0, mutation=0.5, recombination=0.9, seed=None,
                 maxfun=np.inf, callback=None, disp=False, polish=False,
                 init='random', atol=0, updating='deferred',
                 workers=1, constraints=(), x0=None, *,
                 integrality=None, vectorized=False, block_size=0, save_link=None):

        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy

        self.callback = callback
        self.polish = polish

        # set the updating / parallelisation options
        if updating in ['immediate', 'deferred']:
            self._updating = updating

        self.vectorized = vectorized

        # want to use parallelisation, but updating is immediate
        if workers != 1 and updating == 'immediate':
            warnings.warn("differential_evolution: the 'workers' keyword has"
                          " overridden updating='immediate' to"
                          " updating='deferred'", UserWarning, stacklevel=2)
            self._updating = 'deferred'

        if vectorized and workers != 1:
            warnings.warn("differential_evolution: the 'workers' keyword"
                          " overrides the 'vectorized' keyword", stacklevel=2)
            self.vectorized = vectorized = False

        if vectorized and updating == 'immediate':
            warnings.warn("differential_evolution: the 'vectorized' keyword"
                          " has overridden updating='immediate' to updating"
                          "='deferred'", UserWarning, stacklevel=2)
            self._updating = 'deferred'

        # an object with a map method.
        if vectorized:
            def maplike_for_vectorized_func(func, x):
                # send an array (N, S) to the user func,
                # expect to receive (S,). Transposition is required because
                # internally the population is held as (S, N)
                return np.atleast_1d(func(x.T))

            workers = maplike_for_vectorized_func

        self._mapwrapper = MapWrapper(workers)

        # relative and absolute tolerances for convergence
        self.tol, self.atol = tol, atol

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            raise ValueError('The mutation constant must be a float in '
                             'U[0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in U[0, 2).')

        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        # we create a wrapped function to allow the use of map (and Pool.map
        # in the future)
        self.func = _FunctionWrapper(func, args)
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        if isinstance(bounds, Bounds):
            self.limits = np.array(new_bounds_to_old(bounds.lb,
                                                     bounds.ub,
                                                     len(bounds.lb)),
                                   dtype=float).T
        else:
            self.limits = np.array(bounds, dtype='float').T

        if (np.size(self.limits, 0) != 2 or not
        np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence containing '
                             'real valued (min, max) pairs for each value'
                             ' in x')

        if maxiter is None:  # the default used to be None
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:  # the default used to be None
            maxfun = np.inf
        self.maxfun = maxfun

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.parameter_count = np.size(self.limits, 1)

        self.random_number_generator = check_random_state(seed)

        # Which parameters are going to be integers?
        if np.any(integrality):
            # # user has provided a truth value for integer constraints
            integrality = np.broadcast_to(
                integrality,
                self.parameter_count
            )
            integrality = np.asarray(integrality, bool)
            # For integrality parameters change the limits to only allow
            # integer values lying between the limits.
            lb, ub = np.copy(self.limits)

            lb = np.ceil(lb)
            ub = np.floor(ub)
            if not (lb[integrality] <= ub[integrality]).all():
                # there's a parameter that doesn't have an integer value
                # lying between the limits
                raise ValueError("One of the integrality constraints does not"
                                 " have any possible integer values between"
                                 " the lower/upper bounds.")
            nlb = np.nextafter(lb[integrality] - 0.5, np.inf)
            nub = np.nextafter(ub[integrality] + 0.5, -np.inf)

            self.integrality = integrality
            self.limits[0, self.integrality] = nlb
            self.limits[1, self.integrality] = nub
        else:
            self.integrality = False

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.
        # the minimum is 5 because 'best2bin' requires a population that's at
        # least 5 long
        self.num_population_members = max(5, popsize)
        self.population_shape = (self.num_population_members,
                                 self.parameter_count)
        self.block_size = block_size
        self._nfev = 0
        # check first str otherwise will fail to compare str with array
        if isinstance(init, str):
            if init == 'latinhypercube':
                self.init_population_lhs()
            elif init == 'sobol':
                # must be Ns = 2**m for Sobol'
                n_s = int(2 ** np.ceil(np.log2(self.num_population_members)))
                self.num_population_members = n_s
                self.population_shape = (self.num_population_members,
                                         self.parameter_count)
                self.init_population_qmc(qmc_engine='sobol')
            elif init == 'halton':
                self.init_population_qmc(qmc_engine='halton')
            elif init == 'random':
                self.init_population_random()
            else:
                raise ValueError(self.__init_error_msg)
        else:
            self.init_population_array(init)

        if x0 is not None:
            # scale to within unit interval and
            # ensure parameters are within bounds.
            x0_scaled = self._unscale_parameters(np.asarray(x0))
            if ((x0_scaled > 1.0) | (x0_scaled < 0.0)).any():
                raise ValueError(
                    "Some entries in x0 lay outside the specified bounds"
                )
            self.population[0] = x0_scaled

        # infrastructure for constraints
        self.constraints = constraints
        self._wrapped_constraints = []

        if hasattr(constraints, '__len__'):
            # sequence of constraints, this will also deal with default
            # keyword parameter
            for c in constraints:
                self._wrapped_constraints.append(
                    _ConstraintWrapper(c, self.x)
                )
        else:
            self._wrapped_constraints = [
                _ConstraintWrapper(constraints, self.x)
            ]
        self.total_constraints = np.sum(
            [c.num_constr for c in self._wrapped_constraints]
        )
        self.constraint_violation = np.zeros((self.num_population_members, 1))
        self.feasible = np.ones(self.num_population_members, bool)

        self.disp = disp
        self.save_link = save_link
        self.best_sol_generations_fit = []

    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the following
        # size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (segsize * rng.uniform(size=self.population_shape)

                   # Offset each segment to cover the entire parameter range [0, 1)
                   + np.linspace(0., 1., self.num_population_members,
                                 endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        self.population = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_qmc(self, qmc_engine):
        """Initializes the population with a QMC method.

        QMC methods ensures that each parameter is uniformly
        sampled over its range.

        Parameters
        ----------
        qmc_engine : str
            The QMC method to use for initialization. Can be one of
            ``latinhypercube``, ``sobol`` or ``halton``.

        """
        from scipy.stats import qmc

        rng = self.random_number_generator

        # Create an array for population of candidate solutions.
        if qmc_engine == 'latinhypercube':
            sampler = qmc.LatinHypercube(d=self.parameter_count, seed=rng)
        elif qmc_engine == 'sobol':
            sampler = qmc.Sobol(d=self.parameter_count, seed=rng)
        elif qmc_engine == 'halton':
            sampler = qmc.Halton(d=self.parameter_count, seed=rng)
        else:
            raise ValueError(self.__init_error_msg)

        self.population = sampler.random(n=self.num_population_members)

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_random(self):
        """
        Initializes the population at random. This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.uniform(low=self.limits[0], high=self.limits[1], size=self.population_shape)

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_array(self, init):
        """
        Initializes the population with a user specified population.

        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (S, N), where N is the number of parameters.
            The population is clipped to the lower and upper bounds.
        """
        # make sure you're using a float array
        popn = np.asfarray(init)

        if (np.size(popn, 0) < 5 or
                popn.shape[1] != self.parameter_count or
                len(popn.shape) != 2):
            raise ValueError("The population supplied needs to have shape"
                             " (S, len(x)), where S > 4.")

        # scale values and clip to bounds, assigning to population
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)

        self.num_population_members = np.size(self.population, 0)

        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver
        """
        return self.population[self.population_energies.argmin()]

    @property
    def convergence(self):
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        if np.any(np.isinf(self.population_energies)):
            return np.inf
        return (np.std(self.population_energies) /
                np.abs(np.mean(self.population_energies) + _MACHEPS))

    def converged(self):
        """
        Return True if the solver has converged.
        """
        if np.any(np.isinf(self.population_energies)):
            return False

        return (np.std(self.population_energies) <=
                self.atol +
                self.tol * np.abs(np.mean(self.population_energies)))

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """
        nit, warning_flag = 0, False
        status_message = _status_message['success']

        # The population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).

        # do the optimization.
        for nit in range(1, self.maxiter):
            # evolve the population by a generation
            try:
                next(self)
                self.best_sol_generations_fit.append(self.population_energies.min())
            except StopIteration:
                warning_flag = True
                if self._nfev > self.maxfun:
                    status_message = _status_message['maxfev']
                elif self._nfev == self.maxfun:
                    status_message = ('Maximum number of function evaluations'
                                      ' has been reached.')
                self.best_sol_generations_fit.append(self.population_energies.min())
                break

            if self.disp:
                print("differential_evolution step %d: f(x)= %g"
                      % (nit,
                         self.population_energies.min()))

            """if self.callback:
                #c = self.tol / (self.convergence + _MACHEPS)
                #warning_flag = bool(self.callback(self.x, convergence=c))
                if warning_flag:
                    status_message = ('callback function requested stop early'
                                      ' by returning True')"""

        # np.savez(self.save_link, best_sol_generations_fit=self.best_sol_generations_fit)

        DE_result = OptimizeResult(
            x=self.x,
            fun=self.population_energies.min(),
            nfev=self._nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True))

        if self.polish and not np.all(self.integrality):
            # can't polish if all the parameters are integers
            if np.any(self.integrality):
                # set the lower/upper bounds equal so that any integrality
                # constraints work.
                limits, integrality = self.limits, self.integrality
                limits[0, integrality] = DE_result.x[integrality]
                limits[1, integrality] = DE_result.x[integrality]

            polish_method = 'L-BFGS-B'

            if self._wrapped_constraints:
                polish_method = 'trust-constr'

                constr_violation = self._constraint_violation_fn(DE_result.x)
                if np.any(constr_violation > 0.):
                    warnings.warn("differential evolution didn't find a"
                                  " solution satisfying the constraints,"
                                  " attempting to polish from the least"
                                  " infeasible solution", UserWarning)

            result = minimize(self.func,
                              np.copy(DE_result.x),
                              method=polish_method,
                              bounds=self.limits.T,
                              constraints=self.constraints)

            self._nfev += result.nfev
            DE_result.nfev = self._nfev

            # Polishing solution is only accepted if there is an improvement in
            # cost function, the polishing was successful and the solution lies
            # within the bounds.
            if (result.fun < DE_result.fun and
                    result.success and
                    np.all(result.x <= self.limits[1]) and
                    np.all(self.limits[0] <= result.x)):
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = result.x

        if self._wrapped_constraints:
            DE_result.constr = [c.violation(DE_result.x) for
                                c in self._wrapped_constraints]
            DE_result.constr_violation = np.max(
                np.concatenate(DE_result.constr))
            DE_result.maxcv = DE_result.constr_violation
            if DE_result.maxcv > 0:
                # if the result is infeasible then success must be False
                DE_result.success = False
                DE_result.message = ("The solution does not satisfy the "
                                     f"constraints, MAXCV = {DE_result.maxcv}")

        return DE_result

    def _calculate_population_energies(self, population):
        # print(population[0])
        """
        Calculate the energies of a population.

        Parameters
        ----------
        population : ndarray
            An array of parameter vectors normalised to [0, 1] using lower
            and upper limits. Has shape ``(np.size(population, 0), N)``.

        Returns
        -------
        energies : ndarray
            An array of energies corresponding to each population member. If
            maxfun will be exceeded during this call, then the number of
            function evaluations will be reduced and energies will be
            right-padded with np.inf. Has shape ``(np.size(population, 0),)``
        """
        num_members = np.size(population, 0)
        # S is the number of function evals left to stay under the
        # maxfun budget
        S = min(num_members, self.maxfun - self._nfev)

        energies = np.full(num_members, np.inf)

        parameters_pop = self.population
        try:
            calc_energies = list(
                self._mapwrapper(self.func, parameters_pop[0:S])
            )
            calc_energies = np.squeeze(calc_energies)
        except (TypeError, ValueError) as e:
            # wrong number of arguments for _mapwrapper
            # or wrong length returned from the mapper
            raise RuntimeError(
                "The map-like callable must be of the form f(func, iterable), "
                "returning a sequence of numbers the same length as 'iterable'"
            ) from e

        if calc_energies.size != S:
            if self.vectorized:
                raise RuntimeError("The vectorized function must return an"
                                   " array of shape (S,) when given an array"
                                   " of shape (len(x), S)")
            raise RuntimeError("func(x, *args) must return a scalar value")

        energies[0:S] = calc_energies
        # print(calc_energies.min())
        print("best f(x)= %g, new best f(x)= %g"
              % (self.population_energies.min(), calc_energies.min()))

        if self.vectorized:
            self._nfev += 1
        else:
            self._nfev += S

        return energies

    def _promote_lowest_energy(self):
        # swaps 'best solution' into first population entry

        idx = np.arange(self.num_population_members)
        feasible_solutions = idx[self.feasible]
        if feasible_solutions.size:
            # find the best feasible solution
            idx_t = np.argmin(self.population_energies[feasible_solutions])
            l = feasible_solutions[idx_t]
        else:
            # no solution was feasible, use 'best' infeasible solution, which
            # will violate constraints the least
            l = np.argmin(np.sum(self.constraint_violation, axis=1))

        self.population_energies[[0, l]] = self.population_energies[[l, 0]]
        self.population[[0, l], :] = self.population[[l, 0], :]
        self.feasible[[0, l]] = self.feasible[[l, 0]]
        self.constraint_violation[[0, l], :] = (
            self.constraint_violation[[l, 0], :])

    def _constraint_violation_fn(self, x):
        """
        Calculates total constraint violation for all the constraints, for a
        set of solutions.

        Parameters
        ----------
        x : ndarray
            Solution vector(s). Has shape (S, N), or (N,), where S is the
            number of solutions to investigate and N is the number of
            parameters.

        Returns
        -------
        cv : ndarray
            Total violation of constraints. Has shape ``(S, M)``, where M is
            the total number of constraint components (which is not necessarily
            equal to len(self._wrapped_constraints)).
        """
        # how many solution vectors you're calculating constraint violations
        # for
        S = np.size(x) // self.parameter_count
        _out = np.zeros((S, self.total_constraints))
        offset = 0
        for con in self._wrapped_constraints:
            # the input/output of the (vectorized) constraint function is
            # {(N, S), (N,)} --> (M, S)
            # The input to _constraint_violation_fn is (S, N) or (N,), so
            # transpose to pass it to the constraint. The output is transposed
            # from (M, S) to (S, M) for further use.
            c = con.violation(x.T).T

            # The shape of c should be (M,), (1, M), or (S, M). Check for
            # those shapes, as an incorrect shape indicates that the
            # user constraint function didn't return the right thing, and
            # the reshape operation will fail. Intercept the wrong shape
            # to give a reasonable error message. I'm not sure what failure
            # modes an inventive user will come up with.
            if c.shape[-1] != con.num_constr or (S > 1 and c.shape[0] != S):
                raise RuntimeError("An array returned from a Constraint has"
                                   " the wrong shape. If `vectorized is False`"
                                   " the Constraint should return an array of"
                                   " shape (M,). If `vectorized is True` then"
                                   " the Constraint must return an array of"
                                   " shape (M, S), where S is the number of"
                                   " solution vectors and M is the number of"
                                   " constraint components in a given"
                                   " Constraint object.")

            # the violation function may return a 1D array, but is it a
            # sequence of constraints for one solution (S=1, M>=1), or the
            # value of a single constraint for a sequence of solutions
            # (S>=1, M=1)
            c = np.reshape(c, (S, con.num_constr))
            _out[:, offset:offset + con.num_constr] = c
            offset += con.num_constr

        return _out

    def _calculate_population_feasibilities(self, population):
        """
        Calculate the feasibilities of a population.

        Parameters
        ----------
        population : ndarray
            An array of parameter vectors normalised to [0, 1] using lower
            and upper limits. Has shape ``(np.size(population, 0), N)``.

        Returns
        -------
        feasible, constraint_violation : ndarray, ndarray
            Boolean array of feasibility for each population member, and an
            array of the constraint violation for each population member.
            constraint_violation has shape ``(np.size(population, 0), M)``,
            where M is the number of constraints.
        """
        num_members = np.size(population, 0)
        if not self._wrapped_constraints:
            # shortcut for no constraints
            return np.ones(num_members, bool), np.zeros((num_members, 1))

        # (S, N)
        parameters_pop = population

        if self.vectorized:
            # (S, M)
            constraint_violation = np.array(
                self._constraint_violation_fn(parameters_pop)
            )
        else:
            # (S, 1, M)
            constraint_violation = np.array([self._constraint_violation_fn(x)
                                             for x in parameters_pop])
            # if you use the list comprehension in the line above it will
            # create an array of shape (S, 1, M), because each iteration
            # generates an array of (1, M). In comparison the vectorized
            # version returns (S, M). It's therefore necessary to remove axis 1
            constraint_violation = constraint_violation[:, 0]

        feasible = ~(np.sum(constraint_violation, axis=1) > 0)

        return feasible, constraint_violation

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._mapwrapper.__exit__(*args)

    def _accept_trial(self, energy_trial, feasible_trial, cv_trial,
                      energy_orig, feasible_orig, cv_orig):
        """
        Trial is accepted if:
        * it satisfies all constraints and provides a lower or equal objective
          function value, while both the compared solutions are feasible
        - or -
        * it is feasible while the original solution is infeasible,
        - or -
        * it is infeasible, but provides a lower or equal constraint violation
          for all constraint functions.

        This test corresponds to section III of Lampinen [1]_.

        Parameters
        ----------
        energy_trial : float
            Energy of the trial solution
        feasible_trial : float
            Feasibility of trial solution
        cv_trial : array-like
            Excess constraint violation for the trial solution
        energy_orig : float
            Energy of the original solution
        feasible_orig : float
            Feasibility of original solution
        cv_orig : array-like
            Excess constraint violation for the original solution

        Returns
        -------
        accepted : bool

        """
        return energy_trial <= energy_orig
        # if feasible_orig and feasible_trial:
        #     return energy_trial <= energy_orig
        # elif feasible_trial and not feasible_orig:
        #     return True
        # elif not feasible_trial and (cv_trial <= cv_orig).all():
        #     # cv_trial < cv_orig would imply that both trial and orig are not
        #     # feasible
        #     return True
        #
        # return False

    def __next__(self):
        """
        Evolve the population by a single generation

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        if np.all(np.isinf(self.population_energies)):
            # self.feasible, self.constraint_violation = (
            #    self._calculate_population_feasibilities(self.population))

            # only need to work out population energies for those that are
            # feasible
            self.population_energies = (
                self._calculate_population_energies(
                    self.population))

            self.best_sol_generations_fit.append(self.population_energies.min())

            # self._promote_lowest_energy()

        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(self.dither[0],
                                                              self.dither[1])

        if self._updating == 'immediate':
            # update best solution immediately
            for candidate in range(self.num_population_members):
                if self._nfev > self.maxfun:
                    raise StopIteration

                # create a trial solution
                trial = self._mutate(candidate)

                # ensuring that it's in the range [0, 1)
                # self._ensure_constraint(trial)

                # scale from [0, 1) to the actual parameter value
                parameters = trial

                # determine the energy of the objective function
                if self._wrapped_constraints:
                    cv = self._constraint_violation_fn(parameters)
                    feasible = False
                    energy = np.inf
                    if not np.sum(cv) > 0:
                        # solution is feasible
                        feasible = True
                        energy = self.func(parameters)
                        self._nfev += 1
                else:
                    feasible = True
                    cv = np.atleast_2d([0.])
                    energy = self.func(parameters)
                    self._nfev += 1

                # compare trial and population member
                if self._accept_trial(energy, feasible, cv,
                                      self.population_energies[candidate],
                                      self.feasible[candidate],
                                      self.constraint_violation[candidate]):
                    self.population[candidate] = trial
                    self.population_energies[candidate] = np.squeeze(energy)
                    self.feasible[candidate] = feasible
                    self.constraint_violation[candidate] = cv

                    # if the trial candidate is also better than the best
                    # solution then promote it.
                    if self._accept_trial(energy, feasible, cv,
                                          self.population_energies[0],
                                          self.feasible[0],
                                          self.constraint_violation[0]):
                        self._promote_lowest_energy()

        elif self._updating == 'deferred':
            # update best solution once per generation
            if self._nfev >= self.maxfun:
                raise StopIteration

            # 'deferred' approach, vectorised form.
            # create trial solutions
            trial_pop = np.array(
                [self._mutate(i) for i in range(self.num_population_members)])

            # only calculate for feasible entries
            # trial_energies = self._calculate_population_energies(trial_pop)

            # determine the energies of the objective function, but only for
            # feasible trials
            # feasible, cv = self._calculate_population_feasibilities(trial_pop)
            # trial_energies = np.full(self.num_population_members, np.inf)

            # only calculate for feasible entries
            trial_energies = self._calculate_population_energies(trial_pop)

            # which solutions are 'improved'?
            # loc = [self._accept_trial(*val) for val in
            #        zip(trial_energies, feasible, cv, self.population_energies,
            #            self.feasible, self.constraint_violation)]
            loc = trial_energies <= self.population_energies
            loc = np.array(loc)
            # print(loc)
            self.population = np.where(loc[:, np.newaxis],
                                       trial_pop,
                                       self.population)

            self.population_energies = np.where(loc,
                                                trial_energies,
                                                self.population_energies)

            # self.feasible = np.where(loc,
            #                          feasible,
            #                          self.feasible)
            # self.constraint_violation = np.where(loc[:, np.newaxis],
            #                                      cv,
            #                                      self.constraint_violation)

        return self.population[self.population_energies.argmin()], self.population_energies.min()

    def _scale_parameters(self, trial):
        """Scale from a number between 0 and 1 to parameters."""
        # trial either has shape (N, ) or (L, N), where L is the number of
        # solutions being scaled
        scaled = self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2
        if np.any(self.integrality):
            i = np.broadcast_to(self.integrality, scaled.shape)
            scaled[i] = np.round(scaled[i])
        return scaled

    def _unscale_parameters(self, parameters):
        """Scale from parameters to a number between 0 and 1."""
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """Make sure the parameters lie between the limits."""
        mask = np.where((trial > 1) | (trial < 0))
        trial[mask] = self.random_number_generator.uniform(size=mask[0].shape)

    def _mutate(self, candidate):
        """Create a trial vector based on a mutation strategy."""
        trial = np.copy(self.population[candidate])

        rng = self.random_number_generator

        fill_point = rng.choice(self.parameter_count)

        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidate,
                                        self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        if self.strategy in self._binomial:
            crossovers = rng.uniform(size=self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.strategy in self._exponential:
            i = 0
            crossovers = rng.uniform(size=self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            while (i < self.parameter_count and crossovers[i]):
                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1

            return trial

    def _best1(self, samples):
        """best1bin, best1exp"""
        r0, r1 = samples[:2]
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        """rand1bin, rand1exp"""
        r0, r1, r2 = samples[:3]
        # print(self.population[r0], self.population[r1], self.population[r2])
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, samples):
        """randtobest1bin, randtobest1exp"""
        r0, r1, r2 = samples[:3]
        bprime = np.copy(self.population[r0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r1] -
                                self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        """currenttobest1bin, currenttobest1exp"""
        r0, r1 = samples[:2]
        bprime = (self.population[candidate] + self.scale *
                  (self.population[0] - self.population[candidate] +
                   self.population[r0] - self.population[r1]))
        return bprime

    def _best2(self, samples):
        """best2bin, best2exp"""
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.scale *
                  (self.population[r0] + self.population[r1] -
                   self.population[r2] - self.population[r3]))

        return bprime

    def _rand2(self, samples):
        """rand2bin, rand2exp"""
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.scale *
                  (self.population[r1] + self.population[r2] -
                   self.population[r3] - self.population[r4]))

        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement. You can't have the original candidate either.
        """
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs


class _ConstraintWrapper:
    """Object to wrap/evaluate user defined constraints.

    Very similar in practice to `PreparedConstraint`, except that no evaluation
    of jac/hess is performed (explicit or implicit).

    If created successfully, it will contain the attributes listed below.

    Parameters
    ----------
    constraint : {`NonlinearConstraint`, `LinearConstraint`, `Bounds`}
        Constraint to check and prepare.
    x0 : array_like
        Initial vector of independent variables, shape (N,)

    Attributes
    ----------
    fun : callable
        Function defining the constraint wrapped by one of the convenience
        classes.
    bounds : 2-tuple
        Contains lower and upper bounds for the constraints --- lb and ub.
        These are converted to ndarray and have a size equal to the number of
        the constraints.
    """

    def __init__(self, constraint, x0):
        self.constraint = constraint

        if isinstance(constraint, NonlinearConstraint):
            def fun(x):
                x = np.asarray(x)
                return np.atleast_1d(constraint.fun(x))
        elif isinstance(constraint, LinearConstraint):
            def fun(x):
                if issparse(constraint.A):
                    A = constraint.A
                else:
                    A = np.atleast_2d(constraint.A)
                return A.dot(x)
        elif isinstance(constraint, Bounds):
            def fun(x):
                return np.asarray(x)
        else:
            raise ValueError("`constraint` of an unknown type is passed.")

        self.fun = fun

        lb = np.asarray(constraint.lb, dtype=float)
        ub = np.asarray(constraint.ub, dtype=float)

        x0 = np.asarray(x0)

        # find out the number of constraints
        f0 = fun(x0)
        self.num_constr = m = f0.size
        self.parameter_count = x0.size

        if lb.ndim == 0:
            lb = np.resize(lb, m)
        if ub.ndim == 0:
            ub = np.resize(ub, m)

        self.bounds = (lb, ub)

    def __call__(self, x):
        return np.atleast_1d(self.fun(x))

    def violation(self, x):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        x : array-like
            Vector of independent variables, (N, S), where N is number of
            parameters and S is the number of solutions to be investigated.

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by, for each of the
            constraints specified by `_ConstraintWrapper.fun`.
            Has shape (M, S) where M is the number of constraint components.
        """
        # expect ev to have shape (num_constr, S) or (num_constr,)
        ev = self.fun(np.asarray(x))

        try:
            excess_lb = np.maximum(self.bounds[0] - ev.T, 0)
            excess_ub = np.maximum(ev.T - self.bounds[1], 0)
        except ValueError as e:
            raise RuntimeError("An array returned from a Constraint has"
                               " the wrong shape. If `vectorized is False`"
                               " the Constraint should return an array of"
                               " shape (M,). If `vectorized is True` then"
                               " the Constraint must return an array of"
                               " shape (M, S), where S is the number of"
                               " solution vectors and M is the number of"
                               " constraint components in a given"
                               " Constraint object.") from e

        v = (excess_lb + excess_ub).T
        return v