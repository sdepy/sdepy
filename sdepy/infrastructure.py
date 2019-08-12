"""
=========================================
INFRASTRUCTURE FOR THE STOCHASTIC PACKAGE
=========================================

*  ``process`` class,
*  stochasticity source classes,
*  ``montecarlo`` class.
"""

import numpy as np
from numpy import sqrt, exp
import scipy
import scipy.stats
import scipy.interpolate
import bisect
import inspect
import warnings


########################################
#  Private functions for recurring tasks
########################################

def _shape_setup(shape):
    """Array shape preprocessing, return (shape,) if shape is an integer."""
    return (shape,) if isinstance(shape, int) else shape


def _const_param_setup(z):
    """Preprocessing of quantitative parameters that cannot depend on time."""
    return z if (z is None) else np.asarray(z)


def _variable_param_setup(z):
    """Preprocessing of quantitative parameters that may be time-varying.

    If z is None, returns None.
    If z is array-like, returns its constant value as an array.
    If z is a process instance, returns z.
    If z is callable, returns a callable f with a ``shape`` attribute:
       - A test call is made to f(1.)
       - If succesful, f.shape is set to the shape of the test value,
         and f is wrapped with numpy.asarray if the test value is not
         an array
       - If fails, f.shape is set to None and f is returned as is
    """
    if z is None:
        return z
    elif isinstance(z, process):
        return z
    elif callable(z):
        # if time-dependent,
        # get a test value if possible
        # and find out if it is an array or not
        test_t = 1.
        try:
            x = z(test_t)
            isarray = isinstance(x, np.ndarray)
            shape = np.asarray(x).shape
        except Exception:
            # if evaluation fails, ignore
            # and let events unfold later
            isarray = False
            shape = None
        # return the callable result if it returns arrays,
        # otherwise pass it through np.asarray
        if isarray:
            def wrapped_callable_z(s):
                return z(s)
        else:
            def wrapped_callable_z(s):
                return np.asarray(z(s))
        # add the gathered shape info (possibly None)
        wrapped_callable_z.shape = shape
        return wrapped_callable_z
    else:
        # if not callable, convert to array and return
        return np.asarray(z)


def _get_param_shape(z):
    """Shape of z, or of z(t) in case z
    is a process or is callable.

    Expects z to have been initialized via
    z = _variable_param_setup(z)
    """
    if isinstance(z, process):
        return z.vshape + (z.paths,)
    elif z is None:
        return None
    else:
        # z is an array, or a callable
        # wrapped by _variable_param_setup
        return z.shape


def _const_rho_to_corr(rho):
    """Transform time-independent correlation values
    into a correlation matrix."""
    if rho is None:
        return None

    # _const_param_setup should already have been called on rho,
    # rho is an array
    n = rho.size
    if rho.shape not in {(), (n,), (n, 1)}:
        raise ValueError(
            "correlation ``rho`` should be a vector, "
            "possibly with a trailing 1-dimensional axis matching "
            "the paths axis, not an array with shape {}"
            .format(rho.shape))
    elif n == 1:
        rho = rho.reshape(())
        return np.array(((1, rho), (rho, 1)))
    else:
        rho = rho.reshape(n)
        I, R = np.eye(n), np.diag(rho)
        return np.concatenate((np.concatenate((I, R)),
                               np.concatenate((R, I))), axis=1)


def _get_corr_matrix(corr, rho):
    """Preprocessing of correlation matrix ``corr`` or
    correlation values ``rho``.

    Given either ``corr`` or ``rho`` (each may be an array,
    callable or process instance), returns the corresponding,
    possibly time-dependent correlation matrix,
    with a ``shape`` attribute set to
    its shape (may be set to None if attempts to
    retrieve shape information fail).

    If ``corr`` is not None, ``rho`` is ignored.
    If both are None, returns None.
    """
    # exit if no correlations specified
    if corr is None and rho is None:
        return None
    elif corr is not None:
        # if present, corr overrides rho
        corr = _variable_param_setup(corr)
        cshape = _get_param_shape(corr)
        if cshape is not None:
            if len(cshape) not in (2, 3) or cshape[0] != cshape[1] or \
               (len(cshape) == 3 and cshape[2] != 1):
                raise ValueError(
                    "the correlation matrix ``corr`` should be square, "
                    "possibly with a trailing 1-dimensional axis matching "
                    "the paths axis, not an array with shape {}"
                    .format(cshape))
    else:
        # corr is None: build correlation matrix from rho,
        # either statically or dynamically
        rho = _variable_param_setup(rho)
        rho_shape = _get_param_shape(rho)
        if rho_shape is not None:
            if len(rho_shape) > 2 or \
               (len(rho_shape) == 2 and rho_shape[1] != 1):
                raise ValueError(
                    "correlation ``rho`` should be a vector, "
                    "possibly with a trailing 1-dimensional axis matching "
                    "the paths axis, not an array with shape {}"
                    .format(rho.shape))
        if callable(rho):

            def corr(t):
                return _const_rho_to_corr(rho(t))

            corr.shape = None if rho_shape is None else \
                (2, 2) if rho_shape == () else \
                (2*rho_shape[0], 2*rho_shape[0])
        else:
            corr = _const_rho_to_corr(rho)

    return corr


def _check_source(src, paths, vshape):
    """Checks (non exaustive) on the validity of a stochasticity source."""
    # check compliance with the source protocol
    if callable(src) and hasattr(src, 'paths') and hasattr(src, 'vshape'):
        # check paths and vshape
        paths_ok = (src.paths == paths)
        try:
            vshape_ok = True
            np.broadcast_to(np.empty(src.vshape), vshape)
        except ValueError:
            vshape_ok = False
        if not paths_ok or not vshape_ok:
            raise ValueError(
                'invalid stochasticity source: '
                'expecting soruce paths={} and vshape broadcastable to {}, '
                'but paths={}, vshape={} were found'
                .format(paths, vshape, src.paths, src.vshape))
        return
    else:
        raise ValueError(
            "stochasticity source of type '{}', not compliant with the "
            'source protocol (should be callable with properly '
            'defined paths and vshape attributes)'
            .format(type(src).__name__))


def _source_setup(dz, source_type, paths, vshape, **args):
    """Preprocessing and setup of stochasticity sources."""
    if dz is None:
        return source_type(paths=paths, vshape=vshape, **args)
    elif inspect.isclass(dz):
        return dz(paths=paths, vshape=vshape, **args)
    else:
        _check_source(dz, paths, vshape)
        return dz


def _wraps(wrapped):
    """Decorator to preserve some basic attributes
    when wrapping a function or class"""
    def decorator(wrapper):
        for attr in ('__module__', '__name__', '__qualname__', '__doc__'):
            setattr(wrapper, attr, getattr(wrapped, attr))
        return wrapper
    return decorator


_empty = inspect.Signature.empty


def _signature(f):
    """
    List of (parameter name, parameter default)
    tuples for function f.
    """
    return [(k, p.default) for k, p in
            inspect.signature(f).parameters.items()]


#############################################
#  The process class
#############################################

class process(np.ndarray):
    """
    process(t=0., *, x=None, v=None, c=None, dtype=None)
    Array representation of a process (a subclass of numpy.ndarray).

    If ``p`` is a process instance, ``p[i, ..., k]`` is the value
    that the ``k``-th path of the represented process takes at time ``p.t[i]``.
    The first and last indexes of ``p`` are reserved for the timeline and
    paths respectively. A process should contain no less than 1 time point and
    1 path. Zero or more middle indexes refer to the values that the process
    takes at each given time and path.

    If ``p`` has ``N`` time points, ``paths`` is its number of paths and
    ``vshape`` is the shape of its values at any given time point and path,
    then ``p.shape`` is ``(N,) + vshape + (paths,)``. ``N, vshape, paths``
    are inferred at instantiation from the shape of ``t`` and
    ``x, v`` or ``c`` parameters.

    Parameters
    ----------
    t : array-like
        Timeline of the process, as a one dimensional array
        with shape ``(N,)``, in increasing order.
        Defaults to 0.
    x : array-like, optional
        Values of the process along the timeline and across paths.
        Should broadcast to ``(N,) + vshape + (paths,)``.
        The shapes of ``t`` and of the firs index of ``x`` must match.
        One and only one of ``x``, ``v``, ``c`` must be provided upon process
        creation, as a keyword argument.
    v : array-like, optional
        Values of a deterministic process along the timeline.
        Should broadcast to ``(N,) + vshape``.
        The shapes of ``t`` and of the firs index
        of ``v`` must match.
    c : array-like, optional
        Value of a constant, single-path process,  with shape ``vshape``.
        Each time point of the resulting process contains a copy of ``c``.
    dtype : data-type, optional
        Data-type of the values of the process. ``x``, ``v`` or ``c`` will
        be converted to ``dtype`` if need be.

    Notes
    -----
    A reference and not a copy of ``t, x, v, c`` is stored if possible.

    A process is a subclass of numpy.ndarray, where its values as an array
    are the process values along the timeline and across paths. All
    numpy.ndarray methods, attributes and properties are guaranteed to act
    upon such values, as would those of the parent class. Such no overriding
    commitment is intended to safeguard predictablity of array operations
    on process instances; process-specific functionalities are delegated
    to process-specific methods, attributes and properties.

    A process with a single time point is assumed to be constant.

    Processes have the ``__array_priority__`` attribute
    set to 1.0 by default. Ufuncs acting on a process,
    or on a process and an array, or on different processes sharing
    the same timeline, or on different processes one of which is constant,
    return a process with the timeline of the original
    process(es) passed as a reference. Ufuncs calls on different processes
    fail if non constant processes do not share the same timeline
    (interpolation should be handled explicitly), or in case broadcasting
    rules would result in mixing time, values and/or paths axes.

    Let p be a process instance. Standard numpy indexing acts on the
    process values and returns numpy.ndarray instances: in fact, ``p[i]``
    is equivalent to ``p.x[i]``, i.e. the same as ``p.view(numpy.ndarray)[i]``.
    Process-specific indexing is addressed via the following syntax,
    where ``i`` can be an integer, a multi-index or smart indexing reference
    consistent with the process shape:

    - ``p['t', i]`` : timeline indexing,
      roughly equivalent to ``process(t=p.t[i], x=p.x[i, ..., :])``

    - ``p['v', i]`` : values indexing,
      roughly equivalent to ``process(t=p.t, x=p.x[:, i, :])``

    - ``p['p', i]`` : paths indexing,
      roughly equivalent to ``process(t=p.t, x=p.x[:, ..., i])``

    Attributes
    ----------
    x
    paths
    vshape
    tx
    dt
    dtx
    t : array
        Stores the timeline of the process.
    interp_kind : str
        Stores the default interpolation kind, passed upon interpolation
        (``interp`` and ``__call__`` methods) to ``scipy.interpolate.interp1d``
        unless a specific kind is provided. Defaults to the class attribute
        of the same name, initialized to ``'linear'``.
        Note that ufuncs and methods, when returning new processes, do *not*
        preserve the ``interp_kind`` attribute, which falls back on the
        class default and should be set explicitly again if needed.

    Methods
    -------
    interp
    __call__
    __getitem__

    rebase
    shapeas

    pcopy
    xcopy
    tcopy

    pmin
    pmax
    psum
    pmean
    pvar
    pstd

    vmin
    vmax
    vsum
    vmean
    vvar
    vstd

    tmin
    tmax
    tsum
    tmean
    tvar
    tstd

    tdiff
    tder
    tint

    chf
    cdf

    """

    # ---------------------------------
    # Core class attributes and methods
    # ---------------------------------

    __array_priority__ = 1.0
    interp_kind = 'linear'  # default interpolation kind for the class

    def __new__(cls, t=0., *, x=None, v=None, c=None, dtype=None):

        t = np.asarray(t)
        if t.ndim > 1 or t.size == 0:
            raise ValueError('the shape of a process timeline should be '
                             '() or (n,), not {}'.format(t.shape))
        if t.ndim == 0:
            t = t.reshape(1)

        if sum(z is not None for z in (x, v, c)) != 1:
            raise ValueError('when creating a process instance, one and '
                             'only one of x or v or c should be provided')
        if x is not None:
            x = np.asarray(x, dtype=dtype)
        elif v is not None:
            x = np.asarray(v, dtype=dtype)[..., np.newaxis]
        elif c is not None:
            c = np.asarray(c, dtype=dtype)
            if t.size == 1:
                x = c[np.newaxis, ..., np.newaxis]
            else:
                x = np.empty(shape=t.shape + c.shape + (1,), dtype=dtype)
                x[...] = c[np.newaxis, ..., np.newaxis]
        else:
            assert False

        if t.shape != x.shape[:1]:
            raise ValueError('process could not be created from timeline t '
                             'shaped {} and body shaped {}'
                             .format(t.shape, x.shape))

        obj = x.view(cls)
        obj.t = t
        return obj

    def _is_compatible(self, other):
        """Check compatibility of two processes.

        Broadcasting is restricted to processes
        a and b such that _is_compatible(a, b) is True."""
        # self and other have the same timeline, or are both constant
        t_compatible = np.array_equal(self.t, other.t) or \
            (self.t.size == 1 or other.t.size == 1)
        # can broadcast process values and paths
        shape1, shape2 = self.shape[1:], other.shape[1:]
        vp_compatible = len(shape1) == len(shape2) and \
            all(n1 == n2 or n1 == 1 or n2 == 1 for
                n1, n2 in zip(shape1, shape2))
        return t_compatible and vp_compatible

    def __array_finalize__(self, obj):
        if obj is None:
            # this should never be triggered
            self.t = None
        elif isinstance(obj, process):
            # handle new from template
            if not hasattr(obj, 't') or obj.t.shape != self.shape[:1]:
                self.t = None
            else:
                self.t = obj.t
        else:
            # view casting - unsafe unless
            # self.t is taken care of afterwards
            self.t = None
            pass

    def __array_prepare__(self, out_array, context):
        ufunc, inputs, domain = context
        assert hasattr(self, 't')
        assert any(self is a for a in inputs)
        for a in inputs:
            if isinstance(a, process):
                if not hasattr(a, 't') or a.t is None:
                    raise ValueError(
                        'cannot operate on a process without a timeline. '
                        'if this results from array operations on processes, '
                        'try using their array views instead (x attribute)')
                if not a._is_compatible(self):
                    raise ValueError(
                        'processes could not be broadcast '
                        'together due to incompatible shapes {}, {} and/or '
                        'timelines'.format(a.shape, self.shape))
        return out_array

    def __array_wrap__(self, out_array, context=None):
        if context is None:
            # this may happen since numpy 1.16.0 when a process instance
            # invokes a numpy.ndarray method (eg. sum, mean, etc.):
            # in such case the resulting out_array is returned, as
            # needed to comply with the no overriding commitment
            # for numpy.ndarray methods
            return out_array
        else:
            ufunc, inputs, domain = context
            assert hasattr(self, 't')
            assert any(self is a for a in inputs)

            # get process inputs
            p_inputs = [a for a in inputs
                        if isinstance(a, process)]

            # ??? overcautious - to be eliminated
            for a in p_inputs:
                if not self._is_compatible(a):
                    assert False, 'this should never occur - '\
                           '__array_prepare__ should enforce compatibility'

            # set t to the common non constant timeline
            # or to the constant timeline of the first input
            t = p_inputs[0].t
            for a in p_inputs[1:]:
                if len(a.t) > 1:
                    t = a.t
                    break
            cls = type(self)
            return cls(t=t, x=out_array)

    # -------------
    # interpolation
    # -------------

    def interp(self, *, kind=None):
        """
        Interpolation in time of the process values.

        Returns a callable ``f``, as returned by
        ``scipy.interpolate.interp1d``, such that ``f(s)``
        approximates the value of the process at time point ``s``.
        ``f`` refers to the process timeline and values,
        without storing copies. ``s`` may be of any shape.

        Parameters
        ----------
        kind : string, optional
            An interpolation kind as accepted by
            ``scipy.interpolate.interp1d``. If None, defaults to
            the ``interp_kind`` attribute.

        Returns
        -------
        f : callable
            ``f``, as returned by scipy.interpolate.interp1d,
            such that ``f(s)`` approximates the value of the process
            at time point s. ``f`` refers to the process timeline and values,
            without storing copies.

            ``s`` may be of any shape: if ``p`` is a process instance,
            ``p.interp()(s).shape == s.shape + p.vshape + (p.paths,)``.

            In case ``p`` has a single time point, interpolation
            is not handled via ``scipy.interpolate.interp1d``;
            the process is assumed to be constant in time, and ``f``
            is a function object behaving accordingly.

        See Also
        --------
        process.__call__

        Notes
        -----
        The process is extrapolated as constant outside the timeline
        boundaries.

        If ``p`` is a process instance, ``p.interp(s)`` is an array,
        not a process.
        If an interpolated process is needed, it should be explicitly
        created using ``q = process(s, x=p(s))``, or its shorthand
        ``q = p.rebase(s)``.
        """

        t, x = self.t, self.view(np.ndarray)
        kind = self.interp_kind if kind is None else kind
        if t.size == 1:
            def f(s):
                s = np.asarray(s)
                return np.ones(s.shape + x.shape[1:], dtype=x.dtype) * \
                    x.reshape(tuple(1 for i in s.shape) +
                              x.shape[1:])
            return f
        else:
            g = scipy.interpolate.interp1d(
                t, x, axis=0,
                kind=kind,
                assume_sorted=True, copy=False,
                bounds_error=False,
                fill_value=(x[0], x[-1])
                )

            def f(s):
                return g(s).astype(x.dtype, copy=False)
            return f

    def __call__(self, s, ds=None, *, kind=None):
        """Interpolation in time of process values or increments.

        If ``p`` is a process instance and ``f = p.interp(kind)``:

        - ``p(s)`` returns ``f(s)``,
        - ``p(s, ds)`` returns ``f(s + ds) - f(s)``.

        See Also
        --------
        process.interp
        """
        s = np.asarray(s)
        f = self.interp(kind=kind)
        if ds is None:
            return f(s)
        else:
            ds = np.asarray(ds)
            return f(s+ds) - f(s)

    def rebase(self, t, *, kind=None):
        """Change the process timeline to t, using interpolation.

        A new process is returned with timeline ``t`` and values
        set to the calling process values, interpolated at
        ``t`` using ``process.interp`` with the given interpolation kind.

        If ``t`` is a scalar, a constant process is returned.
        """
        t = np.asarray(t)
        if t.ndim == 0:
            t = t.reshape(1)
        return process(t, x=self(t, kind=kind))

    # -------------------------
    # process-specific indexing
    # -------------------------

    def __getitem__(self, key):
        """See documentation of the process class."""
        x = self.view(np.ndarray)
        cls = type(self)
        colon = (slice(None),)

        # handle general indexing of self as a ndarray
        #
        if not isinstance(key, (tuple, str)):
            return x[key]  # standard indexing with integer or array key
        elif isinstance(key, str):
            key = (key,)  # special indexing with empty index (handled below)
        elif len(key) == 0:
            return x[()]  # standard indexing with empty key
        elif isinstance(key[0], str):
            pass  # special indexing with non-empty index (handled below)
        else:
            return x[key]  # ordinary indexing, key is a tuple

        # at this point key is a tuple, and key[0] is a str containing
        # a special indexing flag

        # key preprocessing
        #
        a, key = key[0], key[1:]
        if len(key) == 1 and isinstance(key[0], tuple):
            key = key[0]
            # if i = (i1, ..., ik) is a multi index,
            # p['v', i] is treated as p['v', i1, ..., ik]

        # handle process-specific indexing modes
        #
        if a not in ('v', 't', 'p'):
            raise IndexError('process indexing error - '
                             'unsupported indexing mode ' + repr(a))
        # 'v' modes - values indexing
        if a == 'v':
            return cls(self.t, x=x[colon + key + colon])
        # 't' and 'p' modes - timeline an paths indexing
        assert len(key) <= 1  # only one index is expected
        if len(key) == 1 and isinstance(key[0], int):
            # an integer index is treated as a slice of 1
            i = key[0]
            key = (slice(i, None, None),) if i == -1 else \
                  (slice(i, i+1, None),)
        if a == 't':
            return cls(self.t[key], x=x[key])
        elif a == 'p':
            return cls(self.t, x=x[(Ellipsis,) + key])
        else:
            assert False

    # ---------------------------------------------------------
    # convenience methods and properties (**NEVER** used above)
    # ---------------------------------------------------------

    # properties
    # ----------

    @property
    def x(self):
        """Process values, viewed as a numpy.ndarray."""
        return self.view(np.ndarray)

    @property
    def paths(self):
        """
        Number of paths of the process (coincides with the size
        of the last dimension of the process).
        """
        return self.shape[-1]

    @property
    def vshape(self):
        """Shape of the values of the process."""
        return self.shape[1:-1]

    @property
    def tx(self):
        """
        Timeline of the process, reshaped to be broadcastable to
        the process values and paths across time.
        """
        t = self.t
        return t.reshape(t.shape + tuple(1 for i in self.shape[1:]))

    @property
    def dt(self):
        """
        Process timeline increments, as returned by numpy.diff.

        Notes
        -----
        The result is computed upon first call and cached, and will not
        reflect subsequent modifications to the ``t`` attribute.
        """
        if not hasattr(self, '_dt'):
            self._dt = np.diff(self.t)
        return self._dt

    @property
    def dtx(self):
        """
        Process timeline increments, as returned by numpy.diff,
        reshaped to be broadcastable to the process values.

        Notes
        -----
        The result is computed upon first call and cached, and will not
        reflect subsequent modifications to the ``t`` attribute.
        """
        dt = self.dt
        return dt.reshape(dt.shape + (1,)*(len(self.shape) - 1))

    # reshaping
    # ----------
    def shapeas(self, vshape_or_process):
        """
        Reshape process values according to the given target shape.

        Returns a process pointing to the same data as the calling process,
        adding new 1-dimensional axes, or removing existing 1-dimensional axes
        to the left of the first dimension of process values, as needed to make
        the returned process broadcastable to a process with values of the
        given shape.

        To achieve broadcastability the unaffected dimensions, including the
        shape of the timeline and the number of paths, have to be compatible.

        Raises
        ------
        ValueError : if requested to remove a non 1-dimensional axis
        """
        vshape = (vshape_or_process.vshape
                  if isinstance(vshape_or_process, process)
                  else vshape_or_process)
        cls = type(self)
        k = self.ndim-2  # length of current vshape
        h = len(vshape)  # length of target vshape
        if h > k:
            newshape = self.shape[:1] + (1,)*(h-k) + self.shape[1:]
        else:
            if h < k and set(self.shape[1:k-h+1]) != {1}:
                raise ValueError('could not reshape {} process values as {}'
                                 .format(self.vshape, vshape))
            newshape = self.shape[:1] + self.shape[k-h+1:]
        return cls(self.t, x=self.view(np.ndarray).reshape(newshape))

    # copying
    # ----------

    def pcopy(self, **args):
        """
        Copy timeline and values of the process
        (``args`` are passed to ``numpy.ndarray.copy``).
        """
        cls = type(self)
        return cls(t=self.t.copy(**args),
                   x=self.view(np.ndarray).copy(**args))

    def xcopy(self, **args):
        """
        Copy values of the process, share timeline
        (``args`` are passed to ``numpy.ndarray.copy``).
        """
        cls = type(self)
        return cls(t=self.t,
                   x=self.view(np.ndarray).copy(**args))

    def tcopy(self, **args):
        """Copy timeline of the process, share values.
        (``args`` are passed to ``numpy.ndarray.copy``).
        """
        cls = type(self)
        return cls(t=self.t.copy(**args),
                   x=self.view(np.ndarray))

    # summary operations across paths
    # -------------------------------

    def pmin(self, out=None):
        """
        One path process exposing for each time point
        the minimum process value attained across paths.
        """
        return process(t=self.t,
                       x=self.min(axis=-1, out=out,
                                  keepdims=True))

    def pmax(self, out=None):
        """
        One path process exposing for each time point
        the maximum process value attained across paths.
        """
        return process(t=self.t,
                       x=self.max(axis=-1, out=out,
                                  keepdims=True))

    def psum(self, dtype=None, out=None):
        """
        One path process exposing for each time point
        the sum of process values across paths.
        """
        dtype = self.dtype if dtype is None else dtype
        return process(t=self.t,
                       x=self.sum(axis=-1, dtype=dtype, out=out,
                                  keepdims=True))

    def pmean(self, dtype=None, out=None):
        """
        One path process exposing for each time point
        the mean of process values across paths.
        """
        dtype = self.dtype if dtype is None else dtype
        return process(t=self.t,
                       x=self.mean(axis=-1, dtype=dtype, out=out,
                                   keepdims=True))

    def pvar(self, dtype=None, out=None, ddof=0):
        """
        One path process exposing for each time point
        the variance of process values across paths.
        """
        dtype = self.dtype if dtype is None else dtype
        return process(t=self.t,
                       x=self.var(axis=-1, dtype=dtype, out=out,
                                  ddof=ddof, keepdims=True))

    def pstd(self, dtype=None, out=None, ddof=0):
        """
        One path process exposing for each time point
        the standard deviation of process values across paths.
        """
        dtype = self.dtype if dtype is None else dtype
        return process(t=self.t,
                       x=self.std(axis=-1, dtype=dtype, out=out,
                                  ddof=ddof, keepdims=True))

    # summary operations across values
    # --------------------------------

    def vmin(self, out=None):
        """
        Process exposing for each time point and path
        the minimum of process values.
        """
        return process(
            t=self.t,
            x=self.min(axis=tuple(range(1, len(self.vshape) + 1)),
                       out=out),
        )

    def vmax(self, out=None):
        """
        Process exposing for each time point and path
        the maximum of process values.
        """
        return process(
            t=self.t,
            x=self.max(axis=tuple(range(1, len(self.vshape) + 1)),
                       out=out),
        )

    def vsum(self, dtype=None, out=None):
        """
        Process exposing for each time point and path
        the sum of process values.
        """
        return process(
            t=self.t,
            x=self.sum(axis=tuple(range(1, len(self.vshape) + 1)),
                       dtype=dtype, out=out),
        )

    def vmean(self, dtype=None, out=None):
        """
        Process exposing for each time point and path
        the mean of process values.
        """
        return process(
            t=self.t,
            x=self.mean(axis=tuple(range(1, len(self.vshape) + 1)),
                        dtype=dtype, out=out),
        )

    def vvar(self, dtype=None, out=None, ddof=0):
        """
        Process exposing for each time point and path
        the variance of process values.
        """
        return process(
            t=self.t,
            x=self.var(axis=tuple(range(1, len(self.vshape) + 1)),
                       dtype=dtype, out=out, ddof=ddof),
        )

    def vstd(self, dtype=None, out=None, ddof=0):
        """
        Process exposing for each time point and path
        the standard deviation of process values.
        """
        return process(
            t=self.t,
            x=self.std(axis=tuple(range(1, len(self.vshape) + 1)),
                       dtype=dtype, out=out, ddof=ddof),
        )

    # summary operations along the timeline
    # -------------------------------------

    def tmin(self, out=None):
        """
        Constant process exposing for each path the minimum
        process value attained along time.
        """
        return process(t=self.t[:1],
                       x=self.min(axis=0, out=out,
                                  keepdims=True))

    def tmax(self, out=None):
        """Constant process exposing for each path the maximum
        process value attained along time.
        """
        return process(t=self.t[:1],
                       x=self.max(axis=0, out=out,
                                  keepdims=True))

    def tsum(self, dtype=None, out=None):
        """
        Constant process exposing for each path the sum of
        process values along time.
        """
        dtype = self.dtype if dtype is None else dtype
        return process(t=self.t[:1],
                       x=self.sum(axis=0, dtype=dtype, out=out,
                                  keepdims=True))

    def tcumsum(self, dtype=None, out=None):
        """
        Process exposing for each path and time point
        the cumulative sum of process values along time.
        """
        dtype = self.dtype if dtype is None else dtype
        return process(t=self.t,
                       x=self.cumsum(axis=0, dtype=dtype, out=out))

    def tmean(self, dtype=None, out=None):
        """
        Constant process exposing for each path the mean of
        process values along time.
        """
        dtype = self.dtype if dtype is None else dtype
        return process(t=self.t[:1],
                       x=self.mean(axis=0, dtype=dtype, out=out,
                                   keepdims=True))

    def tvar(self, dtype=None, out=None, ddof=0):
        """
        Constant process exposing for each path the variance of
        process values along time.
        """
        dtype = self.dtype if dtype is None else dtype
        return process(t=self.t[:1],
                       x=self.var(axis=0, dtype=dtype, out=out,
                                  ddof=ddof, keepdims=True))

    def tstd(self, dtype=None, out=None, ddof=0):
        """
        Constant process exposing for each path the standard
        deviation of process values along time.
        """
        dtype = self.dtype if dtype is None else dtype
        return process(t=self.t[:1],
                       x=self.std(axis=0, dtype=dtype, out=out,
                                  ddof=ddof, keepdims=True))

    # time increments, differences and sums along the timeline
    # --------------------------------------------------------

    def tdiff(self, dt_exp=0, fwd=True):
        """
        Process increments along the timeline, optionally
        weighted by time increments.

        Parameters
        ----------
        dt_exp : int or float, optional
            Exponent applied to time increment weights.
            If 0, returns process increments.
            If 1, approximates a time derivative.
            If 0.5, approximates realized volatility.
        fwd : bool, optional
            If True, the differences are forward-looking

        Returns
        -------
        q : process
            If ``p`` is a process shaped ``(N,) + p.vshape + (p.paths,)``,
            with timeline ``t``, ``p.tdiff(dt_exp, fwd)`` returns
            a process ``q``, shaped ``(N-1,) + p.vshape + (p.paths,)``
            with values

                ``q[i] = (p[i+1] - p[i])/(t[i+1] - t[i])**dt_exp``

            If ``fwd`` evaluates to ``True``, ``q[i]`` is assigned
            to time point ``t[i]`` (``q`` stores at ``t[i]``
            the increments of ``p`` looking forwards)
            or to ``t[i+1]`` otherwise (increments looking backwards).

        See also
        --------
        tder
        tint

        Notes
        -----
        if ``p`` is a process instance realizing a solution of the SDE
        ``dp(t) = sigma(t)*dw(t)`` across several paths, then

            ``p.tdiff(dt_exp=0.5).pstd()``

        is a 1-path process that estimates ``sigma(t)``.
        """
        t = self.t[:-1] if fwd else self.t[1:]
        x = np.diff(self, axis=0)
        if dt_exp:
            np.divide(x, self.dtx**dt_exp, out=x)
        return process(t=t, x=x)

    def tder(self):
        """
        Forward looking derivative of the given process,
        linearly interpolated between time points.

        Shorthand for ``p.tdiff(dt_exp=1)``.

        See Also
        --------
        tdiff
        tint

        Notes
        -----
        ``p.tder().tint()`` equals, within rounding errors,
        ``p['t', :-1] - p['t', 0]``
        """
        return self.tdiff(fwd=True, dt_exp=1)

    def tint(self):
        """
        Integral of the given process, linearly interpolated
        between time points.

        See Also
        --------
        tdiff
        tder

        Notes
        -----
        ``p.tin().tder()`` equals, within rounding errors,
        ``p['t', :-1]``
        """
        x = np.empty(self.shape, dtype=self.dtype)
        x[0] = 0
        x[1:] = self[:-1]*self.dtx
        return process(t=self.t,
                       x=x.cumsum(axis=0, out=x))

    # characteristic function estimator
    # ---------------------------------

    def chf(self, t=None, u=None):
        """
        Characteristic function of the probability distribution
        of process values.

        ``p.chf(t, u)`` estimates the characteristic function
        of interpolated process values ``p(t)`` at time(s) ``t``.
        ``p.chf(u)`` is a shorthand for ``p.chf(p.t, u)`` (no interpolation).

        Parameters
        ----------
        t : array-like, optional
            Time points at which to compute the characteristic function.
            If omitted or ``None``, the entire process timeline is used.
        u : array-like, mandatory
            Values at which to evaluate the characteristic function.

        Returns
        -------
        array
            Returns an array, with shape ``t.shape + u.shape + vshape``,
            where ``vshape`` is the shape of values of the calling
            process ``p``, containing the average across paths of
            ``exp(1j*u*p(t))``.
        """

        if t is None and u is None:
            raise TypeError('u argument missing')
        elif u is None:
            t, u = None, t

        if t is None:
            # use as t the process timeline
            t, u = self.t, np.asarray(u)
            x = self.x
        else:
            # interpolate the process to the given t
            t, u = np.asarray(t), np.asarray(u)
            x = self(t)

        uu = u.reshape((1,)*t.ndim + u.shape + (1,)*(x.ndim - t.ndim))
        xx = x.reshape(t.shape + (1,)*u.ndim + x.shape[t.ndim:])
        return exp(1j*uu*xx).mean(axis=-1)

    def cdf(self, t=None, x=None):
        """
        Cumulative probability distribution function of process values.

        ``p.cdf(t, x)`` estimates the cumulative probability distribution
        function of interpolated process values ``p(t)`` at time(s) ``t``.
        ``p.cdf(x)`` is a shorthand for ``p.cdf(p.t, x)`` (no interpolation).

        Parameters
        ----------
        t : array-like, optional
            Time points along the process timeline. If omitted or ``None``, the
            entire process timeline is used.
        x : array-like, mandatory
            Values at which to evaluate the cumulative probability
            distribution function.

        Returns
        -------
        array
            Returns an array, with shape ``t.shape + x.shape + vshape``,
            where ``vshape`` is the shape of the values of the calling
            process ``p``, containing the average across paths of
            ``1 if p(t) <= x else 0``.
        """

        if t is None and x is None:
            raise TypeError('x argument missing')
        elif x is None:
            t, x = None, t

        if t is None:
            t, x = self.t, np.asarray(x)
            y = self.x
        else:
            t, x = np.asarray(t), np.asarray(x)
            y = self(t)

        xx = x.reshape((1,)*t.ndim + x.shape + (1,)*(y.ndim - t.ndim))
        yy = y.reshape(t.shape + (1,)*x.ndim + y.shape[t.ndim:])
        return (yy <= xx).sum(axis=-1)/self.paths


# ----------------------------------------------
# A constructor for piecewise constant processes
# ----------------------------------------------

def piecewise(t=0., *, x=None, v=None, dtype=None, mode='mid'):
    """
    Return a process that interpolates to a piecewise constant function.

    Parameters
    ----------
    t : array-like
        Reference timeline (see below).
    x : array-like, optional
        Values of the process along the timeline and across paths.
        One and only one of ``x``, ``v``, must be provided,
        as a keyword argument.
    v : array-like, optional
        Values of a deterministic (one path) process along the timeline.
    dtype : data-type, optional
        Data-type of the values of the process.
    mode : string, optional
        Specifies how the piecewise constant segments relate
        to the reference timeline: 'mid', 'forward', 'backward'
        set ``t[i]`` to be the midpoint, start or end point respectively,
        of the constant segment with value ``x[i]`` or ``v[i]``.

    See Also
    --------
    process

    Notes
    -----
    Parameters ``t``, ``x``, ``v``, ``dtype`` conform to the ``process``
    instantiation interface and shape requirements.

    The returned process ``p`` behaves as advertised upon interpolation
    with default interpolation kind (set to ``'nearest'``
    via the ``interp_kind`` attribute), and may be used
    as a time dependent piecewise constant parameter in SDE integration.
    However, its timeline ``p.t`` and values ``p.x``
    are not guaranteed to coincide with the given ``t`` or ``x``,
    and should not be relied upon.
    """
    # delegate preprocessing of arguments to the process class
    p = process(t, x=x, v=v, dtype=dtype)
    t, x = p.t, p.x

    if mode == 'mid':
        s, y = t, x
    else:
        s = np.full(t.size*2, np.nan, dtype=t.dtype)
        y = np.full((x.shape[0]*2,) + x.shape[1:], np.nan, dtype=x.dtype)
        s[::2] = t
        s[1::2] = t
        if mode == 'forward':
            y[1::2] = x
            y[2:-1:2] = x[:-1]
            y[0] = x[0]
        elif mode == 'backward':
            y[::2] = x
            y[1:-1:2] = x[1:]
            y[-1] = x[-1]
        else:
            raise ValueError(
                "mode should be one of 'mid', 'forward', 'backward', "
                'but {} was given'.format(mode))

    p = process(s, x=y, dtype=dtype)
    p.interp_kind = 'nearest'
    return p


# safeguard backward compatibility
def _piecewise_constant_process(*args, **kwds):
    """private alias of sdepy.piecewise
    (unused, deprecated)"""

    warnings.warn('call to sdepy.infrastructure._piecewise_constant_process, '
                  'use sdepy.piecewise instead',
                  DeprecationWarning)
    return piecewise(*args, **kwds)


#######################################
#  Stochasticity sources without memory
#######################################

# base class for sources
# ----------------------

class source:
    """
    Base class for stochasticity sources.

    Parameters
    ----------
    paths : int
        Number of paths (last dimension) of the source realizations.
    vshape : tuple of int
        Shape of source values.
    dtype : data-type
        Data type of source values. Defaults to ``None``.

    Returns
    -------
    array
       Once instantiated as ``dz``, ``dz(t, dt)`` returns a random realization
       of the stochasticity source increments from time ``t``
       to time  ``t + dt``, with shape ``(t + dt).shape + vshape + (paths,)``.
       For sources with memory (``true_source`` class and subclasses),
       ``dz(t)`` returns the realized value at time ``t`` of the source
       process, according to initial conditions set at instantiation.
       The definition of source specific parameters, and computation of
       actual source realizations, are delegated to subclasses.
       Defaults to an array of ``numpy.nan``.

    Notes
    -----
    Any callable object ``dz(t, dt)``, with attributes ``paths`` and
    ``vshape``, returning arrays broadcastable to shape
    ``t_shape + vshape + (paths,)``, where ``t_shape`` is the shape
    of ``t`` and/or ``dt``, complies with the ``source`` protocol.
    Such object may be passed to any of the process realization classes,
    to be used as a stochasticity source in integrating or computing
    the relevant SDE solution. ``process`` instances, in particular,
    may be used as stochasticity sources.

    When calling ``dz(t, dt)``, ``t`` and/or ``dt`` can take any shape.

    Attributes
    ----------
    size
    t
    """

    def __init__(self, *, paths=1, vshape=(), dtype=None):
        self.paths, self.vshape, self.dtype = paths, vshape, dtype
        self.vshape = _shape_setup(self.vshape)

    def __call__(self, t, dt=None):
        """Realization of stochasticity source values or increments."""
        dt = 0 if dt is None else dt
        t, dt = np.asarray(t), np.asarray(dt)
        return t + dt + np.nan

    @property
    def size(self):
        """
        Returns the number of stored scalar values from previous
        evaluations, or 0 for sources without memory.
        """
        return 0

    @property
    def t(self):
        """
        Returns a copy of the time points at which source values
        have been stored from previous evaluations, as an array,
        or an empty array for sources without memory.
        """
        return np.array((), dtype=float)


# Wiener process stochasticity source
# -----------------------------------

class wiener_source(source):
    """
    dw, a source of standard Wiener process (Brownian motion) increments.

    Parameters
    ----------
    paths : int
        Number of paths (last dimension) of the source realizations.
    vshape : tuple of int
        Shape of source values.
    dtype : data-type
        Data type of source values. Defaults to ``None``.
    corr : array-like, or callable, or None
        Correlation matrix of the standard Wiener process increments,
        possibly time-dependent, or ``None`` for no correlations,
        or for correlations specified by the ``rho`` parameter.
        If not ``None``, overrides ``rho``.
        If ``corr`` is a square matrix of shape ``(M, M)``,
        or callable with ``corr(t)`` evaluating to such matrix,
        the last dimension of the source values must be of size ``M``
        (``vshape[-1] == M``), and increments along
        the last axis of the source values will be correlated accordingly.
    rho : array-like, or callable, or None
        Correlations of the standard Wiener process increments,
        possibly time-dependent, or ``None`` for no correlations.
        If ``rho`` is scalar, or callable with ``rho(t)`` evaluating
        to a scalar, ``M=2`` is assumed, and ``corr=((1, rho), (rho, 1))``.
        If ``rho`` is a vector of shape ``(K,)``, or a callable
        with ``rho(t)`` evaluating to such vector, ``M=2*K`` is assumed,
        and the ``M`` source values along the last ``vshape`` dimension
        are correlated so that ``rho[i]`` correlates the ``i``-th and
        ``K+i``-th values, other correlations being zero
        (``corr = array((I, R), (R, I))`` where ``I = numpy.eye(K)`` and
        ``R = numpy.diag(rho)``).

    Returns
    -------
    array
       Once instantiated as ``dw``, ``dw(t, dt)`` returns a random realization
       of standard Wiener process increments from time ``t``
       to time  ``t + dt``, with shape ``(t + dt).shape + vshape + (paths,)``.
       The increments are normal variates with mean 0, either independent
       with standard deviation ``sqrt(dt)``, or correlated with
       covariance matrix ``corr*dt``, or ``corr(t + dt/2)*dt``
       (the latter approximates the integral of ``corr(t)`` from ``t``
       to ``t + dt``).

    Attributes
    ----------
    corr : array, or callable
        Stores the correlation matrix used computing increments. May expose
        either a reference to ``corr``, if provided explicitly, or an
        appropriate object, in case ``rho`` was specified.

    See Also
    --------
    source

    Notes
    -----
    Realizations across different ``t`` and/or ``dt`` array elements,
    and/or across different paths, and/or along axes of the source values
    other than the last axis of ``vshape``, are independent.
    ``corr`` should be a correlation matrix with unit diagonal elements
    and off-diagonal correlation coefficients, not a covariance matrix.

    ``corr`` and ``rho`` values with a trailing one-dimensional paths axis
    are accepted, of shape ``(M, M, 1)`` or ``(M/2, 1)`` respectively.
    This last axis is ignored: this allows for deterministic ``process``
    instances (single path processes) to be passed as valid ``corr`` or
    ``rho`` values. Path dependent ``corr`` and ``rho`` are not supported.

    For time-dependent correlations, ``dw(t, dt)`` approximates the increments
    of a process ``w(t)`` obeying the SDE ``dw(t) = D(t)*dz(t)``,
    where ``z(t)`` are standard uncorrelated Wiener processes, and ``D(t)``
    is a time-dependent matrix such that ``D(t) @ (D(t).T) == corr(t)``.
    Note that, given any two time points ``s`` and ``t > s``,
    by the Ito isometry the expectation value of
    ``(w(t)-w(s))[i] * (w(t)-w(s))[j]``, i.e. the ``i``, ``j`` element of the
    covariance matrix of increments of ``w`` from ``s`` to ``t``,
    equals the integral of ``corr(u)[i, j]`` in ``du`` from ``s`` to ``t``.

    For time-independent correlations, as well as for correlations that
    depend linearly on ``t``, the resulting ``dw(t, dt)`` is exact, as
    far as it can be within the accuracy of the pseudo-random
    normal variate generator of NumPy. Otherwise, mind using small enough
    ``dt`` intervals.
    """

    def __init__(self, *, paths=1, vshape=(), dtype=None,
                 corr=None, rho=None):
        super().__init__(paths=paths, vshape=vshape, dtype=dtype)

        # get the correlation matrix from 'corr' and 'rho'
        self.corr = corr = _get_corr_matrix(corr, rho)
        cshape, vshape = _get_param_shape(corr), self.vshape
        if corr is not None:
            # if a correlation matrix was given, check shapes if possible
            if self.vshape == ():
                raise ValueError(
                    'if vshape is (), no correlations apply, but '
                    'corr={}, rho={} were given'
                    .format(corr, rho))
            elif cshape is not None:
                if cshape[:2] != 2*vshape[-1:] or \
                   (len(cshape) == 3 and cshape[-1] != 1):
                    raise ValueError(
                        'cannot instantiate a Wiener source with '
                        'values shape {} and correlation matrix shape {}'
                        .format(vshape, cshape))

    def __call__(self, t, dt):
        """See wiener_source class documentation."""
        paths, vshape, dtype = self.paths, self.vshape, self.dtype
        corr = self.corr
        t, dt = np.broadcast_arrays(t, dt)
        tshape = dt.shape
        # using np.random.normal and np.random.multivariate_normal,
        # instead of scipy.stats.norm and scipy.stats.multivariate_normal
        # to imporve speed (aviod overhead of scipy.stats random variable
        # instantiation at each call)
        if corr is None:
            # --- handle uncorrelated samples (vshape may be any shape)
            dz = np.random.normal(
                0., 1., size=tshape + vshape + (paths,)
               ).astype(dtype, copy=False)
        else:
            # --- handle correlated samples
            M = vshape[-1]
            mean = np.zeros(M)
            if not callable(corr):
                # --- constant correlations
                # generate dz, shaped as ``tshape + vshape[:-1] + (paths, M)``
                cov = corr
                if cov.ndim == 3:
                    if cov.shape[2] != 1:
                        raise ValueError(
                            'invalid correlation matrix shape {}'
                            .format(cov.shape))
                    cov = cov[..., 0]  # remove paths axis if present
                dz = np.random.multivariate_normal(
                    mean=mean, cov=cov, size=tshape + vshape[:-1] + (paths,)
                    ).astype(dtype, copy=False)
            else:
                # --- time-dependent correlations
                dz = np.empty(tshape + vshape[:-1] + (paths, M), dtype=dtype)
                for i in np.ndindex(tshape):
                    # generate dz[i], shaped as ``vshape[:-1] + (paths, M)``
                    cov = corr(t[i] + dt[i]/2)
                    # this approximates (integral of corr(t) from t to t+dt)/dt
                    if cov.ndim == 3:
                        if cov.shape[2] != 1:
                            raise ValueError(
                                'invalid correlation matrix shape {}'
                                .format(cov.shape))
                        cov = cov[..., 0]  # remove paths axis if present
                    dz[i] = np.random.multivariate_normal(
                        mean=mean, cov=cov, size=vshape[:-1] + (paths,)
                        )
            # reshape dz to ``tshape + vshape + (paths,)``
            dz = dz.swapaxes(-1, -2)
            if dz.shape != tshape + vshape + (paths,):
                raise RuntimeError(
                    'unexpected error - inconsistent shapes')

        # apply sqrt(dt) normalization factor
        dt = dt.reshape(tshape + (1,)*len(self.vshape) + (1,))
        dz *= sqrt(np.abs(dt))
        return dz


# Poisson process stochasticity source
# ------------------------------------

class poisson_source(source):
    """dn, a source of Poisson process increments.

    Parameters
    ----------
    paths : int
        Number of paths (last dimension) of the source realizations.
    vshape : tuple of int
        Shape of source values.
    dtype : data-type
        Data type of source values. Defaults to ``int``.
    lam : array-like, or callable
        Intensity of the Poisson process, possibly time-dependent.
        Should be an array of non-negative values, broadcastable to shape
        ``vshape + (paths,)``, or a callable with ``lam(t)`` evaluating
        to such array.

    Returns
    -------
    array
       Once instantiated as ``dn``, ``dn(t, dt)`` returns a random
       realization of Poisson process increments from time ``t`` to time
       ``t + dt``, with shape ``(t + dt).shape + vshape + (paths,)``.
       The increments are independent Poisson variates with mean
       ``lam*dt``, or ``lam(t + dt/2)*dt`` (the latter approximates
       the integral of ``lam(t)`` from ``t`` to ``t + dt``).

    See Also
    --------
    source
    """

    def __init__(self, *, paths=1, vshape=(), dtype=int, lam=1.):
        super().__init__(paths=paths, vshape=vshape, dtype=dtype)
        self.lam = lam = _variable_param_setup(lam)
        lam_shape = _get_param_shape(lam)
        if lam_shape is not None:
            try:
                np.broadcast_to(np.empty(lam_shape),
                                self.vshape + (paths,))
            except ValueError:
                raise ValueError(
                    'cannot broadcast lambda parameter shaped {} to'
                    'requested poisson source shape = vshape + (paths,) = {}'
                    .format(self.lam.shape, self.vshape + (paths,))
                    )

    def __call__(self, t, dt):
        """See poisson_source class documentation."""
        t, dt = np.broadcast_arrays(t, dt)
        abs_dt, sign_dt = np.abs(dt), np.sign(dt).astype(int)
        tshape = dt.shape
        paths, vshape, dtype = self.paths, self.vshape, self.dtype
        lam = self.lam
        dn = np.empty(tshape + vshape + (paths,),
                      dtype=dtype)
        # using np.random.poisson instead of scipy.stats.poisson
        # to imporve speed (aviod overhead of scipy.stats random variable
        # instantiation at each call)
        for i in np.ndindex(tshape):
            L = (lam(t[i] + dt[i]/2) if callable(lam) else lam)
            dn[i] = sign_dt[i]*np.random.poisson(abs_dt[i]*L,
                                                 vshape + (paths,))
        return dn


# Probability distributions with variable parameters
# to be used in compound Poisson stochasticity sources
# ----------------------------------------------------

def _make_rv_params_variable(rv, **params):
    """
    Wraps the random variable rv, allowing it
    to accept time-dependent parameters.
    """
    if any(callable(x) for x in params.values()):
        return lambda t: rv(
            **{k: (x(t) if callable(x) else x)
               for k, x in params.items()})
    else:
        return rv(**params)


def norm_rv(a=0, b=1):
    """
    Normal distribution with mean a and standard deviation b, possibly
    time-dependent.

    Wraps ``scipy.stats.norm(loc=a, scale=b)``.

    See Also
    --------
    cpoisson_source
    """
    return _make_rv_params_variable(_norm_rv, a=a, b=b)


def uniform_rv(a=0, b=1):
    """
    Uniform distribution between a and b, possibly time-dependent.

    Wraps ``scipy.stats.uniform(loc=a, scale=b-a)``.

    See Also
    --------
    cpoisson_source
    """
    return _make_rv_params_variable(_uniform_rv, a=a, b=b)


def exp_rv(a=1):
    """
    Exponential distribution with scale a, possibly time-dependent.

    Wraps ``scipy.stats.expon(scale=a)``.
    The probability distribution function is:
      -  if ``a > 0``, ``pdf(x) =  a*exp(-a*x)``, with support in ``[0, inf)``
      -  if ``a < 0``, ``pdf(x) = -a*exp( a*x)``, with support in ``(-inf, 0]``

    See Also
    --------
    cpoisson_source
    """
    return _make_rv_params_variable(_exp_rv, a=a)


def double_exp_rv(a=1, b=1, pa=0.5):
    """
    Double exponential distribution, with scale a with
    probability pa, and -b with probability (1 - pa), possibly
    time-dependent.

    Double exponential distribution, with probability distribution
      -  for ``x`` in ``[0, inf)``, ``pdf(x) = pa*exp(-a*x)*a``
      -  for ``x`` in ``(-inf, 0)``, ``pdf(x) = (1-pa)*exp(b*x)*b``
    where ``a`` and ``b`` are positive and ``pa`` is in ``[0, 1]``.

    See Also
    --------
    cpoisson_source
    """
    return _make_rv_params_variable(_double_exp_rv,
                                    a=a, b=b, pa=pa)


def _norm_rv(a=0, b=1):
    """See norm_rv."""
    a, b = np.broadcast_arrays(a, b)
    rv = scipy.stats.norm(loc=a, scale=b)
    rv.exp_mean = lambda: exp(a + b*b/2) + 0
    return rv


def _uniform_rv(a=0, b=1):
    """See uniform_rv"""
    a, b = np.broadcast_arrays(a, b)
    rv = scipy.stats.uniform(loc=a, scale=b-a)
    rv.exp_mean = lambda: (exp(b) - exp(a))/(b - a) + 0
    return rv


class _exp_rv:
    """See exp_rv."""
    def __init__(self, a=1):
        a = self._a = np.asarray(a)
        if (a == 0).any():
            raise ValueError('domain error in arguments')
        self._s = np.sign(a)
        self._rv = scipy.stats.expon(scale=np.abs(a))

    def rvs(self, size):
        return self._s*self._rv.rvs(size)

    def mean(self):
        return self._s*self._rv.mean()

    def var(self):
        return self._rv.var()

    def std(self):
        return sqrt(self._rv.var())

    def exp_mean(self):
        a = self._a
        return np.where(a < 1, 1/(1 - a), np.inf) + 0


class _double_exp_rv:
    """see double_exp_rv."""

    def __init__(self, a=1, b=1, pa=0.5):
        a, b, pa, pb = \
            self._a, self._b, self._pa, self._pb = \
            np.broadcast_arrays(a, b, pa, 1-np.asarray(pa))
        if (a <= 0).any() or (b <= 0).any() \
           or (pa > 1).any() or (pa < 0).any():
            raise ValueError('domain error in arguments')
        self._rvxa = scipy.stats.expon(scale=a)
        self._rvxb = scipy.stats.expon(scale=b)
        self._rvu = scipy.stats.uniform(scale=1.)

    def rvs(self, size):
        pa = self._pa
        rvs_plus = self._rvxa.rvs(size)
        rvs_minus = self._rvxb.rvs(size)
        uniform = self._rvu.rvs(size)
        return np.where(uniform <= pa, rvs_plus, -rvs_minus) + 0

    def mean(self):
        a, b, pa, pb = self._a, self._b, self._pa, self._pb
        return (pa*a - pb*b) + 0

    def var(self):
        a, b, pa, pb = self._a, self._b, self._pa, self._pb
        return pa*pb*(a+b)**2 + (pa*a**2 + pb*b**2) + 0

    def std(self):
        return sqrt(self.var())

    def exp_mean(self):
        a, b, pa, pb = self._a, self._b, self._pa, self._pb
        return (pa/(1 - a) if a < 1 else np.inf) + pb/(1 + b) + 0


# Convenience methods for handling distributions
# to be passed to compound_poisson
# ----------------------------------------------


def rvmap(f, y):
    """
    Map f to random variates of distribution y, possibly time-dependent.

    Parameters
    ----------
    f : callable
        Callable with signature ``f(y)``, or
        ``f(t, y)`` or ``f(s, y)``, to be mapped to the
        random variates of ``y`` or ``y(t)``
    y : distribution, or callable
        Distribution, possibly time-dependent, as accepted by
        ``cpoisson_source``.

    Returns
    -------
    new_y : Distribution, or callable
        An object with and ``rvs(shape)`` method, or a callable
        with ``new_y(t)`` evaluating to such object, as accepted
        by ``cpoisson_source``.
        ``new_y.rvs(shape)``, or ``new_y(t).rvs(shape)``, returns
        ``f(y.rvs(shape))``, or ``f([t, ] y(t).rvs(shape)``.

    See Also
    --------
    cpoisson_source
    norm_rv
    uniform_rv
    exp_rv
    double_exp_rv

    Notes
    -----
    ``new_y`` does not provide any ``mean, std, var, exp_mean`` method.

    To be recognized as time-dependent, ``f`` should have its first
    parameter named ``t`` or ``s``.

    """

    time_dependent_f = _signature(f)[0][0] in ('t', 's')

    if callable(y) or time_dependent_f:
        def new_y(t):
            yt = y(t) if callable(y) else y

            class new_yt_class:
                def rvs(self, size):
                    return (f(t, yt.rvs(size)) if time_dependent_f
                            else f(yt.rvs(size)))

            new_yt = new_yt_class()
            return new_yt
        return new_y
    else:

        class new_y_class:
            def rvs(self, size):
                return f(y.rvs(size))

        new_y = new_y_class()
        return new_y


def _exp_mean(rv, eps=0.00001):
    """
    Average of the exponential of the random variable rv.

    Returns an approximate value of the average of the exponential
    of the ``scipy.stats`` random variable ``rv``, as the expectation
    value of ``exp(x)`` between ``rv.ppf(eps)`` and ``rv.ppf(1-eps)``,
    computed via ``rv.expect``.
    """
    lb, ub = rv.ppf(eps), rv.ppf(1 - eps)
    exp_mean = rv.expect(lambda x: exp(x), lb=lb, ub=ub)
    return exp_mean


# Compound Poisson stochasticity source
# -------------------------------------

class cpoisson_source(source):
    """
    dj, a source of compound Poisson process increments (jumps).

    Parameters
    ----------
    paths : int
        Number of paths (last dimension) of the source realizations.
    vshape : tuple of int
        Shape of source values.
    dtype : data-type
        Data type of source values. Defaults to ``None``.
    dn : source or source class, optional
        If given, ``dn`` is used as the underlying source of Poisson process
        increments, overriding the ``ptype`` and ``lam`` parameters.
    ptype : data-type
        Data type of Poisson process increments. Defaults to ``int``.
    lam : array-like, or callable
        Intensity of the underlying Poisson process, possibly time-dependent.
        See ``poisson_source`` class documentation.
    y : distribution, or callable, or None
        Distribution of random variates to be compounded with the
        Poisson process increments, possibly time-dependent.
        May be any ``scipy.stats`` distribution instance,
        or any object exposing an ``rvs(shape)`` method
        to generate independent random variates of the given shape,
        or a callable with ``y(t)`` evaluating to such object.
        The following preset distributions may be specified, possibly
        with time-varying parameters:
          -  ``y=norm_rv(a, b)`` - normal distribution with mean ``a``
             and standard deviation ``b``.
          -  ``y=uniform_rv(a, b)`` - uniform distribution
             between ``a`` and ``b``.
          -  ``y=exp_rv(a)`` - exponential distribution with scale ``a``.
          -  ``y=double_exp_rv(a, b, pa)`` - double exponential distribution,
             with scale ``a`` with probability ``pa``, and ``-b``
             with probability ``1 - pa``.
        where ``a, b, pa`` are array-like with values in the appropriate
        domains, broadcastable to a shape ``vshape + (paths,)``,
        or callables with ``a(t), b(t), pa(t)`` evaluating to such arrays.
        If ``None``, defaults to ``uniform_rv(a=0, b=1)``.

    Returns
    -------
    array
        Once instantiated as ``dj``, ``dj(t, dt)`` returns a random realization
        of compound Poisson process increments from time ``t`` to time
        ``t + dt``, with shape ``(t + dt).shape + vshape + (paths,)``.
        The increments are independent compound Poisson variates, consisting of
        the sum of ``N`` independent ``y`` or ``y(t + dt/2)`` variates,
        where ``N`` is a Poisson variate with mean ``lam*dt``,
        or ``lam(t + dt/2)*dt`` (approximates each variate being taken
        from ``y`` at the time of the corresponding Poisson process event).

    See Also
    --------
    poisson_source
    source
    norm_rv
    uniform_rv
    exp_rv
    double_exp_rv
    rvmap

    Notes
    -----
    Preset distributions ``norm_rv, uniform_rv, exp_rv, double_exp_rv``
    behave as follows:

        * If all parameters are array-like, return an object with an
          ``rvs`` method as described above, and with methods
          ``mean, std, var, exp_mean`` with signature ``()``, returning
          the mean, standard deviation, variance and mean of the exponential
          of the random variate.
        * If any parameter is callable, returns a callable ``y`` such
          that ``y(t)`` evaluates to the corresponding distribution
          with parameter values at time ``t``.

    To compound the Poisson process increments with a function ``f(z)``,
    or time-dependent function ``f(t, z)``, of a given random variate ``z``,
    one can pass ``y=rvmap(f, z)`` to ``compound_poisson``.

    [ToDo: make a note on martingale correction using exp_mean]

    Attributes
    ----------
    y : distribution, or callable
        Stores the distribution used computing the Poisson process increments.
    dn_value : array of int
        After each realization, this attribute stores the underlying
        Poisson process increments.
    y_value : list of array
        After each realization, this attribute stores the underlying
        ``y`` random variates.
    """

    def __init__(self, *, paths=1, vshape=(), dtype=None,
                 dn=None, ptype=int, lam=1.,
                 y=None):
        super().__init__(paths=paths, vshape=vshape, dtype=dtype)

        # setup of poisson source
        self.dn = _source_setup(dn, poisson_source,
                                paths=self.paths, vshape=self.vshape,
                                dtype=ptype, lam=lam)

        # mind not breaking the source protocol
        self.ptype = self.dn.dtype if hasattr(dn, 'dtype') else ptype
        self.lam = self.dn.lam if hasattr(dn, 'lam') else lam

        # setup of random variable sampling source
        self.y = uniform_rv(a=0, b=1) if y is None else y

    def __call__(self, t, dt):
        """See cpoisson_source class documentation."""

        t, dt = np.broadcast_arrays(t, dt)
        sign_dt = np.sign(dt).astype(int)
        shape = self.vshape + (self.paths,)
        # dn may be positive or negative according to sign(dt)
        dn = self.dn(t, dt)
        dz = np.zeros(dt.shape + shape, dtype=self.dtype)
        y = self.y
        y_value = []
        for i in np.ndindex(dt.shape):
            dn_positive_i = sign_dt[i]*dn[i]
            nmax = (dn_positive_i).max()
            rv = y(t[i] + dt[i]/2) if callable(y) else y
            for j in range(1, nmax+1):
                index = (dn_positive_i == j)
                if index.any():
                    y_sample = rv.rvs(size=(index.sum(), j))
                    dz[i][index] = sign_dt[i]*y_sample.sum(axis=-1)
                    y_value.append(y_sample)
        self.dn_value = dn
        self.y_value = y_value
        return dz


##############################################
#  Stochasticity sources with antithetic paths
##############################################

def _antithetics(source_class, transform):
    """
    Builds a source subclass generating antithetic paths
    from source_class, using the given transformation.

    The returned class is *not* a subclass
    of source_class.
    """

    class antithetics_class(source):
        def __init__(self, *, paths=2, vshape=(), **args):
            self.paths = paths
            self.vshape = _shape_setup(vshape)
            if paths % 2:
                raise ValueError(
                    'the number of paths for sources with antithetics '
                    'should be even, not {}'.format(paths))
            self._dz = source_class(paths=paths//2, vshape=vshape, **args)

        __init__.__doc__ = ("See {} class documentation"
                            .format(source_class.__name__))

        def __call__(self, t, dt=None):
            dz = self._dz(t, dt)
            return np.concatenate((dz, transform(dz)), axis=-1)

    return antithetics_class


# using this, instead of
#    >>> new_source = _antithetics(base_source, lambda z: ...)
# to get sphinx documentation right
class odd_wiener_source(_antithetics(wiener_source, lambda z: -z)):
    """
    dw, a source of standard Wiener process (Brownian motion) increments with
    antithetic paths exposing opposite increments (averages exactly to 0
    across paths).

    Once instantiated as ``dw`` with ``paths=2*K`` paths, ``x = dw(t, dt)``
    consists of leading ``K`` paths with independent increments,
    and trailing ``K`` paths consisting of a copy of the leading paths
    with sign reversed (``x[..., i] == -x[..., K + i]``).

    See Also
    --------
    wiener_source
    """
    pass


class even_poisson_source(_antithetics(poisson_source, lambda z: z)):
    """
    dn, a source of Poisson process increments with antithetic
    paths exposing identical increments.

    Once instantiated as ``dn`` with ``paths=2*K`` paths, ``x = dn(t, dt)``
    consists of leading ``K`` paths with independent increments,
    and trailing ``K`` paths consisting of a copy of the leading paths:
    (``x[..., i] == x[..., K + i]``).
    Intended to be used together with ``odd_wiener_source`` to generate
    antithetic paths in jump-diffusion processes.

    See Also
    --------
    source
    poisson_source
    """
    pass


class even_cpoisson_source(_antithetics(cpoisson_source, lambda z: z)):
    """
    dj, a source of compound Poisson process increments (jumps) with antithetic
    paths exposing identical increments.

    Once instantiated as ``dj`` with ``paths=2*K`` paths, ``x = dj(t, dt)``
    consists of leading ``K`` paths with independent increments,
    and trailing ``K`` paths consisting of a copy of the leading paths:
    ``x[..., i]`` equals ``x[..., K + i]``.
    Intended to be used together with ``odd_wiener_source`` to generate
    antithetic paths in jump-diffusion processes.

    See Also
    --------
    source
    cpoisson_source
    """
    pass


####################################
#  Stochasticity sources with memory
####################################

class _indexed_true_source:
    """Mimics a true_source, addressing part of its values
    via indexing. Used by true_source.__getitem__."""

    def __init__(self, source, vindex):
        self._source = source
        self._index = np.index_exp[vindex] + np.index_exp[..., :]

        self.paths = source.paths
        self.vshape = source(source.t0)[self._index][..., 0].shape
        self.dtype = source.dtype

    def __call__(self, t, dt=None):
        """See true_source class documentation."""
        return self._source(t, dt)[self._index]

    @property
    def size(self):
        return self._source.size

    @property
    def t(self):
        return self._source.t


class true_source(source):
    """
    Base class for stochasticity sources with memory.

    Parameters
    ----------
    paths, vshape, dtype
        See ``source`` class documentation.
    rtol : float, or 'max'
        relative tolerance used in assessing the coincidence
        of ``t`` with the time of a previously stored realization
        of the source.
        If set to ``max``, the resolution of the ``float`` type is used.
    t0, z0 : array-like
        z0 is the initial value of the source at time t0.

    Returns
    -------
    array
       Once instantiated as ``dz``, ``dz(t)`` returns the realized
       value at time ``t`` of the source process, such that
       ``dz(t0) = z0``, with shape ``(t + dt).shape + vshape + (paths,)``,
       as specified by subclasses.
       ``dz(t, dt)`` returns ``dz(t + dt) - dz(t)``.
       New values of ``dz(t)`` should follow a probability distribution
       conditional on values realized in previous calls.
       Defaults to an array of ``numpy.nan``.

    See Also
    --------
    source

    Methods
    -------
    __getitem__
    new_inside
    new_outside
    """

    def __init__(self, *, paths=1, vshape=(), dtype=None,
                 rtol='max', t0=0., z0=0.):
        self.paths, self.vshape, self.dtype = paths, vshape, dtype
        self.vshape = _shape_setup(self.vshape)
        # time points are handled one by one,
        # their type is set to float
        self.rtol = np.finfo(float).resolution \
            if rtol == 'max' else float(rtol)
        self.t0, self.z0 = t0, z0

        # initialize memory lists
        self._zlist = [self.init()]
        self._tlist = [float(self.t0)]

    def __getitem__(self, index):
        """
        Reference to a sub-array or element of the source values
        sharing the same memory of past realizations.

        Returns a ``true_source`` instance ``s[i]`` sharing with the calling
        instance ``s`` the previously stored realizations. New realizations
        will update the full extent of values for both instances.

        Notes
        -----
        If ``s.vshape == (10,)`` and ``s`` has been realized at ``t1`` but not
        at ``t2``, then: ``s[:2](t1)`` (the realization of ``s[:2]`` at ``t1``)
        will retrieve ``s(t1)[:2]`` (the sub-array of the stored realization
        of ``s`` at ``t1``); ``s[:2](t2)`` will generate and store
        all 10 values of ``s(t2)`` and return the leading two.
        """
        return _indexed_true_source(self, index)

    def __call__(self, t, dt=None):
        """See true_source class documentation."""

        if dt is None:
            t = np.asarray(t)
            return self._retrieve_old_or_generate_new(t)
        else:
            t, dt = np.broadcast_arrays(t, dt)
            s = t + dt
            return (self._retrieve_old_or_generate_new(s) -
                    self._retrieve_old_or_generate_new(t))

    def _retrieve_old_or_generate_new(self, s):
        output_shape = s.shape + self.vshape + (self.paths,)
        output = np.empty(output_shape, dtype=self.dtype)

        z, t = self._zlist, self._tlist
        rtol = self.rtol
        getvalue = self.getvalue

        def f(s):
            k = bisect.bisect_right(t, s)
            if np.isclose(s, t[k-1], rtol=rtol, atol=0.):
                return getvalue(z[k-1])
            elif k == len(t):
                z.append(self.new_outside(z[-1], t[-1], s))
                t.append(s)
                return getvalue(z[-1])
            elif k == 0:
                z.insert(0, self.new_outside(z[0], t[0], s))
                t.insert(0, s)
                return getvalue(z[0])
            else:
                z_new = self.new_inside(z[k-1], z[k], t[k-1], t[k], s)
                if z_new is not None:
                    z.insert(k, z_new)
                    t.insert(k, s)
                return getvalue(z[k])

        for i in np.ndindex(s.shape):
            output[i] = f(float(s[i]))

        return output

    # interface vs subclasses
    # -----------------------

    def init(self):
        return np.full(self.vshape + (self.paths,),
                       fill_value=self.z0,
                       dtype=self.dtype)

    def new_outside(self, z, t, s):
        """
        Generate a new process increment, at a time s above or below
        those of formerly realized values.

        Parameters
        ----------
        z : array
            Formerly realized value of the source at time ``t``.
        t, s : float
            ``t`` is the highest (lowest) time of former realizations,
            and s is above (below) ``t``.

        Returns
        -------
        array
            Value of the source at ``s``, conditional on formerly
            realized value ``z`` at ``t``. Should be defined by subclasses.
            Defaults to an array of ``numpy.nan``.
        """
        return z + np.nan

    def new_inside(self, z1, z2, t1, t2, s):
        """
        Generate a new process increment, at a time s between
        those of formerly realized values.

        Parameters
        ----------
        z1, z2 : array
            Formerly realized values of the source at times ``t1, t2``
            respectively.
        t1, t2 : float
            ``t1, t2`` are the times of former realizations closest to
            ``s``, with ``t1 < s < t2``.

        Returns
        -------
        array
            Value of the source at ``s``, conditional on formerly
            realized value ``z1`` at ``t1`` and ``z2`` at ``t2``.
            Should be defined by subclasses. Defaults to an array
            of ``numpy.nan``.
        """
        return z1 + np.nan

    def getvalue(self, z):
        return z

    def getsize(self, z):
        return z.size

    # convenience properties
    # ----------------------

    @property
    def size(self):
        """
        Returns the number of stored scalar values from previous
        evaluations, or 0 for sources without memory.
        """
        return sum(self.getsize(z) for z in self._zlist)

    @property
    def t(self):
        """
        Returns a copy of the time points at which source values
        have been stored from previous evaluations, as an array,
        or an empty array for sources without memory.
        """
        return np.array(self._tlist, dtype=float)


class true_wiener_source(true_source):
    """
dw, source of standard Wiener process (brownian motion) increments with memory.

    Parameters
    ----------
    paths, vshape, dtype, corr, rho
        See ``wiener_source`` class documentation.
    rtol, t0, z0
        See ``true_source`` class documentation.

    Returns
    -------
    array
        Once instantiated as ``dw``, ``dw(t)`` returns ``z0``
        plus a realization of the standard Wiener process increment
        from time ``t0`` to ``t``, and ``dw(t, dt)`` returns
        ``dw(t + dt) - dw(t)``.
        The returned values follow a probability distribution conditional
        on values realized in previous calls.

    See Also
    --------
    source
    wiener_source
    true_source

    Notes
    -----
    For time-independent correlations, as well as for correlations that
    depend linearly on ``t``, the resulting ``w(t)`` is exact, as
    far as it can be within the accuracy of the pseudo-random
    normal variate generator of NumPy. Otherwise,
    mind running a first evaluation of ``w(t)`` on a sequence of
    consecutive closely spaced time points in the region of interest.

    Given ``t1 < s < t2``, the value of ``w(s)`` conditional on ``w(t1)``
    and ``w(t2)`` is computed as follows.

    Let ``A`` and ``B`` be respectively the time integral of
    ``corr(t)`` between ``t1`` and ``s``, and between ``s`` and ``t2``,
    such that:
      - ``A + B`` is the expected covariance matrix of ``w(t2) - w(t1)``,
      - ``A`` is the expected covariance matrix of ``w(s) - w(t1)``,
      - ``B`` is the expected covariance matrix of ``w(t2) - w(s)``.

    Let ``Z = B @ np.linalg.inv(A + B)``, and let ``y`` be a random
    normal variate, independent from ``w(t1)`` and ``w(t2)``,
    with covariance matrix ``Z @ A`` (note that the latter is a symmetric
    matrix, as a consequence of the symmetry of ``A`` and ``B``).

    Then, the follwing expression provides for a ``w(s)`` with the
    needed correlations, and with ``w(s) - w(t1)`` independent from ``w(t1)``,
    ``w(t2) - w(s)`` independent from ``w(s)``:

    ``w(s) = Z @ w(t1) + (1 - Z) @ w(t2) + y``

    This is easily proved by direct computation of the relevant correlation
    matrices, and by using the fact that the random variables at play
    are jointly normal, and hence lack of correlation entails independence.

    Note that, when invoking ``w(s)``,
    ``A`` is approximated as ``corr((t1+s)/2)*(s-t1)``, and
    ``B`` is approximated as ``corr(s+t2)/2)*(t2-s)``.

    Methods
    -------
    See source and true_source methods.
    """

    def __init__(self, *, paths=1, vshape=(), dtype=None,
                 corr=None, rho=None,
                 rtol='max', t0=0., z0=0.):

        self._dw = wiener_source(paths=paths,
                                 vshape=vshape, dtype=dtype,
                                 corr=corr, rho=rho)
        self.corr = self._dw.corr
        super().__init__(paths=paths, vshape=vshape, dtype=dtype,
                         rtol=rtol, t0=t0, z0=z0)

    def new_outside(self, w, t, s):
        # approximate in case of time depentent correlations
        # (uses corr((t+s)/2) - exact only if time dependence is linear)
        t0, corr, dw = self.t0, self.corr, self._dw
        assert t0 <= t < s or s < t <= t0

        # hack - restore needed self._dw correlations (new_inside
        # may leave here the wrong value)
        dw.corr = corr((t + s)/2) if callable(corr) else corr
        return w + dw(t, s - t)

    @staticmethod
    def _mult(x, y):
        return np.einsum('ij,...jk->...ik', x, y)

    def new_inside(self, w1, w2, t1, t2, s):
        # upon call, always t1 < s < t2; need to
        # enforce t0 <= t1 < s < t2 or t2 < s < t1 <= t0
        t0, corr, dw = self.t0, self.corr, self._dw
        if t2 <= t0:
            w2, w1 = w1, w2
            t2, t1 = t1, t2
        assert t0 <= t1 < s < t2 or t2 < s < t1 <= t0

        # hack - override self._dw correlations to the needed value
        # (avoid instantiating a new wiener_source at each call)
        if callable(corr):
            a, b = (s - t1), (t2 - s)
            A, B = corr((t1+s)/2)*a, corr((s+t2)/2)*b
            Z = B @ np.linalg.inv(A + B)
            Id = np.eye(A.shape[0])
            dw.corr = (Z @ A)*np.sign(a)
            ws = self._mult(Z, w1) + self._mult((Id - Z), w2) + dw(0, 1)
        else:
            a, b = (s - t1), (t2 - s)
            z = b/(a + b)
            dw.corr = corr
            ws = z*w1 + (1 - z)*w2 + dw(0, z*a)

        return ws


class true_poisson_source(true_source):
    """
    dn, a source of Poisson process increments with memory.

    Parameters
    ----------
    paths, vshape, dtype, lam
        See ``poisson_source`` class documentation.
    rtol, t0, z0
        See ``true_source`` class documentation.

    Returns
    -------
    array
        Once instantiated as ``dn``, ``dn(t)`` returns ``z0`` plus
        a realization of Poisson process increments from time ``t0`` to ``t``,
        and ``dn(t, dt)`` returns ``dn(t + dt) - dn(t)``.
        The returned values follow a probability distribution conditional
        on the realized values in previous calls.

    See Also
    --------
    source
    poisson_source
    true_source

    Notes
    -----
    For time-dependent intensity ``lam(t)`` the result is approximate,
    mind running a first evaluation on a sequence of consecutive
    closely spaced time points in the region of interest.

    Methods
    -------
    See ``source`` and ``true_source`` methods.
    """
    def __init__(self, *, paths=1, vshape=(), dtype=int, lam=1.,
                 rtol='max', t0=0., z0=0):

        super().__init__(paths=paths, vshape=vshape, dtype=dtype,
                         rtol=rtol, t0=t0, z0=z0)
        self._dn = poisson_source(paths=self.paths,
                                  vshape=self.vshape,
                                  dtype=self.dtype,
                                  lam=lam)
        self.lam = self._dn.lam

    def new_outside(self, n, t, s):
        return n + self._dn(t, s - t)

    def new_inside(self, n1, n2, t1, t2, s):
        if (n1 == n2).all():
            return None
        p = (s - t1)/(t2 - t1)
        n = n2 - n1
        return n1 + np.random.binomial(n, p, n.shape)


class true_cpoisson_source(true_source):
    """
    dj, a source of compound Poisson process increments (jumps) with memory.

    Parameters
    ----------
    paths, vshape, dtype, dn, ptype, lam, y
        See ``cpoisson_source`` class documentation.
    rtol, t0, z0
        See ``true_source`` class documentation.

    Returns
    -------
    array
        Once instantiated as ``dj``, ``dj(t)`` returns ``z0`` plus
        a realization of compound Poisson process increments from time ``t0``
        to ``t``, and ``dj(t, dt)`` returns ``dj(t + dt) - dj(t)``.
        The returned values follow a probability distribution conditional
        on the realized values in previous calls.

    See Also
    --------
    source
    cpoisson_source
    true_source

    Notes
    -----
    For time-dependent intensity ``lam(t)`` and compounding random
    variable ``y(t)`` the result is approximate,
    mind running a first evaluation on a sequence of consecutive
    closely spaced time points in the region of interest.

    Methods
    -------
    See ``source`` and ``true_source`` methods.
    """
    def __init__(self, *, paths=1, vshape=(), dtype=None,
                 rtol='max', t0=0., z0=0.,
                 dn=None, ptype=int, lam=1.,
                 y=None):

        super().__init__(paths=paths, vshape=vshape, dtype=dtype,
                         rtol=rtol, t0=t0, z0=z0)
        if dn is None:
            dn = true_poisson_source(paths=paths, vshape=vshape,
                                     dtype=ptype, lam=lam)
        self._dj = cpoisson_source(paths=self.paths,
                                   vshape=self.vshape, dtype=self.dtype,
                                   dn=dn, y=y)  # ptype, lam are set by dn
        self.ptype, self.lam = self._dj.ptype, self._dj.lam
        self.dn = self._dj.dn
        self.y = self._dj.y

    def init(self):
        # the 'z' values stored by true_source consist
        # each of a list of two, the value of j and the
        # y_value looking forward from the current to the
        # next time point. For the last time point, y_value
        # is set to None
        # Note: using a list, not a tuple, to allow to
        # modify y_value
        return [super().init(), None]

    def _decode_y(self, dn_value, y_value):
        nmax = dn_value.max()
        yy = np.zeros(dn_value.shape + (nmax,),
                      dtype=self.dtype)
        for y in y_value:
            i = y.shape[-1]
            index = (dn_value == i)
            yy[index, :i] = y

        ii = [y.shape[-1] for y in y_value]
        jj = [i for i in range(1, nmax + 1) if (dn_value == i).any()]
        assert ii == jj
        return yy

    def _encode_y(self, dn_value, yy):
        nmax = dn_value.max()
        assert nmax == (yy != 0).sum(axis=-1).max()

        y_value = []
        for i in range(1, nmax + 1):
            index = (dn_value == i)
            if index.any():
                y = yy[index, :i]
                assert (yy[index, i:] == 0).all()
                y_value.append(y)
        return y_value

    def new_outside(self, z, t, s):
        j, y_value = z
        dj = self._dj(t, s - t)
        y_new = self._dj.y_value
        if s > t:
            z[1] = y_new
            return [j + dj, None]
        else:
            return [j + dj, y_new]

    def new_inside(self, z1, z2, t1, t2, s):
        j1, y1_value = z1
        j2, _ = z2  # y2_value not needed

        if (j1 == j2).all():
            return None

        dn = self._dj.dn
        n1, ns, n2 = dn(t1), dn(s), dn(t2)  # use memory of dn

        dn_t2_t1 = n2 - n1
        dn_s_t1 = ns - n1
        dn_t2_s = n2 - ns
        nmax = dn_t2_t1.max()

        # decode y_value
        yy1 = self._decode_y(dn_t2_t1, y1_value)

        # split y_value
        yy1_updated = yy1.copy()
        yy_new = np.zeros_like(yy1)
        for i in range(dn_s_t1.max() + 1):
            index = (dn_s_t1 == i)
            yy1_updated[index, i:] = 0
            yy_new[index, :nmax-i] = yy1[index, i:]

        # encode yy1_updated and yy_new
        y1_updated = self._encode_y(dn_s_t1, yy1_updated)
        y_new = self._encode_y(dn_t2_s, yy_new)

        # compute dj from t1 to s
        dj = yy1_updated.sum(axis=-1)

        # store/return result
        z1[1] = y1_updated
        return [j1 + dj, y_new]

    def getvalue(self, z):
        return z[0]

    def getsize(self, z):
        j, y_value = z
        return j.size + (0 if y_value is None else
                         sum(y.size for y in y_value))

    @property
    def size(self):
        return super().size + self._dj.dn.size


#######################
#  The montecarlo class
#######################

class montecarlo:
    """
    Summary statistics of Monte Carlo simulations.

    Compute, store and cumulate results of Monte Carlo simulations
    across multiple runs. Cumulated results include mean, standard deviation,
    standard error, skewness, kurtosis, and 1d-histograms of the distribution
    of outcomes. Probability distribution function estimates are provided,
    based on the cumulated histograms.

    Parameters
    ----------
    sample : array-like, optional
        Initial data set to be summarized.
        If ``None``, an empty instance is provided, initialized with
        the given parameters.
    axis : integer, optional
        Axis of the given ``sample`` enumerating single data points
        (paths, or different realizations of a simulated process or event).
        Defaults to the last axis of the sample.
    use : {'all', 'even', 'odd'}, optional
        If ``'all'`` (default), the data set is processed as is.
        If ``'even'`` or ``'odd'``, the sample ``x`` is assumed to consist
        of antithetic values along the specified axis,
        assumed of even size ``2*N``, where ``x[0], x[1], ...``
        is antithetic respectively to  ``x[N], x[N+1], ...``.
        Summary operations are then applied to a sample of size ``N``
        consisting of the half-sum (``'even'``) or half-difference (``'odd'``)
        of antithetic values.
    bins : array-like, or int, or str, optional
        Bins used to evaluate the counts' cumulated distribution are computed,
        against the first data set encountered, according
        to the ``bins`` parameter:
          -  If ``int`` or ``str``, it dictates the number of bins or their
             determination method, as passed to ``numpy.histogram``
             when processing the first sample.
          -  If array-like, overrides ``range``, setting explicit bins'
             boundaries, so that ``bins[i][j]`` is the lower bound
             of the ``j``-th bin used for the distribution of the
             ``i``-th component of data points.
          -  If ``None``, no distribution data will be computed.
        Defaults to ``100``.
    range : (float, float) or None, optional
        Bins range specification, as passed to ``numpy.histogram``.
    dtype : data-type, optional
        Data type used for cumulating moments. If ``None``, the data-type
        of the first sample is used, if of float kind, or ``float``
        otherwise.
    ctype : data-type, optional
        Data type used for cumulating histogram counts.
        Defaults to ``numpy.int64``.

    Notes
    -----
    The shape of cumulated statistics is set as the shape of the
    data points of the first data set processed (shape of the first
    ``sample`` after summarizing along the paths axis). When cumulating
    subsequent samples, broadcasting rules apply.

    Indexing can be used to access single values or slices of the
    stored data. Given a montecarlo instance ``a``, ``a[i]`` is a new
    instance referencing statistics of the ``i``-th component of
    data summarized in ``a`` (no copying).

    The first data set encountered fixes the histogram bins.
    Points of subsequent data sets that fall outside the bins,
    while properly taken into account in summary statistics
    (mean, standard error etc.), are ignored when building
    cumulated histograms and probability distribution functions.
    Their number is accounted for in the ``outpaths`` property
    and ``outerr`` method.

    Histograms and distributions, and the related ``outpaths``
    and ``outerr``, must be invoked on single-valued ``montecarlo``
    instances. For multiple valued simulations, use indexing
    to select the value to be addressed (e.g. ``a[i].histogram()``).

    Attributes
    ----------
    paths
    vshape
    shape
    outpaths
    m
    s
    e
    stats
    h
    dh

    Methods
    -------
    update
    mean
    var
    std
    skew
    kurtosis
    stderr
    histogram
    density_histogram
    pdf
    cdf
    outerr
    """

    # initialization and paths/shape properties
    # -----------------------------------------
    def __init__(self, sample=None, axis=-1,
                 bins=100, range=None, use='all',
                 dtype=None, ctype=np.int64):

        self.dtype, self.ctype = dtype, ctype
        # paths number is stored as a pointer
        # (self._paths[0] is shared in read-write access by instances
        # returned by __getitem__)
        self._paths = [0]
        self._bins = bins
        self._range = range
        self._use = use
        if sample is not None:
            self.update(sample=sample, axis=axis)
        else:
            self._mean = self._moments = self._counts = None

    @property
    def paths(self):
        """
        Number of cumulated sample data points
        (``0`` for an empty instance).
        """
        return self._paths[0]

    @property
    def vshape(self):
        """Shape of cumulated sample data points."""
        if self._moments is None:
            raise ValueError('no sample data: vshape not defined')
        return self._moments[0].shape

    @property
    def shape(self):
        """
        Shape of cumulated sample data set, rearranged with
        averaging axis as last axis.
        """
        return self.vshape + (self.paths,)

    # methods to update moments and distribution data
    # according to new sample data
    # -----------------------------------------------
    def update(self, sample, axis=-1):
        """
        Add the given sample to the montecarlo simulation.

        Combines the given sample data with summary statistics
        obtained (if any) from former samples to which the ``montecarlo``
        instance was exposed at instantiation and at previous calls
        to this method. Updates cumulated statistics and histograms
        accordingly.

        Parameters
        ----------
        sample : array-like
            Data set to be summarized.
        axis : integer, optional
            Axis of the given ``sample`` enumerating single data points
            (paths, or different realizations of a simulated process or event).
            Defaults to the last axis of the sample.
        """

        # prepare sample with paths axis as last axis
        sample = np.asarray(sample)
        if sample.ndim == 0:
            sample = sample.reshape(1)
        sample = np.moveaxis(sample, axis, -1)
        sample_paths = sample.shape[-1]

        # use all, even or odd sample values
        # for antithetics sampling
        # (with 2*N samples, sample[k] is assumed to be antithetic
        # to sample[N+k])
        use = self._use
        if use not in ('all', 'even', 'odd'):
            raise ValueError(
                "use must be one of 'all', 'even', 'odd', not {}"
                .format(use))
        if use != 'all':
            if sample_paths % 2:
                raise ValueError(
                    'the sample axis for even or odd antithetics sampling '
                    'should be of even length, but {} was found'
                    .format(sample_paths))
            sample_paths //= 2
            sign = 1 if use == 'even' else -1
            sample = (sample[..., :sample_paths] +
                      sign*sample[..., sample_paths:])/2

        # set flag to identify first run
        isfirst = (self.paths == 0)

        # compute/cumulate value, error and stats
        self._update_moments(isfirst, sample)
        self._update_histogram(isfirst, sample)
        self._paths[0] += sample_paths

    def _update_moments(self, isfirst, sample, max_moment=4):
        if isfirst:
            # initialize moments upon first call (this sets vshape)
            vshape = sample.shape[:-1]
            dtype = ((sample.dtype if sample.dtype.kind == 'f'
                      else float) if self.dtype is None
                     else self.dtype)
            self._moments = tuple(np.zeros(vshape, dtype=dtype)
                                  for i in range(max_moment))
            self._mean = np.zeros(vshape, dtype=dtype)
            self._center = sample.mean(axis=-1).astype(dtype)

        # number of paths already stored N and new to be added M
        N, M = self.paths, sample.shape[-1]

        # allocate memory
        s = tuple(np.zeros(sample.shape, float)
                  for k in range(max_moment))

        # compute powers of (sample - self._center)
        s[0][...] = sample - self._center[..., np.newaxis]
        for i in range(1, max_moment):
            s[i][...] = s[i-1]*s[0]

        # compute moments (centered on the average of the first sample)
        # and cumulate with previous results
        for i in range(max_moment):
            sample_moment = s[i].mean(axis=-1)
            self._moments[i][...] = \
                (N*self._moments[i] + M*sample_moment)/(N + M)

        # compute cumulated mean
        self._mean[...] = (N*self._mean + M*sample.mean(axis=-1))/(N + M)

        # release memory
        del s

    def _update_histogram(self, isfirst, sample):
        # if no histogram is required, exit
        if self._bins is None:
            return

        # number of paths already stored N and new to be added M
        N, M = self.paths, sample.shape[-1]
        # shape of values (one histogram is computed
        # for each index in vshape)
        vshape = self.vshape

        if isfirst:
            # initializations and computations for the first sample:
            #   self._bins are initialized via
            #   np.histogram unless explicitly
            #   given as an appropriately shaped array-like object
            mybins = self._bins
            self._bins = np.empty(vshape, dtype=object)
            self._counts = np.empty(vshape, dtype=object)
            self._paths_outside = np.zeros(vshape, dtype=self.ctype)
            args = np.empty(vshape, dtype=object)
            if isinstance(mybins, (int, str)):
                # setup if no bins are provided
                args[...] = dict(bins=mybins, range=self._range)
            else:
                # setup if bins are explicitly given (range is ignored)
                mybins = np.asarray(mybins)
                if (mybins.shape[:-1] == vshape) or \
                   (mybins.dtype == object and mybins.shape == vshape):
                    for i in np.ndindex(vshape):
                        self._bins[i] = mybins[i]
                        args[i] = dict(bins=mybins[i])
                else:
                    raise ValueError(
                        'shape of the bins {} not compatible with '
                        'the shape {} of sample data points'
                        .format(mybins.shape, vshape)
                        )
            for i in np.ndindex(vshape):
                self._counts[i], self._bins[i] = \
                        np.histogram(sample[i], **args[i])
                self._counts[i] = self._counts[i].astype(self.ctype,
                                                         copy=False)
                self._paths_outside[i] = (N + M - self._counts[i].sum())

        else:
            # computations for subsequent samples:
            #   histograms of subsequent samples are generated using
            #   previously stored bins and cumulated
            for i in np.ndindex(vshape):
                counts, bins = np.histogram(sample[i], bins=self._bins[i])
                self._counts[i] += counts
                self._paths_outside[i] += (M - counts.sum())

        # a final consistency check
        for i in np.ndindex(vshape):
            if self._counts[i].sum() + self._paths_outside[i] != (N + M):
                raise RuntimeError(
                    'total number of cumulated paths inconsistent with stored '
                    'cumulated counts - may occur if a multiple valued '
                    'simulation is update in only part of its components')

    # indexing of montecarlo objects
    # ------------------------------
    def __getitem__(self, i):
        """See montecarlo class documentation"""
        a = montecarlo()
        a._paths = self._paths
        if (self._bins is None) or isinstance(self._bins, (int, str)):
            a._bins = self._bins
            a._range = self._range
        else:
            a._bins = self._bins[i]
        if self.paths != 0:
            a._mean = self._mean[i]
            a._moments = tuple(moment[i] for moment in self._moments)
            a._counts = self._counts[i]
            a._paths_outside = self._paths_outside[i]
        return a

    # user access to statistics
    # -------------------------
    def mean(self):
        """Mean of cumulated sample data points."""
        return self._mean

    def var(self):
        """Variance of cumulated sample data points."""
        m1, m2 = self._moments[:2]
        return (0.*m1 if self.paths < 2 else m2 - m1*m1)

    def std(self):
        """Standard deviation of cumulated sample data points."""
        return sqrt(self.var())

    def skew(self):
        """Skewness of cumulated sample data points."""
        m1, m2, m3 = self._moments[:3]
        return (0.*m1 if self.paths < 2 else
                (m3 - 3*m1*m2 + 2*m1**3)/(m2 - m1*m1)**1.5)

    def kurtosis(self):
        """Kurtosis of cumulated sample data points."""
        m1, m2, m3, m4 = self._moments[:4]
        return (-3.0 +
                0.*m1 if self.paths < 2 else
                (m4 - 4*m1*m3 + 6*m1*m1*m2 - 3*m1**4)/(m2 - m1*m1)**2)

    def stderr(self):
        """
        Standard error of the mean of cumulated sample data points.

        ``a.stderr()`` equals ``a.std()/sqrt(a.paths - 1)``.
        """
        return (np.nan if self.paths < 2
                else sqrt(self.var()/(self.paths - 1)))

    def __repr__(self):
        if self.paths == 0:
            return '<empty montecarlo object>'
        else:
            mean, err = np.asarray(self.mean()), np.asarray(self.stderr())
            if mean.size == 1 and err.size == 1:
                mean = mean.flatten()[0]
                err = err.flatten()[0]
            return repr(mean) + ' +/- ' + repr(err)

    # user access to distribution data
    # --------------------------------
    def histogram(self):
        """
        Distribution of the cumulated sample data, as a counts histogram.

        Returns a ``(counts, bins)`` tuple of arrays representing the
        one-dimensional histogram data of the distribution of cumulated samples
        (as returned by ``numpy.histogram``).
        """

        if self.paths == 0 or self._bins is None:
            raise ValueError('no distribution data available')

        counts, bins = self._counts, self._bins
        if (counts.dtype == object and counts.size > 1):
            raise IndexError(
                'histograms and distributions must be invoked '
                'on single-valued montecarlo instances; '
                'use indexing to select the value to be addressed '
                '(es. ``a[i].histogram()``)'
                )
        if (counts.dtype == object and counts.size == 1):
            assert bins.dtype == object and bins.size == 1
            counts = counts.flatten()[0]
            bins = bins.flatten()[0]

        return counts, bins

    def density_histogram(self):
        """
        Distribution of the cumulated sample data, as a normalized counts
        histogram.

        Returns a ``(counts, bins)`` tuple of arrays representing the
        one-dimensional density histogram data of the distribution of cumulated
        samples (as returned by ``numpy.histogram``, the sum of the counts
        times the bins' widths is 1).

        May systematically over-estimate the probability distribution within
        the bins' boundaries if part of the cumulated samples data
        (accounted for in the ``outpaths`` property and ``outerr`` method)
        fall outside.
        """
        counts, bins = self.histogram()  # raises error if not single valued
        return counts/counts.sum()/np.diff(bins), bins

    @property
    def outpaths(self):
        """
        Data points fallen outside of the bins' boundaries.
        """
        counts, bins = self.histogram()  # raises error if not single valued
        return self._paths_outside

    def outerr(self):
        """Fraction of cumulated data points fallen outside
        of the bins' boundaries.
        """
        return self.outpaths/self.paths

    def _pdf_or_cdf(self, x, *, method, bandwidth, kind, cdf_flag):
        """Compute pdf or cdf, evaluated at x"""

        ncounts, bins = self.density_histogram()
        x = np.asarray(x)

        if method == 'gaussian_kde':
            newaxes = (1,)*x.ndim
            deltas = np.diff(bins).reshape(newaxes + ncounts.shape)
            widths = deltas*bandwidth
            midpoints = ((bins[:-1]+bins[1:])/2). \
                reshape(newaxes + ncounts.shape)
            weights = ncounts.reshape(newaxes + ncounts.shape)

            if cdf_flag:
                def kernel(x):
                    return (scipy.special.erf(x/sqrt(2))+1)/2
                kernels = widths/bandwidth * \
                    kernel((x[..., np.newaxis] - midpoints)/widths)
            else:
                def kernel(x):
                    return exp(-x*x/2)/sqrt(2*np.pi)
                kernels = 1/bandwidth * \
                    kernel((x[..., np.newaxis] - midpoints)/widths)

            pdf_or_cdf = (kernels*weights).sum(axis=-1)
            return pdf_or_cdf

        elif method == 'interp':
            if cdf_flag:
                xx = bins
                pp = np.empty((bins.size,), dtype=ncounts.dtype)
                deltas = np.diff(bins)
                pp[0], pp[1:] = (0, (deltas*ncounts).cumsum())
                fill_value = (0., 1.)
            else:
                xx = np.empty((bins.size+1,), dtype=bins.dtype)
                xx[0], xx[1:-1], xx[-1] = (bins[0], (bins[:-1] + bins[1:])/2,
                                           bins[-1])
                pp = np.empty((bins.size+1,), dtype=ncounts.dtype)
                pp[0], pp[1:-1], pp[-1] = (0, ncounts, 0)
                fill_value = 0.

            pdf_or_cdf = scipy.interpolate.interp1d(
                xx, pp, kind=kind, assume_sorted=True,
                bounds_error=False, copy=False, fill_value=fill_value
                )(x)
            return pdf_or_cdf

        else:
            raise ValueError('pdf or cdf method should be '
                             "'gaussian_kde' or 'interp', not {}"
                             .format(method))

    def pdf(self, x, method='gaussian_kde', bandwidth=1., kind='linear'):
        """
        Normalized sample probability density function, evaluated at ``x``.

        Parameters
        ----------
        x : array-like
            Values at which to evaluate the pdf.
        method : {'gaussian_kde', 'interp'}
            Specifies the method used to estimate the pdf value. One of:
            'gaussian_kde' (default), smooth Gaussian kernel
            density estimate of the probability density function;
            'interp', interpolation of density histogram values, of the
            given ``kind``.
        bandwidth : float
            The bandwidth of Gaussian kernels is set to ``bandwidth``
            times each bin width.
        kind : str
            Interpolation kind for the 'interp' method, passed to
            ``scipy.interpolate.intep1d``.

        Returns
        -------
        array
            An estimate of the sample probability density function of the
            cumulated sample data, at the given 'x' values,
            according to the stated method.

        Notes
        -----
        For the 'gaussian_kde' method, kernels are computed at bins midpoints,
        weighted according to the density histogram counts,
        using in each bin a bandwidth set to ``bandwidth`` times the bin width.
        The resulting pdf:
          -  Has support on the real line.
          -  Integrates exactly to 1.
          -  May not closely track the density histogram counts.

        For the 'interp' method, the pdf evaluates to
        the density histogram counts at each bin midpoint,
        and to 0 at the bins boundaries and outside. The resulting pdf:
          -  Has support within the bins boundaries.
          -  Is intended to track the density histogram counts.
          -  Integrates close to, but not exactly equal to, 1.

        May systematically overestimate the probability distribution within
        the bins' boundaries if part of the cumulated samples data
        (accounted for in the ``outpaths`` property and ``outerr`` method)
        fall above or below the bins boundaries.
        """
        return self._pdf_or_cdf(x, method=method, bandwidth=bandwidth,
                                kind=kind, cdf_flag=False)

    def cdf(self, x, method='gaussian_kde', bandwidth=1., kind='linear'):
        """
        Cumulative sample probability density function, evaluated at ``x``.

        See ``pdf`` method documentation.

        Notes
        -----
        For the 'gaussian_kde' method, the integral of the Gaussian kernels
        is expressed in terms of ``scipy.special.erf``, and coincides with
        the integral of the pdf computed with the same method.

        For the 'interp' method, the cdf evaluates as follows:
          -  At bin endpoints, to the cumulated density histogram values
             weighed by the bins width.
          -  Below the bins boundaries, to 0.
          -  Above the bins boundaries, to 1.

        It is close to, but not exactly equal to, the integral of the pdf
        computed with the same method.
        """
        return self._pdf_or_cdf(x, method=method, bandwidth=bandwidth,
                                kind=kind, cdf_flag=True)

    # shortcuts
    # ---------
    @property
    def m(self):
        """Shortcut for the ``mean`` method."""
        return self.mean()

    @property
    def s(self):
        """Shortcut for the ``std`` method."""
        return self.std()

    @property
    def e(self):
        """Shortcut for the ``stderr`` method."""
        return self.stderr()

    @property
    def stats(self):
        """Dictionary of cumulated statistics."""
        return dict(mean=self.mean(), stderr=self.stderr(),
                    std=self.std(), skew=self.skew(),
                    kurtosis=self.kurtosis())

    @property
    def h(self):
        """Shortcut for the ``histogram`` method."""
        return self.histogram()

    @property
    def dh(self):
        """Shortcut for the ``density_histogram`` method."""
        return self.density_histogram()
