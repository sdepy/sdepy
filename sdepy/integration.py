"""
====================================
SDE INTEGRATION AND
REALIZATIONS OF STOCHASTIC PROCESSES
====================================

*  ``paths_generator``, ``integrator``, ``SDE`` and ``SDEs`` classes,
*  ``integrate`` decorator,
*  realization of stochastic processes.
"""

import numpy as np
from numpy import sqrt, exp, log
from .infrastructure import (
    process,
    wiener_source, poisson_source, cpoisson_source,
    norm_rv, double_exp_rv
    )


########################################
#  Private functions for recurring tasks
########################################

from .infrastructure import (
    _shape_setup,
    _const_param_setup,
    _variable_param_setup,
    _source_setup,
    _signature,
    _empty
    )


############################
#  The paths_generator class
############################

class paths_generator:
    """
    Step by step generation of stochastic simulations across multiple
    paths, intended for subclassing.

    Given a number of requested paths, shapes and output timeline,
    encapsulates the low level tasks of memory allocation and step by step
    iteration along the timeline.

    The definition of the target iteration steps (``pace`` method),
    initialization (``begin`` method),
    computation of next step (``next`` method),
    storing results at points on the requested timeline (``store`` method),
    cleaning up (``end`` method) and
    evaluation of a final result to be returned (``exit`` method),
    are delegated to subclasses.

    Instances are callables with signature ``(timeline)`` that iterate
    subclass methods along the given timeline, using the configuration
    set out at instantiation.

    Parameters
    ----------
    paths : int
        Size of last axis of the allocated arrays
        (number of paths of the simulation).
    xshape : int or tuple of int
        Shape of values that will be stored at each point
        of the output timeline.
    wshape : int or tuple of int
        Shape of working space used for step by step iteration.
    dtype : data-type
        Data-type of the output and working space.
    steps : iterable, or int, or None
        Specification of the time points to be touched during the simulation
        (as defined by the ``pace`` method). Default behaviour is:
          -  if ``None``, the simulated steps coincide with the timeline;
          -  if ``int``, the simulated steps touch all timeline points,
             as well as ``steps`` equally spaced points between the minimum
             and maximum point in the timeline;
          -  if iterable, the simulated steps touch all timeline points,
             as well as all values in ``steps`` between the minimum and maximum
             points in the timeline.
    i0 : int
        Index along the timeline at which the simulation starts. The timeline
        is assumed to be in ascending order. The simulation is performed
        backwards from ``timeline[i0]`` to ``timeline[0]``, and forwards
        from ``timeline[i0]`` to ``timeline[-1]``.
    info : dict, or None
        Diagnostic information about the simulation is stored in this
        dictionary and is accessible as the ``info`` attribute.
        If ``None``, a new empty ``dict`` is used.
    getinfo : bool
        If ``True`` (default), records basic information in the ``info``
        attribute about the last performed simulation (if the simulation is
        both backwards and forwards, the information pertains to the forwards
        part only). Used by subclasses to enable/disable diagnostic
        info generation.

    Returns
    -------
    simulation results
        Once instantiated as ``p``, ``p(timeline)`` runs the simulation
        along the given timeline, based on parameters of instantiation,
        returning results as determined by subclass methods.
        Defaults to ``(tt, xx)`` where ``tt`` is a reference to ``timeline``
        and ``xx`` is an array of ``numpy.nan`` of the requested shape.

    See Also
    --------
    integrator
    SDE
    SDEs

    Notes
    -----
    All initialization parameters are stored as attributes of the same name,
    and may be accessed by subclasses.

    During the simulation, a ``itervars`` attribute is present,
    pointing to a dictionary that contains the following items,
    to be used by subclass methods (double letters refer to values
    along the entire timeline, single letters refer to single time points):

        *  ``steps_tt`` : an array of all time points to be touched
           by the simulation. It consolidates the output timeline
           and the time points to be touched, as specified by ``steps``.
        *  ``tt``: the output timeline.
        *  ``xx``: simulation output, an array of shape
           ``tt.shape + xshape + (paths,)``. ``xx[i]`` is the simulated value
           at time ``tt[i]``.
        *  ``sw``: working space for time points, an array
           of shape ``(depth,)``.
        *  ``xw``: working space for paths generation, an array of objects
           of shape ``(depth,)``, where each of ``xw[k]`` is an array
           of shape ``wshape + (paths,)``.
        *  ``reverse`` : ``True`` if the simulation runs backwards,
           ``False`` otherwise. If ``True``, ``steps_tt`` and ``tt`` are
           in descending order.
        *  ``i`` : such that ``tt[i]`` is the next point
           along the timeline that will be reached (when invoking ``next``),
           or the point that was just reached (when invoking ``store``).

    Note that:
        *  ``sw`` and ``xw`` are rolled at each iteration: subclass methods
           should not rely on storing references to their elements
           across iterations.
        *  ``xw[k][...] = value`` broadcasts ``value`` into the allocated
           memory of ``xw[k]`` (this is usually what you want),
           ``xw[k] = value`` stores ``value``, as an object, in ``xw[k]``
           (avoid).
        *  ``xx`` and ``xw[k]`` are initialized to arrays filled
           with ``numpy.nan``.

    Attributes
    ----------
    depth : int
        Number of time points to be stored in the working space. Defaults to 2.

    Methods
    -------
    __call__
    pace
    begin
    next
    store
    end
    exit
    """

    def __init__(self, *, paths=1,
                 xshape=(), wshape=(),
                 dtype=None, steps=None, i0=0,
                 info=None, getinfo=True):

        self.paths = paths
        self.xshape = _shape_setup(xshape)
        self.wshape = _shape_setup(wshape)
        self.dtype = dtype
        self.steps, self.i0 = steps, i0
        self.info = {} if info is None else info
        self.getinfo = getinfo
        super().__init__()

    # public interface vs subclasses
    # ------------------------------

    depth = 2  # should be set in subclasses to be an integer >= 2

    def pace(self, timeline):
        """
        Target integration steps for the current integration.

        Parameters
        ----------
        timeline : array
            Requested simulation timeline, cast as an array of float data-type.

        Returns
        -------
        array
            Target time points to be touched during the simulation
            (typically, more thinly spaced than the output time
            points in ``timeline``, based on the ``steps`` parameter),
            to be merged with ``timeline``.

        May be overridden by subclasses. For default behaviour,
        see paths_generator class documentation.
        """
        steps = self.steps
        ttype = timeline.dtype
        if timeline.size == 1:
            return timeline
        elif steps is None:
            return np.array((), dtype=ttype)
        elif np.isscalar(steps):
            return np.linspace(timeline[0], timeline[-1], steps, dtype=ttype)
        else:
            return np.fromiter(steps, dtype=ttype)

    def begin(self):
        """
        Set initial conditions.

        Given the time points ``sw[0], ..., sw[depth - 2]``,
        should define and store in the working space
        the corresponding initial values ``xw[0], ..., xw[depth - 2]``.
        Note that when ``begin`` gets called,
        ``sw[depth - 1]`` and ``xw[depth - 1]`` are undefined and will be
        respectively set, and computed, at the first iteration.

        Notes
        -----
        It is called once for each backwards and forwards simulation,
        after memory allocation and before starting the iteration along
        the time points in ``steps_tt``.

        Outline of expected code for ``depth=2``::

            # access itervars
            iv = self.itervars
            sw, xw = iv['sw'], iv['xw']

            # this is the initial time, taken from the
            # simulation timeline
            t0 = sw[0]
            assert t0 == iv['steps_tt'][0] == iv['tt'][0]

            # store the initial condition
            xw[0][...] = 1.

        Must be provided by subclasses.
        """
        pass

    def next(self):
        """
        Numerical simulation step.

        Given the points ``sw[0], ..., sw[depth - 2]``
        and the corresponding values ``xw[0], ..., xw[depth - 2]``,
        should:
            1. Optionally modify the target next time point ``sw[depth - 1]``,
               to a value between ``sw[depth - 2]`` and ``sw[depth - 1]``
               (this allows for adaptive time steps, with the constraint of
               touching all point specified by the output timeline and the
               ``steps`` parameter).
            2. Compute the corresponding value ``xw[depth - 1]``

        Notes
        -----
        It is called once per iteration step.

        Outline of expected code for ``depth=2``::

            # access itervars
            iv = self.itervars
            sw, xw = iv['sw'], iv['xw']

            # get starting values, and time step to be taken
            s0, x0 = sw[0], xw[0]
            ds = sw[1] - sw[0]

            # compute and store the next step
            xw[1][...] = x0 + ds

        Must be provided by subclasses.
        """
        pass

    def store(self, i, k):
        """
        Store the current integration step into the integration output.

        Should take the k-th value in the working space ``xw``,
        transform it if need be, and store it as the output
        ``xx[i]`` at the output time point ``tt[i]``.

        Parameters
        ----------
        i : int
            Index of the output timeline point to set as output.
        k : int
            Index of the working space point to use as input.

        Notes
        -----
        It is called initially to store the initial values that belong
        to the output timeline, among those put into the working space
        by ``begin``, and later during the iteration, each time the simulation
        touches a point on the output timeline.

        Outline of expected code for ``xshape == wshape`` and
        an exponentiation transformation::

            # access itervars
            iv = self.itervars
            sw, xw = iv['sw'], iv['xw']
            xx = iv['xx']

            # this is the current time, also found
            # along the output timeline
            s = sw[k]
            assert s == iv['tt'][i]

            # transform and store
            np.exp(xw[k], out=xx[i])

        Must be provided by subclasses.
        """
        pass

    def end(self):
        """
        End of iteration optional tasks.

        It is called once for each backwards and forwards simulation,
        once the final point in the output timeline has been reached
        and the simulation ends.

        After it is called, ``itervars`` are deleted.

        May be provided by subclasses.
        """
        pass

    def exit(self, tt, xx):
        """
        Final tasks and construction of the output value(s).

        Parameters
        ----------
        tt : array
            Output timeline. It is the timeline passed to the ``__call__``
            method, cast as an array, with its original data-type
            (if the data-type is of integer kind, the simulation
            is carried out using floats).
        xx : array
            Output values along the timeline, as computed by ``next``
            and stored by ``store`` methods.

        Notes
        -----
        It is called once, after backwards and/or forwards simulations
        have been completed, and its return value is returned.

        Default implementation::

            return tt, xx

        May be provided by subclasses.
        """
        return tt, xx

    # private methods
    # ---------------

    def _store_if_reached(self, i, k):
        iv = self.itervars
        tt, sw, reverse = iv['tt'], iv['sw'], iv['reverse']
        diff = (tt[i] - sw[k]) * (-1 if reverse else 1)
        if diff < 0:  # should never occur
            raise RuntimeError(
                'invalid path generation step encountered in {}: '
                'realized s={} is beyond target value t={}'
                .format(self, sw[k], tt[i]))
        elif tt[i] == sw[k]:
            self.store(i, k)
            return 1
        else:
            return 0

    def _generate_paths(self, steps_tt, tt, xx, sw, xw, reverse):
        K = self.depth

        # set itervars to be used by subclass methods
        # -------------------------------------------
        self.itervars = dict(steps_tt=steps_tt, tt=tt, xx=xx,
                             sw=sw, xw=xw, reverse=reverse)

        # set steps iterator
        # ------------------
        st = iter(steps_tt)

        # initialize working space
        # ------------------------
        # set sw[0] ... sw[K-2]
        for k in range(K-1):

            sw[k] = next(st)
        # invoke cooperative subclass method 'being'
        # (should initialize xw[0:K-1], given sw[0:K-1])
        self.begin()

        # store initial condition x[0] at t[0] == sw[0], and possibly
        # other output values if any of the sw[1], ..., sw[K-1] is
        # part of the requested timeline t
        i = 0
        for k in range(K-1):
            # invoke cooperative subclass method 'store'
            i += self._store_if_reached(i, k)

        # MAIN LOOP
        # ---------
        target_s = current_s = sw[-2]  # any same value will do
        while i < tt.size:
            # when the iteration begins:
            #   sw[0:K-1], xw[0:K-1] refer to the last stored K-1 steps
            #   s[K-1] == s[-1], xw[K-1] == xw[-1] are undefined
            #   and will be used to store the next computed step
            #   i is such that x[i] is the last stored output value
            # ------------------------------------------------------

            # if the previous step reached target_s, get next time point
            if current_s == target_s:
                target_s = next(st)
            # set sw[-1] to the target time point
            sw[-1] = target_s

            self.itervars['i'] = i

            # call cooperative subclass method 'next'
            # (should decrease or leave unchanged, but not
            # increase, the time point sw[-1], and compute the
            # corresponding xw[-1])
            self.next()

            # check the adjusted step size is within permitted bounds
            order_ok = (sw[-2] > sw[-1] >= target_s
                        if reverse else sw[-2] < sw[-1] <= target_s)
            if not order_ok:
                raise RuntimeError(
                    'invalid path generation step encountered in {}: '
                    's={} was followed by s+ds={}, beyond requested s+ds={}'
                    .format(self, sw[-2], sw[-1], target_s))

            # invoke cooperative subclass method 'store'
            di = self._store_if_reached(i=i, k=-1)

            # update diagnostic info
            if self.getinfo:
                self.info['computed_steps'] += 1
                self.info['stored_steps'] += di

            # prepare next iteration
            current_s = sw[-1]
            sw[...] = np.roll(sw, -1)
            xw[...] = np.roll(xw, -1)
            i += di

        # invoke cooperative subclass method 'end'
        self.end()

        # clean up temporary variables
        del self.itervars

    def __call__(self, timeline):
        """
        Run the simulation along the given timeline.

        Parameters
        ----------
        timeline : array-like
            A one dimensional array of strictly increasing numbers,
            defining the timeline of the simulation.

        Returns
        -------
        Simulation results, as specified by subclass methods.
        """
        paths, xshape, wshape = self.paths, self.xshape, self.wshape
        dtype, i0 = self.dtype, self.i0
        K = self.depth

        # set dtype default
        if dtype is None:
            dtype = float

        # enforce depth >= 2
        if K < 2:
            raise ValueError(
                'the depth of the integrator algorithm should be '
                '>= 2, not {}'.format(K))

        # preprocess and validate the result's timeline
        # ---------------------------------------------
        tt = np.asarray(timeline)
        if tt.shape != (tt.size,):
            raise ValueError(
                'the integration timeline should be a one-dimensional array, '
                'not an array of shape {}'.format(tt.shape))
        if not np.array_equal(tt, np.unique(tt)):
            raise ValueError(
                'the integration timeline sholud be '
                'an array of strictly increasing numbers, '
                'but {} was given'
                .format(tt))
        try:
            if not np.isscalar(tt[i0]):
                raise IndexError()
        except IndexError:
            raise IndexError(
                'i0 should be an integer indexing an element of the '
                'integration timeline, but {} was given'
                .format(i0))
        # NOTE:
        #   if tt is an integer type, the use of floats is forced in
        #   computations. the integer tt is finally passed to self.exit
        #   to be set as the timeline of the returned process
        ttype = float if tt.dtype.kind == 'i' else tt.dtype
        tt, tt_asis = tt.astype(ttype), tt

        # built the integration steps
        # ---------------------------
        tmin, tmax = tt[0], tt[-1]
        target_tt = self.pace(tt)
        target_tt = target_tt[np.logical_and(target_tt >= tmin,
                                             target_tt <= tmax)]
        steps_tt = np.unique(np.concatenate((target_tt, tt)))
        if len(steps_tt) < K-1:
            raise ValueError(
                'at least {} time points are needed '
                'for a paths_generator with depth {}'
                .format(K-1, K)
            )
        # steps_tt consolidates all target integration points and all points
        # in the timeline to be returned

        # memory allocation
        # -----------------
        xx = np.full(tt.shape + xshape + (paths,), np.nan,
                     dtype=dtype)
        xw = np.empty(K, dtype=object)
        sw = np.full(K, np.nan, dtype=ttype)
        for k in range(K):
            xw[k] = np.full(wshape + (paths,), np.nan, dtype=dtype)

        # store/initialize info
        # ---------------------
        if self.getinfo:
            self.info.update(
                t0=tt[i0], tmin=tt[0], tmax=tt[-1],
                computed_steps=0, stored_steps=0
            )

        # MAIN LOOP
        # ---------
        # get integration timeline and locate tt[i0] into it
        j0 = np.where(steps_tt == tt[i0])
        assert len(j0) == 1 and j0[0].shape == (1,)
        j0 = j0[0][0]

        # integrate backwards if not starting from the first point
        if i0 != 0:
            self._generate_paths(steps_tt[j0::-1], tt[i0::-1], xx[i0::-1],
                                 sw, xw, reverse=True)
        # integrate forwards if not starting from last point
        # (if tt.size==1, integrates only forwards)
        if i0 == 0 or i0 not in (-1, tt.size - 1):
            self._generate_paths(steps_tt[j0:], tt[i0:], xx[i0:],
                                 sw, xw, reverse=False)

        # Postprocess and exit
        # --------------------
        return self.exit(tt_asis, xx)


############################################
#  SDE integration frameword:
#  the integraor and SDE cooperating classes
############################################

# -------------------------
# The integrator class
# -------------------------

class integrator(paths_generator):
    """
    Step by step numerical integration of Ito Stochastic Differential
    Equations (SDEs), intended for subclassing.

    For usage, see the ``SDE`` class documentation.

    This class encapsulates SDE integration methods, and cooperates
    with the ``SDE`` class, that should always have precedence in
    method resolution order. As long as the respective
    APIs are complied with, a new integrator stated as an
    ``integrator`` subclass will interoperate with existing
    SDEs (as described by ``SDE`` subclasses), and a new SDE
    will interoperate with existing integrators.

    Parameters
    ----------
    paths, xshape, wshape, dtype, steps, i0, info, getinfo
        See ``paths_generator`` class documentation.
    method : string
        Integration method. Defaults to ``'euler'``, for the
        Euler-Maruyama method (at present, this single method
        is supported). It is stored as an attribute of the same name.

    Returns
    -------
    process
        Once instantiated as ``p``, ``p(timeline)`` performs the integration
        along the given timeline, based on parameters of instantiation,
        returning the resulting process as determined by the cooperating
        ``SDE`` subclass and the chosen integration method.
        Defaults to a process of ``numpy.nan`` along the given timeline.

    See Also
    --------
    paths_generator
    SDE
    SDEs

    Notes
    -----
    The equation to be integrated is exposed to the integration
    algorithm in a standardized form, via methods ``A`` and ``dZ``
    delegated to a cooperating ``SDE`` class. The latter should take care
    of equation parameters, initial conditions, expected paths and shapes,
    and should instantiate all necessary stochasticity sources.

    The integration method is exposed as the ``next`` method to the
    ``paths_generator`` parent class.

    If the ``getinfo`` attribute is set to ``True``, at each integration
    step the following items are added to the ``itervars`` dictionary,
    made available to subclasses to track the integration progress:

        *  ``last_t``: starting time point of the last integration step.
        *  ``last_dt``: time increment of the last integration step.
        *  ``last_x`` : starting value of the process, at time ``last_t``.
        *  ``last_A``: dictionary of the last computed values of the SDE
           terms, at time ``last_t``.
        *  ``last_dZ``: dictionary of the last realized SDE stochasticity
           source values, cumulated in the interval from ``last_t``
           to  ``last_t + last_dt``.
        *  ``new_x`` : computed value of the process, at time
           ``last_t + last_dt``.

    This becomes relevant in case the output timeline is coarse
    (e.g. just the initial and final time) but diagnostic information
    is needed about all integration steps performed
    (e.g., to track how often the process has changed sign, or to
    count the number of realized jumps).

    Methods
    -------
    A
    dZ
    next
    euler_next
    """

    def _check_integration_method(self, id):
        if not hasattr(self, id + '_next'):
            raise ValueError(
                'unrecognized integration method {}: '
                "use 'euler' "
                'or provide a properly defined '
                '`{}_next` integrator class method'
                .format(id, id))

    def _get_integration_method(self, id):
        return getattr(self, id + '_next')

    def __init__(self, *, paths=1,
                 xshape=(), wshape=(),
                 dtype=None, steps=None, i0=0,
                 info=None, getinfo=True,
                 method='euler'):

        # setup the required integration method
        self.method = method
        self._check_integration_method(method)
        self._method_next = self._get_integration_method(method)

        # set up the paths_generator parent class
        super().__init__(paths=paths,
                         xshape=xshape, wshape=wshape,
                         dtype=dtype, steps=steps, i0=i0,
                         info=info, getinfo=getinfo)

    # integration methods
    # -------------------

    def euler_next(self):
        """
        Euler-Maruyama integration step.
        """
        iv = self.itervars
        sw, xw = iv['sw'], iv['xw']

        s, ds = sw[0], sw[1] - sw[0]
        x = xw[0]
        # compute A, dZ and make them available as attributes
        A, dZ = self.A(s, x), self.dZ(s, ds)
        xw[1][...] = x + sum(A.get(id, 0)*dZ[id] for id in A.keys())

        if self.getinfo:
            iv.update(last_t=s, last_dt=ds,
                      last_x=xw[0], last_A=A,
                      last_dZ=dZ, new_x=xw[1])

    # interface vs parent paths_generator class, and
    # delegation to cooperating class methods
    # ----------------------------------------------

    depth = 2

    def next(self):
        """Perform an integration step with the requested method."""
        super().next()
        self._method_next()

    def exit(self, tt, xx):
        """See documentation of paths_generator.exit"""
        return process(t=tt, x=xx)

    # interface vs cooperating SDE class
    # ----------------------------------

    def A(self, t, x):
        """Value of the SDE terms at time t and process value x.

        Example of expected code for the SDE ``dx = (1 - x)*dt + 2*dw(t)``::

           return {
               'dt': (1 - x),
               'dw': 2
               }

        The ``SDE`` class takes care of casting user-specified
        equations into this format.
        """
        return {'dt': x + np.nan}

    def dZ(self, t, dt):
        """Value of the SDE differentials at time t, for
        time increment dt.

        Example of expected code for the SDE ``dx = (1 - x)*dt + 2*dw(t)``,
        where ``x`` has two components::

            shape = (2, self.paths)
            return {
                'dt': dt,
                'dw': wiener_source(vshape=2, paths=self.paths)(0, dt)
                }

        The ``SDE`` class takes care of instantiating user-specified
        stochasticity sources and casting them into this format.
        """
        return {'dt': dt + np.nan}


# ------------------------------------------------------------
# The SDE (one equation) and SDEs (multiple equations) classes
# ------------------------------------------------------------

class SDE:
    """
    Class representation of a user defined Stochastic Differential
    Equation (SDE), intended for subclassing.

    This class aims to provide an easy to use and flexible interface,
    allowing to specify user-defined SDEs and expose them in a standardized
    form to the cooperating ``integrator`` class (the latter should
    always follow in method resolution order). A minimal
    definition of an Ornstein-Uhlenbeck process is as follows:

        >>> from sdepy import SDE, integrator
        >>> class my_process(SDE, integrator):
        ...     def sde(self, t, x, theta=1., k=1., sigma=1.):
        ...         return {'dt': k*(theta - x), 'dw': sigma}

    An SDE is stated as a dictionary, containing for each differential
    the value of the corresponding coefficient::

        dx = f(t, x)*dt + g(t, x)*dw + h(t, x)*dj

    translates to::

        {'dt': f(t, x), 'dw': g(t, x), 'dj': h(t, x)}

    Instances are callables with signature ``(timeline)`` that integrate
    the SDE along the given timeline, using the configuration set out in
    the instantiation parameters:

        >>> P = my_process(x0=1, sigma=0.5, paths=100*1000, steps=100)
        >>> x = P(timeline=(0., 0.5, 1.))
        >>> x.shape
        (3, 100000)

    Subclasses can specify or customize:
    the equation and its parameters (``sde`` method),
    initial conditions and preprocessing (``init`` method and
    ``log`` attribute), shape of the values to be computed and stored
    (``shapes`` method), stochastic differentials appearing in the equation
    (``sources`` attribute) and their parameters and initialization (methods
    ``source_dt``, ``source_dw``, ``source_dn``, ``source_dj``, or any custom
    ``source_{id}`` method for a corresponding differential ``'{id}'``
    declared in ``sources`` and used as a key in ``sde`` return values),
    optional non array-like parameters (``more`` method), how to store results
    at points on the requested timeline (``let`` method), and
    postprocessing (``result`` method and ``log`` attribute).

    Parameters
    ----------
    paths : int
        Number of paths of the process.
    vshape : int or tuple of int
        Shape of the values of the process.
    dtype : data-type, optional
        Data-type of the process. Defaults to the numpy default.
    steps : iterable, or int, or None
        Specification of the time points to be touched during integration
        (as accepted by a cooperating ``integrator`` class).
        Default behaviour is:
          -  if ``None``, the simulated steps coincide with the timeline;
          -  if ``int``, the simulated steps touch all timeline points,
             as well as ``steps`` equally spaced points between the minimum
             and maximum point in the timeline;
          -  if iterable, the simulated steps touch all timeline points,
             as well as all values in ``steps`` between the minimum and maximum
             points in the timeline.
    i0 : int
        Index along the timeline at which the integration starts. The timeline
        is assumed to be in ascending order. Initial conditions are set
        at ``timeline[i0]``, the integration is performed
        backwards from ``timeline[i0]`` to ``timeline[0]``, and forwards
        from ``timeline[i0]`` to ``timeline[-1]``.
    info : dict, optional
        Diagnostic information about the integration is stored in this
        dictionary and is accessible as the ``info`` attribute.
        Defaults to a new empty ``dict``.
    getinfo : bool
        If ``True``, subclass methods ``info_begin``, ``info_next``,
        ``info_store``, ``info_end`` are invoked during integration.
        Defaults to ``True``.
    method : str
        Integration method, as accepted by the ``integrator``
        cooperating class.
    **args : SDE-specific parameters
        SDE parameters and initial conditions, as implied by the signature
        of ``sde``, ``init`` and ``more`` methods, and stochasticity sources
        parameters, as implied by the signature of ``source_{id}`` methods.
        Each keyword should be used once (e.g. ``corr``, a ``source_dw``
        parameter, cannot be used as the name of a SDE parameter).

    Returns
    -------
    process
        Once instantiated as ``p``, ``p(timeline)`` performs the integration
        along the given timeline, based on parameters of instantiation,
        and returns the resulting process as defined by subclass methods.
        Defaults to a process of ``numpy.nan`` along the given timeline.

    See Also
    --------
    paths_generator
    integrator
    SDEs

    Notes
    -----
    Custom stochastic differentials used in the SDE should be recognized,
    and treated appropriately, by the chosen integration method. This
    may require customization of the ``next`` method of the ``integrator``
    class.

    All named initialization parameters (``paths``, ``steps`` etc.)
    are stored as attributes.

    Notes on SDE-specific parameters:
        * ``init`` parameters are converted to arrays via ``np.asarray``.
        * ``sde`` and source quantitative parameters may be array-like,
          or time dependent with signature ``(t)``.
        * both are converted to arrays via ``np.asarray``, and
          for both, their constant value, or values at each time point,
          should be broadcastable to a shape ``wshape + (paths,)``.
        * ``more`` parameters undergo no further initialization, before
          being made available to the ``shapes`` and ``more`` methods.

    If ``getinfo`` is ``True``, the invoked info subclass methods
    may initialize and cumulate diagnostic information in items
    of the ``info`` dictionary, based on read-only access of the
    internal variables set during integration by  ``paths_generator``
    and ``integrator`` cooperating classes, as exposed in the ``itervars``
    attribute.

    Attributes
    ----------
    sources : set or dict
        As a class attribute, holds the names of the differentials ``'dz'``
        expected to appear in the equation. As an instance attribute,
        ``sources['dz']`` is an object, complying with the ``source`` protocol,
        that instantiates the differential ``'dz'`` used during integration.
        ``sources['dz'](t, dt)`` is computed at every step for each
        ``'dz' in sources``, as required by the chosen integration method.
    args : dict
        Stores parameters passed as ``**args`` upon initialization
        of the SDE. Should be used by subclass methods to access
        and modify their values.
    log : bool
        If True, the natural logarithm of the initial values set by the
        ``init`` method is taken as the initial value of the integration,
        and the result of the integration is exponentiated back before
        serving it to the ``result`` method. The ``sde`` should expose
        the appropriate equation for integrating the logarithm of the
        intended process.

    Methods
    -------
    sde
    shapes
    source_dt
    source_dw
    source_dn
    source_dj
    more
    init
    let
    result
    info_begin
    info_next
    info_store
    info_end
    """

    # public attributes
    # -----------------

    sources = {'dt', 'dw'}
    log = False
    q = None
    addaxis = None
    # q and addaxis are not used in this class
    # (inserted for consistency with subclass SDEs)

    # private methods and attributes
    # ------------------------------

    def _check_source_id(self, id):
        if not hasattr(self, 'source_' + id):
            raise ValueError(
                'unrecognized source {}: '
                "use one of 'dt', 'dw', 'dn', 'dj', "
                'or provide a properly defined '
                'SDE class method `source_{}`'
                .format(id, id))

    def _get_source_setup_method(self, id):
        return getattr(self, 'source_' + id)

    def _get_args(self, keys):
        return {k: z for k, z in self._args.items()
                if k in keys}

    def _inspect_args_defaults(self, sde_nvars=1):

        # inspect signatures to find expected args and defaults,
        # apply user-defined defaults if any
        expected_source_args = {}
        for id in self.sources:
            self._check_source_id(id)
            source_setup_method = self._get_source_setup_method(id)
            expected_source_args[id] = \
                dict(_signature(source_setup_method))
        # first two parameters of sde.init are expected to be t and out_x
        expected_init_args = dict(_signature(self.init)[2:])
        # first two parameters of sde callable are expected to be t and x,
        # (for SDEs, first 1+q parameters are t, x1, x2, ... xq)
        expected_sde_args = dict(_signature(self.sde)[1+sde_nvars:])

        expected_more_args = dict(_signature(self.more))

        # consolidate
        expected_args = {}
        for D in (tuple(expected_source_args.values()) +
                  (expected_init_args, expected_sde_args,
                   expected_more_args)):
            expected_args.update(D)
        # check for multiple occurrencies of the same expected arg,
        # with different defaults
        defaults = {k: [] for k in expected_args}
        for D in (tuple(expected_source_args.values()) +
                  (expected_init_args, expected_sde_args,
                   expected_more_args)):
            for k, z in D.items():
                defaults[k].append(z)
        repeated = {k for k in defaults
                    if len(set(defaults[k])) > 1}
        if repeated:
            raise TypeError(
                    'two or more incompatible defaults found '
                    'for SDE parameter(s) {}'
                    .format(repeated))

        # store arg keys for later use in self._get_args()
        self._source_args_keys = {}
        for id in self.sources:
            self._source_args_keys[id] = set(expected_source_args[id])
        self._init_args_keys = set(expected_init_args)
        self._sde_args_keys = set(expected_sde_args)
        self._more_args_keys = set(expected_more_args)

        # store args defaults for later use by _consolidate_args
        self._expected_args = expected_args

    def _consolidate_args(self, **args):
        expected_args = self._expected_args

        # consolidate given args with expected defaults and check
        all_args = {**expected_args, **args}
        unexpected = set(all_args).difference(expected_args)
        missing = {k for (k, v) in all_args.items()
                   if v is _empty}
        if unexpected:
            raise TypeError(
                'unexpected keyword(s): {}'
                .format(unexpected))
        if missing:
            raise TypeError(
                'no value and no default found for sde parameter(s) {}'
                .format(missing))

        # return a unique dict with all args
        return all_args

    def _sde_args_setup(self, sde_args):
        # convert to array all sde args, preserving optional
        # time dependence
        new_sde_args = {k: _variable_param_setup(z)
                        for (k, z) in sde_args.items()}
        return new_sde_args

    def _init_args_setup(self, init_args):
        # initialize and convert to array all init args
        new_init_args = {k: _const_param_setup(z)
                         for (k, z) in init_args.items()}
        return new_init_args

    def _sources_setup(self):
        # initialize stochasticity sources
        sources = {id:
                   self._get_source_setup_method(id)(
                       **self._get_args(self._source_args_keys[id]))
                   for id in self.sources}
        return sources

    # initialization
    # --------------

    def __init__(self, *, paths=1,
                 vshape=(),
                 dtype=None, steps=None, i0=0,
                 info=None, getinfo=True,
                 method='euler',
                 **args):

        if not isinstance(self, integrator):
            raise TypeError(
                'cannot instantiate SDE subclass {} that is not a subclass '
                'of a cooperating integrator class'
                .format(type(self)))
        elif hasattr(self, 'method'):
            raise TypeError(
                'improper method resolution order in class {}: '
                'the integrator class cannot precede the the SDE class'
                .format(type(self)))

        # self.vshape is used by SDEs subclass
        # to force self.addaxis in case vshape == ()
        self.vshape = _shape_setup(vshape)

        # get default args by inspection
        # (stored as private attributes)
        self._inspect_args_defaults()

        # consolidate given args with defaults
        self._args = self._consolidate_args(**args)

        # preprocess sde and init args
        init_args = self._init_args_setup(self._get_args(self._init_args_keys))
        self._args.update(init_args)
        sde_args = self._sde_args_setup(self._get_args(self._sde_args_keys))
        self._args.update(sde_args)

        # compute shapes
        (self.vshape,
         self.xshape,
         self.wshape) = self.shapes(self.vshape)

        # set paths and dtype
        self.paths = paths
        self.dtype = dtype

        # setup stochasticity sources
        # (this uses paths, shapes and dtype attributes)
        self.sources = self._sources_setup()
        self._ordered_source_ids = sorted(list(self.sources))

        # optional further initializations delegated to subclasses
        # (may include read or write access to _args items
        # via the 'args' property)
        self.more(**self._get_args(self._more_args_keys))

        # further initialization by a superclass
        # (should be an integrator)
        super().__init__(paths=self.paths,
                         xshape=self.xshape, wshape=self.wshape,
                         dtype=self.dtype, steps=steps, i0=i0,
                         info=info, getinfo=getinfo,
                         method=method)

    # public interface vs the integrator and
    # paths_generator classes
    # --------------------------------------

    def begin(self):
        """See documentation of paths_generator.begin"""
        super().begin()
        iv = self.itervars
        t0 = iv['sw'][0]
        out_x = iv['xw'][0]
        assert t0 == iv['steps_tt'][0] == iv['tt'][0]
        init_args = self._get_args(self._init_args_keys)
        self.init(t0, out_x, **init_args)
        if self.log:
            log(out_x, out=out_x)

        if self.getinfo:
            self.info_begin()

    def next(self):
        """See documentation of paths_generator.next"""
        super().next()
        if self.getinfo:
            self.info_next()

    def store(self, i, k):
        """See documentation of paths_generator.store"""
        super().store(i, k)
        iv = self.itervars
        t = iv['sw'][k]
        x = iv['xw'][k]
        out_x = iv['xx'][i]
        assert iv['tt'][i] == t
        self.let(t, out_x, x)
        if self.getinfo:
            self.info_store()

    def end(self):
        """See documentation of paths_generator.end"""
        super().end()
        if self.getinfo:
            self.info_end()

    def exit(self, tt, xx):
        """See documentation paths_generator.exit"""
        if self.log:
            exp(xx, out=xx)
        return self.result(tt, xx)

    # public interface vs the integration
    # algorithm of the integrator class
    # -----------------------------------

    def _check_sde_values(self, A):
        if not isinstance(A, dict):
            raise TypeError(
                'invalid {} return values: a dict, not a {} object expected'
                .format(self.sde, type(A))
                )
        if not set(A.keys()).issubset(self.sources):
            raise KeyError(
                'invalid {} return values: {} entries expected (one per '
                'stochasticity source), not {}'
                .format(self.sde, set(self.sources), set(A.keys()))
                )

    def A(self, t, x):
        """See documentation integrator.A"""
        sde_args_eval = {
            k: (z(t) if callable(z) else z)
            for (k, z) in self._get_args(self._sde_args_keys).items()
        }
        # get and check sde values
        A_ = self.sde(t, x, **sde_args_eval)
        self._check_sde_values(A_)
        return A_

    def dZ(self, t, dt):
        """See documentation of integrator.dZ"""
        sources = self.sources
        # sources are evaluated in order of ascending id
        # (safeguards reproduciblility of outcomes after
        # seeding numpy.random)
        return {id: sources[id](t, dt)
                for id in self._ordered_source_ids}

    # public interface vs subclasses
    # ------------------------------

    @property
    def args(self):
        """
        Stores parameters passed as ``**args`` upon initialization
        of the SDE. Should be used by subclass methods to access
        and modify their values.
        """
        return self._args

    def shapes(self, vshape):
        """
        Shape of the values to be computed and stored
        upon integration of the SDE.

        Parameters
        ----------
        vshape : int or tuple of int
            Shape of the values of the integration result,
            as requested upon instantiation of ``SDE``.

        Returns
        -------
        vshape : int or tuple of int
            Confirms or overrides the given ``vshape``.
        xshape : int or tuple of int
            Shape of the values stored during integration
            at the output time points. ``out_x`` array
            passed to the ``let`` method has shape
            ``xshape + (paths,)``. Defaults to ``vshape``.
        wshape : int or tuple of int
            Shape of the working space used during integration.
            ``x`` values passed to the ``sde`` and ``let`` methods
            have shape ``wshape + (paths,)``. Defaults to ``vshape``.

        Notes
        -----
        ``xshape`` and ``wshape`` are passed to the
        parent ``paths_generator`` class.

        ``hull_white_SDE`` and ``heston_SDE`` classes illustrate
        use cases for different values of ``vshape``, ``xshape``
        and/or ``wshape``.
        """
        xshape = wshape = vshape
        return vshape, xshape, wshape

    def source_dt(self):
        """
        Setup a source of deterministic increments, to be used
        as 'dt' during integration.

        Returns
        -------
        An object ``z`` complying with the ``source`` protocol,
        such that ``z(t, dt) == dt``.
        """
        def dt(s, ds):
            return ds

        dt.paths = self.paths
        dt.vshape = self.wshape
        return dt

    def source_dw(self,
                  dw=None, corr=None, rho=None):
        """Setup a source of standard Wiener process (Brownian motion)
        increments, to be used as 'dw' during integration.

        Parameters
        ----------
        dw : source, or source subclass, or None
            If an object complying with the ``source`` protocol,
            it is returned (``corr`` and ``rho`` are ignored).
            If a source subclass, it is instantiated with the
            given parameters, and returned. If None, a
            new instance of ``wiener_source`` is returned, with
            the given parameters.
        corr, rho : see ``wiener_source`` documentation

        Returns
        -------
        An object complying with the ``source`` protocol,
        instantiating the requested stochasticity source.
        The shape of source values is set to ``wshape``.

        See Also
        --------
        wiener_source
        """
        return _source_setup(dw, wiener_source,
                             paths=self.paths, vshape=self.wshape,
                             dtype=self.dtype,
                             corr=corr, rho=rho)

    def source_dn(self,
                  dn=None, ptype=int, lam=1.):
        """
        Setup a source of Poisson process increments,
        to be used as 'dn' during integration.

        Parameters
        ----------
        dn : source, or source subclass, or None
            If an object complying with the ``source`` protocol,
            it is returned (``ptype`` and ``lam`` are ignored).
            If a source subclass, it is instantiated with the
            given parameters, and returned. If None, a
            new instance of ``poisson_source`` is returned, with
            the given parameters.
        ptype, lam : see ``poisson_source`` documentation

        Returns
        -------
        An object complying with the ``source`` protocol,
        instantiating the requested stochasticity source.
        The shape of source values is set to ``wshape``.

        See Also
        --------
        poisson_source
        """
        return _source_setup(dn, poisson_source,
                             paths=self.paths, vshape=self.wshape,
                             # dtype of the poisson source is ptype
                             dtype=ptype,
                             lam=lam)

    def source_dj(self,
                  dj=None,
                  dn=None, ptype=int, lam=1.,
                  y=None):
        """
        Set up a source of compound Poisson process increments
        (jumps), to be used as 'dj' during integration.

        Parameters
        ----------
        dj : source, or source subclass, or None
            If an object complying with the ``source`` protocol,
            it is returned (``ptype``, ``lam`` and ``y`` are ignored).
            If a source subclass, it is instantiated with the
            given parameters, and returned. If None, a
            new instance of ``cpoisson_source`` is returned, with
            the given parameters.
        ptype, lam, y : see ``cpoisson_source`` documentation

        Returns
        -------
        An object complying with the ``source`` protocol,
        instantiating the requested stochasticity source.
        The shape of source values is set to ``wshape``.

        See Also
        --------
        cpoisson_source
        """
        return _source_setup(dj, cpoisson_source,
                             paths=self.paths, vshape=self.wshape,
                             dtype=self.dtype,  # dtype of the source
                             dn=dn,
                             ptype=ptype,  # dtype of underlying poisson source
                             lam=lam,
                             y=y)

    def more(self):
        """
        Further optional non array parameters,
        and initializations.

        Parameters
        ----------
        more_args : zero or more keyword arguments
            Further, possibly non array-like, SDE parameters, as implied
            by the ``more`` method signature. Passed upon instantiation
            of the ``SDE`` class, are served to the ``more`` method
            and made available to other methods as items in the ``args``
            attribute.

        Notes
        -----
        The ``factors`` parameter of ``hull_white_SDE`` illustrates
        a use case for the ``more`` method.
        """
        pass

    def init(self, t, out_x, x0=1.):
        """
        Set initial conditions for SDE integration.

        Parameters
        ----------
        t : float
            Time point at which initial conditions should be
            imposed.
        out_x : array
            Array, shaped ``wshape + (paths,)``, where initial
            conditions are to be stored.
        init_args : zero or more arrays, as keyword arguments
            Initialization parameters, as implied by the ``init`` method
            signature. Passed upon instantiation of the ``SDE`` class
            as array-like, these parameters are served to the ``init`` method
            converted to arrays via ``np.asarray``.

        Notes
        -----
        The default implementation has a single ``x0`` parameter,
        and sets ``out_x[...] = x0``.
        """
        out_x[...] = x0

    def sde(self, t, x):
        """
        Stochastic Differential Equation (SDE) to be integrated.

        Parameters
        ----------
        t : float
            Time point at which the SDE should be evaluated.
        x : array
            Values that the stochastic process takes at time ``t``.
        sde_args : zero or more arrays, as keyword arguments
            SDE parameters, as implied by the
            ``sde`` method signature. Passed upon
            instantiation of the ``SDE`` class as possibly time-dependent
            array-like, these parameters are served to the ``sde`` method
            once evaluated at ``t`` and converted to arrays via ``np.asarray``.

        Returns
        -------
        sde_terms : dict of arrays
            Contains, for each differential stated in the ``source``
            attribute, the value of the corresponding coefficient
            in the represented SDE.

        Notes
        -----
        ``x`` should be treated as read-only.
        """
        return {'dt': x + np.nan}

    def let(self, t, out_x, x):
        """
        Store the value of the integrated process at
        time point ``t`` belonging to the requested output timeline.

        Parameters
        ----------
        t : float
            Time point to which the integration result ``x`` refers.
        out_x : array
            Array, shaped ``xshape + (paths,)``, where the result
            ``x`` is to be stored.
        x : array
            Integration result at time ``t``, shaped
            ``wshape + (paths,)``

        Notes
        -----
        The default implementation sets ``out_x[...] = x``.

        In case ``xshape != wshape``, this method should operate as needed
        in order to store in ``out_x`` a value broadcastable to its shape
        (e.g. it might store in ``out_x`` only some of the components of
        ``x``).

        ``x`` should be treated as read-only.
        """
        out_x[...] = x

    def result(self, tt, xx):
        """
        Compute the integration output.

        Parameters
        ----------
        tt : array
            Output integration timeline.
        xx : array
            Integration result, shaped ``tt.shape + xshape + (paths,)``.

        Returns
        -------
        result
            Final result, returned to the user.

        Notes
        -----
        The default implementation returns ``sdepy.process(t=tt, x=xx)``.
        In case ``vshape != xshape``, this method should operate as needed
        in order to return a process with values shaped as ``vshape``
        (e.g. it might return a function of the components of ``xx``).
        """
        return process(t=tt, x=xx)

    def info_begin(self):
        """
        Optional diagnostic information logging function,
        called before the integration begins.
        """
        pass

    def info_next(self):
        """
        Optional diagnostic information logging function,
        called after each integration step.
        """
        pass

    def info_store(self):
        """
        Optional diagnostic information logging function,
        called after each invocation of the let method.
        """
        pass

    def info_end(self):
        """
        Optional diagnostic information logging function,
        called after the integration has been completed.
        """
        pass


class SDEs(SDE):
    """
    Class representation of a user defined system of Stochastic Differential
    Equations (SDEs), intended for subclassing.

    The parent ``SDE`` class represents a single SDE, scalar
    or multidimensional: by an appropriate choice of the ``vshape`` parameter,
    and composition of equation values, it suffices to describe
    any system of SDEs.

    Its ``SDEs`` subclass is added for convenience of representation: it allows
    to state each equation separately and to retrieve separate processes
    as a result. The number of equations must be stated as the ``q`` attribute.
    The ``vshape`` parameter is taken as the common shape of values in each
    equation in the system.

    A minimal definition of a lognormal process ``x`` with stochastic
    volatility ``y`` is as follows::

        >>> from sdepy import SDEs, integrator
        >>> class my_process(SDEs, integrator):
        ...     q = 2
        ...     def sde(self, t, x, y, mu=0., sigma=1., xi=1.):
        ...         return ({'dt': mu*x, 'dw': y*x},
        ...                 {'dt': 0,    'dw': xi*y})

        >>> P = my_process(x0=(1., 2.), xi=0.5, vshape=5,
        ...                paths=100*1000, steps=100, )
        >>> x, y = P(timeline=(0., 0.5, 1.))
        >>> x.shape, y.shape
        ((3, 5, 100000), (3, 5, 100000))

    See Also
    --------
    SDE
    integrator
    paths_generator

    Notes
    -----
    By default, the stochasticity sources of each component equation
    are realized independently, even if represented in the ``sde`` output
    by the same key (``'dw'`` in the example above).

    The way stochasticity sources are instantiated and dispatched to
    each equation, and how correlations of the Wiener source are set
    via the ``corr`` parameter, depend on the value of the ``addaxis``
    attribute:
        * If ``True``, source values have shape ``vshape + (q,)``,
          and the ``[kk, i]`` component of source values
          is dispatched to the ``kk`` component of equation ``i``
          (``kk`` is a multiindex spanning shape ``vshape``).
          If given, ``corr`` must be of shape ``(q, q)`` and correlates
          corresponding components across equations.
        * If ``addaxis`` is ``False`` (default) and ``N`` is the size
          of the last axis of ``vshape``, the values of the sources have shape
          ``vshape[:-1] + (N*q,)``, and the ``[kk, i*N + h]`` component
          of the source values is dispatched to the ``[kk, h]`` component
          of equation ``i`` (``kk`` is a multiindex spanning
          shape ``vshape[:-1]``, and ``h`` is in ``range(N)``).
          If given, ``corr`` must be of shape ``(N*q, N*q)``, and correlates
          all last components of all equations to each other.

    After instantiation, stochasticity sources and correlation matrices
    may be inspected as follows::

        >>> P = my_process(vshape=(), rho=0.5)
        >>> P.sources['dw'].vshape
        (2,)
        >>> P.sources['dw'].corr.shape
        (2, 2)
        >>> P.sources['dw'].corr[0, 1]
        0.5

    Attributes
    ----------
    q : int
        Number of equations.
    addaxis : bool
        Affects the internal representation of the equations: if ``True``,
        a last axis of size ``q`` is added to ``vshape``, if ``False``,
        components are stacked onto the last axis of ``vshape``.
        Defaults to ``False``. It is forced to ``True`` if the process
        components have scalar values.

    Methods
    -------
    pack
    unpack
    """

    # public attributes
    # -----------------
    q = 1
    addaxis = False

    # private methods and attributes
    # ------------------------------

    def _inspect_args_defaults(self):
        if self.q < 1:
            raise ValueError(
                'the number of equations q should be positive, but '
                '{} was given'.format(self.q))
        if self.vshape == ():
            # force adding an axis for equations
            self.addaxis = True
        super()._inspect_args_defaults(sde_nvars=self.q)

    # public interface vs the integration
    # algorithm of the integrator class
    # -----------------------------------

    def _check_sde_values(self, As):
        if not isinstance(As, (list, tuple)):
            raise TypeError(
                'invalid {} return values: a list or tuple of {} dict '
                '(one per equation) expected, not a {} object'
                .format(self.sde, self.q, type(As))
                )
        if len(As) != self.q:
            raise ValueError(
                'invalid {} return values: {} equations expected, '
                'not {}'.format(self.sde, self.q, len(As)))
        for a in As:
            super()._check_sde_values(a)

    def A(self, t, X):
        """See documentation of integrator.A"""
        sde_args_eval = {
            k: (z(t) if callable(z) else z)
            for (k, z) in self._get_args(self._sde_args_keys).items()
        }

        # unpack X, as feeded by integrator.next, in a list
        # of arrays (one per equation)
        xs = self.unpack(X)

        # get and check sde values
        As = self.sde(t, *xs, **sde_args_eval)
        self._check_sde_values(As)
        A_ids = set()
        for a in As:
            A_ids.update(a.keys())
        A = {id: self.pack(tuple(a.get(id, 0) for a in As))
             for id in A_ids}
        return A

    # public interface vs subclasses
    # ------------------------------

    def unpack(self, X):
        """Unpacks the given array into multiple arrays
        (one per equation).

        Parameters
        ----------
        X : array
            Array with a last dimension enumerating paths, and a second
            last dimension to be unpacked according to the ``addaxis``
            attribute setting.

        Returns
        -------
        x, y, ... : list of arrays
            List of ``self.q`` arrays, unpacking the given ``X``.
        """
        q = self.q
        if self.addaxis:
            xs = tuple(X[..., k, :] for k in range(q))
        else:
            d = self.vshape[-1]
            xs = tuple(X[..., k*d:(k+1)*d, :] for k in range(q))
        return xs

    def pack(self, xs):
        """Packs the given arrays (one per equation) into a single array.

        Parameters
        ----------
        xs : list of arrays
            List of ``self.q`` arrays to be packed according to the ``addaxis``
            attribute setting.

        Returns
        -------
        X : array
            Array packing the given ``xs`` along its second-last dimension
            (the last dimension enumerates paths).
        """
        target_shape = self.vshape + (self.paths,)
        if self.addaxis:
            i = np.index_exp[..., np.newaxis, :]
        else:
            i = np.index_exp[...]
        X = np.concatenate(tuple(
            np.broadcast_to(x, target_shape)[i]
            for x in xs
            ), axis=-2)
        return X

    def shapes(self, vshape):
        """See documentation of SDE.shapes"""
        q = self.q
        if self.addaxis:
            xshape = wshape = vshape + (q,)
        else:
            xshape = wshape = vshape[:-1] + (vshape[-1]*q,)
        return vshape, xshape, wshape

    def init(self, t, out_X, x0=1.):
        """See documentation of SDE.init"""
        x0s = x0
        # in init's calling signature, x0 is kept for compatibility with SDE;
        # should be a list or tuple or array of the q initial conditions
        if x0s.shape == ():  # scalar values are broadcasted to all equations
            x0s = (x0s,)*self.q
        X0 = self.pack(x0s)
        super().init(t, out_X, X0)

    def sde(self, t, x):
        """Stochastic Differential Equations (SDEs) to be integrated.

        Parameters
        ----------
        t : float
            Time point at which the SDE should be evaluated.
        x, y, ... : arrays
            Values that each equation variable takes at time ``t``.
            There should be as many parameters as the number of
            equations stated in ``self.q``.
        sde_args : zero or more arrays, as keyword arguments
            See documentation of ``SDE.sde``.

        Returns
        -------
        sde_terms : list or tuple of dict of arrays
            A list or tuple of dictionaries, one per equation.
            See documentation of ``SDE.sde``.

        Notes
        -----
        ``x, y, ...`` should be treated as read-only.
        """
        return ({'dt': x + np.nan},)

    def result(self, tt, XX):
        """See documentation of SDE.result"""
        if self.log:
            exp(XX, out=XX)
        xxs = self.unpack(XX)
        return tuple(process(t=tt, x=xx) for xx in xxs)


# -----------------------
# integration interface:
# the integrate decorator
# -----------------------

def _SDE_from_function(f, q=None, sources=None, log=False, addaxis=False):

    if q is not None and sources is not None:
        neq = q
        ids = set(sources)
        SDE_class = SDE if neq == 0 else SDEs
    else:
        # perform a test evaluation of f
        try:
            try:
                test_val = f()
            except Exception:
                test_val = f(np.array(1.), np.array(1.))
        except Exception:
            raise TypeError(
                'test evaluation of {} failed'
                .format(f))
        # infer neq, ids and SDE_class from test_val
        if isinstance(test_val, (tuple, list)):
            neq = len(test_val)
            if neq == 0:
                raise ValueError('non empty list or tuple expected')
            SDE_class = SDEs
        else:
            neq = 0
            SDE_class = SDE
            test_val = (test_val,)
        ids = set()
        for z in test_val:
            ids.update(z.keys())
        # consistency check
        if ((q is not None and neq != q) or
            (sources is not None and set(sources) != ids)):
            raise TypeError(
                'test evaluation of {} inconsistent with given '
                "'q' or 'sources'".format(f))

    # avoid namespace conflicts inside SDE_wrapper
    log_flag = log
    addaxis_flag = addaxis

    class SDE_wrapper(SDE_class):
        q = neq
        sources = ids
        log = log_flag
        addaxis = addaxis_flag
        sde = staticmethod(f)

    return SDE_wrapper


def integrate(sde=None, *, q=None, sources=None, log=False, addaxis=False):
    """Decorator for Ito Stochastic Differential Equation (SDE)
    integration.

    Decorates a function representing the SDE or SDEs into the corresponding
    ``sdepy`` integrator.

    Parameters
    ----------
    sde : function
        Function to be wrapped. Its signature and values should be
        as expected for the ``sde`` method of the ``sdepy.SDE`` or
        ``sdepy.SDEs`` classes.
    q : int
        Number of equations. If ``None``, attempts a test evaluation
        of ``sde`` to find out. ``q=0`` indicates a single equation.
    sources : set
        Stochasticity sources used in the equation. If ``None``,
        attempts a test evaluation of ``sde`` to find out.
    log : bool
        Sets the ``log`` attribute for the wrapping class.
    addaxis : bool
        Sets the ``addaxis`` attribute for the wrapping class.

    Returns
    -------
    A subclass of ``sdepy.SDE`` or ``sdepy.SDEs`` as appropriate,
    and of ``sdepy.integrator``, with the given ``sde``
    cast as its ``sde`` method.

    Notes
    -----
    To prevent a test evaluation of ``sde``, explicitly provide
    the intended ``q`` and ``sources`` as keyword arguments to ``integrate()``.
    The test evaluation is attempted as ``sde()`` and, upon failure,
    again as ``sde(1., 1.)``.

    Examples
    --------
        >>> from sdepy import integrate
        >>> @integrate
        ... def my_process(t, x, theta=1., k=1., sigma=1.):
        ...     return {'dt': k*(theta - x), 'dw': sigma}

        >>> P = my_process(x0=1, sigma=0.5, paths=100*1000, steps=100)
        >>> x = P(timeline=(0., 0.5, 1.))
        >>> x.shape
        (3, 100000)
    """
    if sde is None:
        def decorator(sde):
            return integrate(sde, q=q, sources=sources,
                             log=log, addaxis=addaxis)
        return decorator
    else:
        SDE_class = _SDE_from_function(sde, q=q, sources=sources,
                                       log=log, addaxis=addaxis)

        class sde_integrator(SDE_class, integrator):
            pass

        return sde_integrator


#############################################
#  Process generators withoud SDE integration
#############################################

# THIS CODE HAS BEEN REMOVED AND NOT KEPT UP TO DATE
# all the point was to gain speed, and as of april 2018
# the cumsum operation on which const_wiener_process and
# const_lognorm_process rely takes longer than step by step
# integration with lognorm_process and wiener_process
# (tested on 1000 time steps and 10000 paths)
#
# the code is frozen as a string in case one might
# consider re-inclusion in the future

'''
class const_wiener_process:
    """
    Wiener process (Brownian motion), with time-independent parameters.
    """

    def __init__(self, paths=1, vshape=(), dtype=None,
                 dw=None, corr=None, rho=None,
                 x0=0., mu=0., sigma=1.
                 ):
        self.paths = paths
        self.vshape = vshape = _shape_setup(vshape)
        self.xshape = self.wshape = vshape
        self.dtype = dtype

        args = {'x0': x0, 'mu': mu, 'sigma': sigma}
        args = {k: _const_param_setup(z) for (k, z) in args.items()}
        self._args = args

        # the setup of corr, rho is delegated to 'wiener'
        self.sources = {'dt': lambda t, dt: dt,
                        'dw': _dw_source_setup(
                            dw, paths, vshape, dtype,
                            corr, rho)}

    @property
    def params(self):
        return self._args.copy()


    def __call__(self, t):
        paths, vshape, dtype = self.paths, self.vshape, self.dtype

        # timeline and time increments setup
        t = np.asarray(t)
        if t.shape != (t.size,):
            raise ValueError(
                'the process timeline should be a '
                'one-dimensional array, not an array of shape {}'
                .format(t.shape))
        dt = np.concatenate((0*t[:1], np.diff(t)))  # dt[0] == 0
        shifted_t = np.concatenate((t[:1], t[:-1]))
        tx = t.reshape((-1,) + (1,)*len(vshape) + (1,)) \

        # initial condition and parameters setup
        x0, mu, sigma = [self._args[k]
                         for k in ('x0', 'mu', 'sigma')]

        # generate process
        dw = self.sources['dw'](shifted_t, dt)
        x = np.empty(t.shape + vshape + (paths,), dtype=dtype)
        x[...] = x0 + mu * (tx - tx[0]) + sigma * dw.cumsum(axis=0)
        return process(t, x=x)


class const_lognorm_process(const_wiener_process):
    """
    Lognormal process, with time-independent parameters.
    """

    def __init__(self, paths=1, vshape=(), dtype=None,
                 dw=None, corr=None, rho=None,
                 x0=1., mu=0., sigma=1.
                 ):
        super().__init__(paths=paths, vshape=vshape, dtype=dtype,
                         dw=dw, corr=corr, rho=rho,
                         x0=x0, mu=mu, sigma=sigma)

    def __call__(self, t):
        x0, mu, sigma = [self._args[k] for k in ('x0', 'mu', 'sigma')]

        wp = const_wiener_process(
            paths=self.paths, vshape=self.vshape, dtype=self.dtype,
            dw=self.sources['dw'],
            x0=log(x0), mu=mu - sigma*sigma/2, sigma=sigma)(t)
        exp(wp, out=wp)
        return wp
'''


##########################################
#  Process generators with SDE integration
##########################################

class wiener_SDE(SDE):
    """
    SDE for a Wiener process (Brownian motion) with drift.

    See Also
    --------
    wiener_process
    """

    # set x0 default value to 0.
    def init(self, s, out_x, x0=0.):
        super().init(s, out_x, x0)

    def sde(self, t, x, mu=0., sigma=1.):
        return {'dt': mu, 'dw': sigma}


class wiener_process(wiener_SDE, integrator):
    """
    wiener_process(paths=1, vshape=(), dtype=None, steps=None, i0=0,
    info=None, getinfo=True, method='euler',
    x0=0., mu=0., sigma=1., dw=None, corr=None, rho=None)

    Wiener process (Brownian motion) with drift.

    Generates a process ``x(t)`` that solves the following SDE::

        dx(t) = mu(t)*dt + sigma(t)*dw(t, dt)

    where ``dw(t, dt)`` are standard Wiener process increments with
    correlation matrix specified by ``corr(t)`` or ``rho(t)``.
    ``x0``, SDE parameters and ``dw(t, dt)`` should broadcast to
    ``vshape + (paths,)``.

    Parameters
    ----------
    paths, vshape, dtype, steps, i0, info, getinfo, method
        See ``SDE`` class documentation.
    x0 : array-like
        Initial condition.
    mu, sigma : array-like, or callable
        SDE parameters.
    dw, corr, rho
        Specification of stochasticity source of Wiener process increments.
        See ``SDE.source_dw`` documentation.

    Returns
    -------
    x : process
        Once instantiated as ``p``, ``p(timeline)`` performs the integration
        along the given timeline, based on parameters of instantiation,
        and returns the resulting process.

    See also
    --------
    SDE
    SDE.source_dw
    wiener_source
    wiener_SDE
    """
    pass


class lognorm_SDE(SDE):
    """
    SDE for a lognormal process with drift.

    See Also
    --------
    lognorm_process
    """
    log = True

    def sde(self, t, x, mu=0., sigma=1.):
        return {'dt': mu - sigma*sigma/2, 'dw': sigma}


class lognorm_process(lognorm_SDE, integrator):
    """
    lognorm_process(paths=1, vshape=(), dtype=None, steps=None, i0=0,
    info=None, getinfo=True, method='euler',
    x0=1., mu=0., sigma=1., dw=None, corr=None, rho=None)

    Lognormal process.

    Generates a process ``x(t)`` that solves the following SDE::

        dx(t) = mu(t)*x(t)*dt + sigma(t)*x(t)*dw(t, dt)

    where ``dw(t, dt)`` are standard Wiener process increments with
    correlation matrix specified by ``corr(t)`` or ``rho(t)``.
    ``x0``, SDE parameters and ``dw(t, dt)`` should broadcast to
    ``vshape + (paths,)``. ``x0`` should be positive.

    Parameters
    ----------
    paths, vshape, dtype, steps, i0, info, getinfo, method
        See ``SDE`` class documentation.
    x0 : array-like
        Initial condition.
    mu, sigma : array-like, or callable
        SDE parameters.
    dw, corr, rho
        Specification of stochasticity source of Wiener process increments.
        See ``SDE.source_dw`` documentation.

    Returns
    -------
    x : process
        Once instantiated as ``p``, ``p(timeline)`` performs the integration
        along the given timeline, based on parameters of instantiation,
        and returns the resulting process.

    See also
    --------
    SDE
    SDE.source_dw
    wiener_source
    wiener_SDE

    Notes
    -----
    ``x(t)`` is obtained via Euler-Maruyama numerical integration of the
    following equivalent SDE for ``a(t) = log(x(t))``::

        da(t) = (mu(t) - sigma(t)**2/2)*dt + sigma(t)*dw(t, dt)
    """


class ornstein_uhlenbeck_SDE(SDE):
    """
    SDE for an Ornstein-Uhlenbeck process.

    See Also
    --------
    ornstein_uhlenbeck_process
    """

    # set x0 default value to 0.
    def init(self, s, out_x, x0=0.):
        super().init(s, out_x, x0)

    def sde(self, s, x, theta=0., k=1., sigma=1.):
        return {'dt': k*(theta - x), 'dw': sigma}


class ornstein_uhlenbeck_process(ornstein_uhlenbeck_SDE, integrator):
    """
    ornstein_uhlenbeck_process(paths=1, vshape=(), dtype=None, steps=None,
    i0=0, info=None, getinfo=True, method='euler',
    x0=0., theta=0., k=1., sigma=1., dw=None, corr=None, rho=None)

    Ornstein-Uhlenbeck process (mean-reverting Brownian motion).

    Generates a process ``x(t)`` that solves the following SDE::

        dx(t) = k(t)*(theta(t) - x(t))*dt + sigma(t)*dw(t, dt)

    where ``dw(t, dt)`` are standard Wiener process increments with
    correlation matrix specified by ``corr(t)`` or ``rho(t)``.
    ``x0``, SDE parameters and ``dw(t, dt)`` should broadcast to
    ``vshape + (paths,)``.

    Parameters
    ----------
    paths, vshape, dtype, steps, i0, info, getinfo, method
        See ``SDE`` class documentation.
    x0 : array-like
        Initial condition.
    theta, k, sigma : array-like, or callable
        SDE parameters.
    dw, corr, rho
        Specification of stochasticity source of Wiener process increments.
        See ``SDE.source_dw`` documentation.

    Returns
    -------
    x : process
        Once instantiated as ``p``, ``p(timeline)`` performs the integration
        along the given timeline, based on parameters of instantiation,
        and returns the resulting process.

    See also
    --------
    SDE
    SDE.source_dw
    wiener_source
    ornstein_uhlenbeck_SDE
    """


class hull_white_SDE(SDE):
    """
    SDE for an F-factors Hull White process.

    See Also
    --------
    hull_white_process
    """

    # set x0 default value to 0.
    def init(self, s, out_x, x0=0.):
        super().init(s, out_x, x0)

    def more(self, factors=1):
        pass

    def shapes(self, vshape):
        xshape = vshape
        wshape = vshape + (self.args['factors'],)
        return vshape, xshape, wshape

    def sde(self, s, x, theta=0., k=1., sigma=1.):
        return {'dt': k*(theta - x), 'dw': sigma}

    def let(self, s, out_x, x):
        out_x[...] = x.sum(axis=-2)


class hull_white_process(hull_white_SDE, integrator):
    """
    hull_white_process(paths=1, vshape=(), dtype=None, steps=None, i0=0,
    info=None, getinfo=True, method='euler',
    factors=1, x0=0., theta=0., k=1., sigma=1., dw=None, corr=None, rho=None)

    F-factors Hull-White process (sum of F correlated mean-reverting Brownian
    motions).

    Generates a process x(t) that solves the following SDE::

        x(t) = y_1(t) + ... + y_F(t)
        dy_i(t) = k_i(t)*(theta_i(t) - y_i(t))*dt +
                  + sigma_i(t)*dw_i(t, dt)

    where ``dw_i(t, dt)`` are standard Wiener process increments with
    correlations ``dw_i(t, dt)*dw_j(t, dt) = corr(t)[i, j]``.
    ``x0``, SDE parameters and ``dw(t, dt)`` should broadcast to
    ``vshape + (factors, paths)``.

    Parameters
    ----------
    paths, vshape, dtype, steps, i0, info, getinfo, method
        See ``SDE`` class documentation.
    x0 : array-like
        Initial condition.
    theta, k, sigma : array-like, or callable
        SDE parameters.
    dw, corr, rho
        Specification of stochasticity source of Wiener process increments.
        See ``SDE.source_dw`` documentation.

    See also
    --------
    SDE
    SDE.source_dw
    wiener_source
    hull_white_SDE
    ornstein_uhlenbeck_process
    """
    pass


class hull_white_1factor_process(ornstein_uhlenbeck_process):
    """
hull_white_1factor_process(paths=1, vshape=(), dtype=None, steps=None, i0=0,
info=None, getinfo=True, method='euler',
x0=0., theta=0., k=1., sigma=1., dw=None, corr=None, rho=None)

1-factor Hull-White process (F=1 Hull-White process with F-index
collapsed to a scalar). See ``hull_white_process`` class documentation.

See Also
--------
hull_white_process
ornstein_uhlenbeck_process

Notes
-----
Class added for naming convenience. Differs from a ``hull_white_process``
with ``factors=1`` in that the last index of the process parameters has not
been reserved to enumerate factors, and no ``factors`` parameter is
present. Synonymous with ``ornstein_uhlenbeck_process``.
"""
    pass


class cox_ingersoll_ross_SDE(SDE):
    """
    SDE for a Cox-Ingersoll-Ross mean reverting process.

    See Also
    --------
    cox_ingersoll_ross_process
    """

    def sde(self, s, x, theta=1., k=1., xi=1.):
        x_plus = np.maximum(x, 0.)
        return {'dt': k*(theta - x_plus),
                'dw': xi * sqrt(x_plus)}


class cox_ingersoll_ross_process(cox_ingersoll_ross_SDE, integrator):
    """
    cox_ingersoll_ross_process(paths=1, vshape=(), dtype=None, steps=None,
    i0=0, info=None, getinfo=True, method='euler',
    x0=1., theta=1., k=1., xi=1., dw=None, corr=None, rho=None)

    Cox-Ingersoll-Ross mean reverting process.

    Generates a process ``x(t)`` that solves the following SDE::

        dx(t) = k(t)*(theta(t) - x(t))*dt + xi(t)*sqrt(x(t))*dw(t, dt)

    where ``dw(t, dt)`` are standard Wiener process increments with
    correlation matrix specified by ``corr(t)`` or ``rho(t)``.
    ``x0``, SDE parameters and ``dw(t, dt)`` should broadcast to
    ``vshape + (paths,)``. ``x0, theta, k`` should be positive.

    Parameters
    ----------
    paths, vshape, dtype, steps, i0, info, getinfo, method
        See ``SDE`` class documentation.
    x0 : array-like
        Initial condition.
    theta, k, xi : array-like, or callable
        SDE parameters.
    dw, corr, rho
        Specification of stochasticity source of Wiener process increments.
        See ``SDE.source_dw`` documentation.

    Returns
    -------
    x : process
        Once instantiated as ``p``, ``p(timeline)`` performs the integration
        along the given timeline, based on parameters of instantiation,
        and returns the resulting process.

    See also
    --------
    SDE
    SDE.source_dw
    wiener_source
    cox_ingersoll_ross_SDE
    """
    pass


class full_heston_SDE(SDEs):
    """
    SDE for a Heston stochastic volatility process.

    See Also
    --------
    full_heston_process
    heston_process
    """
    q = 2
    addaxis = False
    log = False

    def init(self, s, out_X, x0=1., y0=1.):
        out_x, out_y = self.unpack(out_X)
        out_x[...] = log(x0)
        out_y[...] = y0

    def info_begin(self):
        self.info['negative_y_count'] = np.zeros(
            self.vshape + (self.paths,), dtype=int)

    def sde(self, t, x, y, mu=0., sigma=1.,
            theta=1., k=1., xi=1.):
        y_plus = np.maximum(y, 0.)
        return (
            {'dt': (mu - sigma*sigma*y_plus/2),
             'dw': sigma*sqrt(y_plus)},
            {'dt': k*(theta - y_plus),
             'dw': xi*sqrt(y_plus)}
            )

    def info_next(self):
        iv = self.itervars
        x, y = self.unpack(iv['last_x'])
        negative_y = (y < 0)
        self.info['negative_y_count'] += negative_y

    def result(self, tt, xx):
        xx, yy = self.unpack(xx)
        np.exp(xx, out=xx)
        return process(tt, x=xx), process(tt, x=yy)


class full_heston_process(full_heston_SDE, integrator):
    """full_heston_process(paths=1, vshape=(), dtype=None, steps=None,
    i0=0, info=None, getinfo=True, method='euler',
    x0=1., mu=0., sigma=1., y0=1., theta=1., k=1., xi=1.,
    dw=None, corr=None, rho=None)

    Heston stochastic volatility process (returns both process and volatility).

    Generates processes x(t) and an y(t) that solve the following SDEs::

        dx(t) = mu(t)*x(t)*dt + sigma(t)*x(t)*sqrt(y(t))*dw_x(t, dt),
        dy(t) = k(t)*(theta(t) - y(t))*dt + xi(t)*sqrt(y(t))*dw_y(t, dt)

    where, if ``N = vshape[-1]`` is the size of the last dimension of ``x(t)``,
    ``y(t)``, and ``dw(t, dt)`` are standard Wiener process increments
    with shape ``vshape + (2*N, paths)``::

        dw(t)[..., i, :]*dw(t)[..., j, :] = corr(t)[..., i, j]*dt
        dw_x(t) = dw(t)[..., :N, :],
        dw_y(t) = dw(t)[..., N:, :],

    ``x0`` and SDE parameters should broadcast to ``vshape + (paths,)``.
    ``dw(t, dt)`` should broadcast to ``vshape[:-1] + (2*vshape[-1], paths)``.
    ``x0, y0, theta, k`` should be positive.

    Parameters
    ----------
    paths, vshape, dtype, steps, i0, info, getinfo, method
        See ``SDE`` class documentation.
    x0, y0 : array-like
        Initial conditions for ``x(t)`` and ``y(t)`` processes respectively.
    mu, sigma, theta, k, xi : array-like, or callable
        SDE parameters.
    dw, corr, rho
        Specification of stochasticity source of Wiener process increments.
        See ``SDE.source_dw`` documentation.

    Returns
    -------
    x, y : processes
        Once instantiated as ``p``, ``p(timeline)`` performs the integration
        along the given timeline, based on parameters of instantiation,
        and returns the resulting processes.

    See Also
    --------
    SDE
    SDE.source_dw
    wiener_source
    full_heston_SDE

    Notes
    -----
    ``x(t), y(t)`` are obtained via Euler-Maruyama numerical integration of the
    above SDE for ``y(t)`` and of the following equivalent SDE for
    ``a(t) = log(x(t))``, handling negative values of ``y(t)`` via the
    full truncation algorithm [1]_::

         da(t) = (mu(t) - y(t)*sigma(t)**2/2)*dt + sqrt(y(t))*dw_x(t)

    References
    ----------
    .. [1] Andersen L 2007, Efficient Simulation of the Heston
       Stochastic Volatility Model
       (available at: https://ssrn.com/abstract=946405 or
       http://dx.doi.org/10.2139/ssrn.946405)
    """
    pass


class heston_SDE(full_heston_SDE):
    """
    SDE for a Heston stochastic volatility process.

    See Also
    --------
    heston_process
    full_heston_process
    full_heston_SDE
    """

    def shapes(self, vshape):
        vshape, xshape, wshape = super().shapes(vshape)
        xshape = vshape
        return vshape, xshape, wshape

    def let(self, s, out_x, x):
        if self.addaxis:
            out_x[...] = x[..., 0, :]
        else:
            out_x[...] = x[..., :self.vshape[-1], :]

    def result(self, tt, xx):
        np.exp(xx, out=xx)
        return process(tt, x=xx)


class heston_process(heston_SDE, integrator):
    """heston_process(paths=1, vshape=(), dtype=None, steps=None,
    i0=0, info=None, getinfo=True, method='euler',
    x0=1., mu=0., sigma=1., y0=1., theta=1., k=1., xi=1.,
    dw=None, corr=None, rho=None)

    Heston stochastic volatility process (stores and returns process only).

    Generates a process as in ``full_heston_process`` (see its documentation),
    storing and returning the ``x(t)`` component only.

    Parameters
    ----------
    paths, vshape, dtype, steps, i0, info, getinfo, method
        See ``SDE`` class documentation.
    x0, mu, sigma, y0, theta, k, xi, dw, corr, rho
        See ``full_heston_process`` class documentation.

    Returns
    -------
    x : process
        Once instantiated as ``p``, ``p(timeline)`` performs the integration
        along the given timeline, based on parameters of instantiation,
        and returns the resulting process.

    See Also
    --------
    full_heston_process
    """
    pass


class jumpdiff_SDE(SDE):
    """
    SDE for a jump-diffusion process (lognormal process with
    compound Poisson logarithmic jumps).

    See Also
    --------
    jumpdiff_process
    """
    log = True
    sources = {'dt', 'dw', 'dj'}

    def info_begin(self):
        tt = self.itervars['tt']
        self.info['jump_rate'] = np.zeros(tt.shape, dtype=float)
        self.info['jump_count'] = np.zeros(
            self.vshape + (self.paths,), dtype=int)

    def sde(self, s, x, mu=0., sigma=1.):

        # martingale correction, *NOT* applied
        # code needed to apply the correction:
        #   dj = self.sources['dj']
        #   y, lam = dj.y, dj.lam
        #   y = y(s) if callable(y) else y
        #   lam = lam(s) if callable(lam) else lam
        #   y_exp_mean = np.asarray(y.exp_mean())
        #   nu = lam * (y_exp_mean - 1)
        #   {'dt': (mu - sigma*sigma/2 - nu)}

        return {'dt': (mu - sigma*sigma/2),
                'dw': sigma,
                'dj': 1}

    def info_next(self):
        iv = self.itervars
        dj = self.sources['dj']
        tt, i = iv['tt'], iv['i']
        # mind not breaking the source protocol
        if hasattr(dj, 'dn_value'):
            self.info['jump_rate'][i - 1] += \
                dj.dn_value.sum()/(tt[i] - tt[i - 1])/self.paths
            self.info['jump_count'] += dj.dn_value

    def info_end(self):
        if self.itervars['tt'].size > 1:
            # take care of last time point
            self.info['jump_rate'][-1] = self.info['jump_rate'][-2]


class jumpdiff_process(jumpdiff_SDE, integrator):
    """jumpdiff_process(paths=1, vshape=(), dtype=None, steps=None,
    i0=0, info=None, getinfo=True, method='euler',
    x0=1., mu=0., sigma=1., dw=None, corr=None, rho=None, dj=None, dn=None,
    ptype=int, lam=1., y=None)

    Jump-diffusion process (lognormal process with compound Poisson
    logarithmic jumps).

    Generates a process x(t) that solves the following SDE
    (see [1]_)::

        dx(t) = mu(t)*x(t)*dt + sigma(t)*x(t)*dw(t, dt) + x(t)*dj(t, dt)

    where ``dw(t, dt)`` are standard Wiener process increments with
    correlation matrix specified by ``corr(t)`` or ``rho(t)``, and
    ``dj(t, dt)`` are increments of a Poisson process
    with intensity ``lam(t)``, compounded with random variates
    distributed as ``exp(y(t)) - 1``.

    Parameters
    ----------
    paths, vshape, dtype, steps, i0, info, getinfo, method
        See ``SDE`` class documentation.
    x0 : array-like
        Initial condition.
    mu, sigma : array-like, or callable
        SDE parameters.
    dw, corr, rho
        Specification of stochasticity source of Wiener process increments.
        See ``SDE.source_dw`` documentation.
    dj, dn, ptype, lam, y
        Specification of stochasticity source of compound Poisson process
        increments. See ``SDE.source_dj`` documentation.

    See Also
    --------
    SDE
    SDE.source_dw
    SDE.source_dj
    wiener_source
    cpoisson_source
    jumpdiff_SDE

    Notes
    -----
    The drift of the mean value x_mean(t) of x(t) is mu(t) + nu(t),
    i.e. dx_mean(t)/dt = x_mean(t)*(mu(t) + nu(t)), where::

            nu(t) = lam(t)*(y_exp_mean(t) - 1)
            y_exp_mean(t) = average of exp(y(t))

    ``x(t)`` is obtained via Euler-Maruyama numerical integration of the
    following equivalent SDE for ``a(t) = log(x(t))``::

         da(t) = (mu(t) - sigma(t)**2/2)*dt + x(t)*sigma(t)*dw(t, dt)
                 + x(t)*dh(t, dt)

    where ``dh(t, dt)`` are increments of a Poisson process with
    intensity ``lam(t)`` compounded with random variates distributed
    as ``y(t)``.

    References
    ----------
    .. [1] Tankov P Voltchkova E 2009, Jump-diffusion models: a practitioner's
           guide, Banque et Marches, No. 99, March-April 2009
           (available at:
           http://www.proba.jussieu.fr/pageperso/tankov/tankov_voltchkova.pdf)
    """
    pass


class merton_jumpdiff_SDE(jumpdiff_SDE):
    """
    SDE for a Merton jump-diffusion process.

    See Also
    --------
    merton_jumpdiff_process
    jumpdiff_SDE
    """
    def source_dj(self, dj=None, dn=None, ptype=int,
                  lam=1., a=0., b=1.):
        return super().source_dj(dj=dj, dn=dn, ptype=ptype,
                                 lam=lam, y=norm_rv(a=a, b=b))


class merton_jumpdiff_process(merton_jumpdiff_SDE, integrator):
    """merton_jumpdiff_process(paths=1, vshape=(), dtype=None, steps=None,
    i0=0, info=None, getinfo=True, method='euler',
    x0=1., mu=0., sigma=1., dw=None, corr=None, rho=None, dj=None, dn=None,
    ptype=int, lam=1., a=0., b=1.)

    Merton jump-diffusion process (jump-diffusion process with normal jump size
    distribution).

    Same as ``jumpdiff_process``, where the ``y`` parameter
    is specialized to ``norm_rv(a, b)``, a normal variate with mean ``a(t)``
    and standard deviation ``b(t)``.

    See Also
    --------
    jumpdiff_process
    norm_rv
    """
    pass


class kou_jumpdiff_SDE(jumpdiff_SDE):
    """
    SDE for a double exponential (Kou) jump-diffusion process.

    See Also
    --------
    kou_jumpdiff_process
    jumpdiff_SDE
    """
    def source_dj(self, dj=None, dn=None, ptype=int,
                  lam=1., a=0.5, b=0.5, pa=0.5):
        return super().source_dj(dj=dj, dn=dn, ptype=ptype,
                                 lam=lam,
                                 y=double_exp_rv(a=a, b=b, pa=pa))


class kou_jumpdiff_process(kou_jumpdiff_SDE, integrator):
    """kou_jumpdiff_process(paths=1, vshape=(), dtype=None, steps=None,
    i0=0, info=None, getinfo=True, method='euler',
    x0=1., mu=0., sigma=1., dw=None, corr=None, rho=None, dj=None, dn=None,
    ptype=int, lam=1., a=0.5, b=0.5, pa=0.5)

    Double exponential (Kou) jump-diffusion process
    (jump-diffusion process with double exponential
    jump size distribution).

    Same as ``jumpdiff_process``, where the ``y`` parameter
    is specialized to ``double_exp_rv(a, b, pa)``, a double exponential variate
    with scale ``a(t)`` with probability ``pa(t)``, and
    ``-b(t)`` with probability ``(1 - pa(t))``.

    See Also
    --------
    jumpdiff_process
    double_exp_rv
    """
    pass


# -------------------------
# docstrings postprocessing
# -------------------------


# add trailing Attributes and Methods sections to all SDEs and processes
# to avoid repetition of parent class items in documentation
for _cls in (wiener_SDE, wiener_process,
             lognorm_SDE, lognorm_process,
             ornstein_uhlenbeck_SDE, ornstein_uhlenbeck_process,
             hull_white_SDE, hull_white_process, hull_white_1factor_process,
             cox_ingersoll_ross_SDE, cox_ingersoll_ross_process,
             full_heston_SDE, full_heston_process,
             heston_SDE, heston_process,
             jumpdiff_SDE, jumpdiff_process,
             merton_jumpdiff_SDE, merton_jumpdiff_process,
             kou_jumpdiff_SDE, kou_jumpdiff_process
             ):
    _cls.__doc__ += """
    Attributes
    ----------
    See SDE class documentation.

    Methods
    -------
    See SDE class documentation.
    """
