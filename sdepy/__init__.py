"""
This package provides tools to state and numerically
integrate Ito Stochastic Differential Equations (SDEs), including equations
with time-dependent parameters, time-dependent correlations, and
stochastic jumps, and to compute with, and extract statistics from,
their realized paths.

Package contents:

    1.  A set of tools to ease computations with stochastic processes,
        as obtained from numerical integration of the corresponding SDE,
        is provided via the ``process`` and ``montecarlo`` classes
        (see `Infrastructure`_):

        *   The ``process`` class, a subclass of ``numpy.ndarray`` representing
            a sequence of values in time, realized in one or several paths.
            Algebraic manipulations and ufunc computations are supported for
            instances that share the same timeline, or are constant, and
            comply with numpy broadcasting rules. Interpolation along
            the timeline is supported via callability of ``process`` instances.
            Process-specific functionalities, such as averaging and indexing
            along time or across paths, are delegated to process-specific
            methods, attributes and properties (no overriding
            of ``numpy.ndarray`` operations).

        *   The ``montecarlo`` class, as an aid to cumulate the results
            of several Monte Carlo simulations of a given
            stochastic variable, and to extract summary estimates
            for its probability distribution function and statistics.

    2.  Numerical realizations of the differentials commonly found
        as stochasticity sources in SDEs, are provided via
        the ``source`` class and its subclasses, with or without memory
        of formerly invoked realizations (see `Stochasticity Sources`_).

    3.  A general framework for stochastic step by step simulations,
        and for numerical SDE integration, is provided via the
        ``paths_generator`` class, and its cooperating subclasses
        ``integrator``, ``SDE`` and ``SDEs``
        (see `SDE Integration Framework`_).
        The full API allows for extensive customization of preprocessing,
        post-processing, stochasticity sources instantiation and handling,
        integration algorithms etc.
        The ``integrate`` decorator provides a simple and concise interface
        to handle standard use cases, via Euler-Maruyama integration.

    4.  Several preset stochastic processes are provided, including lognormal,
        Ornstein-Uhlenbeck, Hull-White n-factor, Heston, and
        jump-diffusion processes (see `Stochastic Processes`_).
        Each process consists of a process generator class,
        a subclass of ``integrator`` and ``SDE``, named with a
        ``_process`` suffix, and a definition of the underlying SDE,
        a subclass of ``SDE`` or ``SDEs``, named with a ``_SDE`` suffix.

    5.  Several analytical results relating to the preset stochastic
        processes are made available, as a general reference
        and for testing purposes (see `Analytical Results`_).
        They are limited to the case of constant process parameters,
        and with some further limitations on the parameters' domains.
        Function arguments are consistent with those of the
        corresponding processes. Suffixes ``_pdf``, ``_cdf`` and ``_chf``
        stand respectively for probability distribution
        function, cumulative probability distribution function,
        and characteristic function.
        Black-Scholes formulae for the valuation of call and put options
        have been included (with prefix ``bs``).

    6.  As an aid to interactive and notebook sessions, shortcuts are provided
        for stochasticity sources and preset processes (see `Shortcuts`_).
        Shortcuts have been wrapped as "kfuncs", objects with managed
        keyword arguments that simplify interactive workflow when
        frequent parameters tuning operations are needed
        (see ``kfunc`` decorator documentation).
        Analytical results are wrapped as kfuncs as well.

For all sources and processes, values can take any shape,
scalar or multidimensional. Correlated multivariate stochasticity sources are
supported. Poisson jumps are supported, and may be compounded with
any random variable supported by scipy.stats.
Time-varying process parameters (correlations, intensity of Poisson
processes, volatilities etc.) are allowed whenever applicable.
``process`` instances act as valid stochasticity source realizations (as does
any callable object complying with a ``source`` protocol), and may be
passed as a source specification when computing the realization of a given
process.

Computations are fully vectorized across paths, providing an efficient
infrastructure for simulating a large number of process realizations.
Less so, for large number of time steps: integrating 100 time steps
across one million paths takes seconds, one million time steps across
100 paths takes minutes.
"""

import sys
import warnings
from .infrastructure import *
from .integration import *
from .analytical import *
from .kfun import *
from .shortcuts import *

from .tests.shared import _pytest_tester
test = _pytest_tester(__name__)

__version__ = '1.1.2'

_exclude = ('sys',
            'np', 'numpy', 'scipy',
            'exp', 'log', 'sqrt',
            'bisect', 'inspect', 'warnings',
            'infrastructure',
            'integration',
            'analytical',
            'kfun',
            'shortcuts',
            'tests',
            )

_include = ('__version__',)

__all__ = [s for s in dir() if
           not s.startswith('_') and s not in _exclude]

__all__ += _include

if sys.version_info[:2] <= (3, 5):
    if not sys.warnoptions:
        warnings.filterwarnings('default', category=DeprecationWarning)
    warnings.warn('The use of SdePy with Python 3.5 is deprecated '
                  'and will not be supported in future releases.',
                  DeprecationWarning)

