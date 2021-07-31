"""
=============================
COMMON TESTING INFRASTRUCTURE
=============================
"""
# common imports
import numpy as np
from numpy.testing import (
    assert_, assert_raises, assert_warns,
    assert_array_equal, assert_allclose,
    )
import itertools
import os
import sys
import warnings
try:
    import pytest
    PYTEST = True
except ImportError:
    PYTEST = False


# -------------
# pytest tester
# -------------

class _pytest_tester:
    """Tester class invoking pytest.main()"""

    def __init__(self, module_name):
        self.module_name = module_name

    def __call__(self, label='fast', doctests=False, pytest_args=()):
        # TODO: change interface to
        # __call__(self, label='fast', doctests=False, paths=100,
        #          plot=False, outdir=None, verbose=False,
        #          rng='legacy', pytest_args=()):
        #
        """
        Invoke the sdepy testing suite (requires pytest>=3.8.1
        to be installed).

        Parameters
        ----------
        label : string, optional
            Sets the scope of tests. May be one of ``'full'`` (run all tests),
            ``'fast'`` (avoid slow tests), ``'slow'`` (perform slow tests
            only), ``'quant'`` (perform quantitative tests only).
        doctests : bool, optional
            If ``True``, doctests are performed in addition to the tests
            specified by ``label``.
        pytest_args : iterable of strings, optional
            Additional arguments passed to ``pytest.main()``.

        Returns
        -------
        ``True`` if all tests passed, ``False`` otherwise.
        """

        module_path = os.path.abspath(
            sys.modules[self.module_name].__path__[0])
        ini_file = os.path.join(module_path, 'tests', '_pytest.ini')

        label = {
            'fast': 'not slow',
            'full': None
            }.get(label, label)

        pytest_args = (
            list(pytest_args) +
            ['--verbose', '--capture=no'] +
            ['-c', ini_file] +
            (['-m', label] if label else []) +
            (['--doctest-modules'] if doctests else []) +
            ['--pyargs', self.module_name]
            )

        if PYTEST:
            test_result = pytest.main(pytest_args)
            return (test_result == 0)
        else:
            raise ImportError(
                'sdepy.test() requires pytest to be installed')


# ----------------------------
# *** PACKAGE TO BE TESTED ***
# ----------------------------

# in all test modules, 'sp' is the package
# to be tested
import sdepy
import sdepy as sp


# ---------------------------
# set up plotting if required
# ---------------------------

try:
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = 12, 6

except ImportError:
    pass


# -----------------------------------
# additional parameters for tests
# (local to the .tests.shared module)
# -----------------------------------

# exact equality is tested up to the float resolution times _EPS_FACTOR
# (see eps function below)
_EPS_FACTOR = 16

# each testing routine should call rng_setup() and
# access random numbers via rng().random, rng().normal etc.
_LEGACY_SEED = 1234

# DIR is the directory used, in quantitative tests,
# to retrieve expected errors and to save realized errors and plots
_PACKAGE_DIR = os.path.split(sdepy.__file__)[0]
# input dir for legacy rng tests
_INPUT_DIR = os.path.join(_PACKAGE_DIR, 'tests', 'cfr')

# Relative tolerance when checking expected vs. realized errors:
# the expected errors to be tested, returned by load_errors,
# are the stored expected error times (1 + _EXPECTED_ERROR_RTOL).
_EXPECTED_ERROR_RTOL = 0.01


# --------------------------
# functions for common tasks
# --------------------------

if PYTEST:
    # Decorator to mark test as quantitative
    quant = pytest.mark.quant
    # Decorator to mark test as slow
    slow = pytest.mark.slow
else:
    # dummy decorators in case pytest is not installed
    quant = slow = lambda f: f


def rng_setup():
    """Use in each test to setup the sdepy default random number generator
    (replaces legacy `np.random.seed(SEED)` calls).
    """
    test_rng = sdepy._config.TEST_RNG
    if sdepy.infrastructure.default_rng == np.random:
        # running with legacy numpy versions, without np.random.RandomState:
        # fall back on numpy global state
        np.random.seed(_LEGACY_SEED)
    elif test_rng == 'legacy':
        # test with numpy Legacy Random Generation
        sdepy.infrastructure.default_rng = (
            np.random.RandomState(_LEGACY_SEED))
    elif test_rng is None:
        # do not interfere with global sdepy rng
        pass
    # else, use rng or rng maker given in _config.TEST_RNG
    elif callable(test_rng):
        sdepy.infrastructure.default_rng = test_rng()
    else:
        sdepy.infrastructure.default_rng = test_rng


def rng():
    """Use `rng()` in each test to access the current
    sdepy default random number generator."""
    return sdepy.infrastructure.default_rng


def do(testing_function, *case_iterators, **args):
    """call a test function f across all test cases exposed
    in case_iterators, passing keyword arguments in args"""

    for z in case_iterators:
        # enforce predictability of test order
        assert isinstance(z, (list, tuple))

    for case in itertools.product(*case_iterators):
        testing_function(*case, **args)
        print('.', sep='', end='')


def eps(dtype):
    if np.dtype(dtype).kind in ('f', 'c'):
        return np.finfo(dtype).resolution * _EPS_FACTOR
    else:
        return 0


class const_errors:

    def __init__(self, error):
        self.error = error

    def __getitem__(self, key):
        return (self.error, self.error)

    def __setitem__(self, key, value):
        pass


noerrors_expected = const_errors(1000)
noerrors_realized = const_errors(0)


def load_errors(context, dtype=float):
    if sdepy._config.TEST_RNG != 'legacy':
        return noerrors_expected

    fname = os.path.join(_INPUT_DIR, context + '_err_expected.txt')
    min_error = eps(float)

    def err(error):
        return max(min_error, abs(float(error)*(1 + _EXPECTED_ERROR_RTOL)))

    if os.path.exists(fname):
        with open(fname, 'r') as f:
            lines = [line.split() for line in f.readlines()]
            errors = {test_key: (err(mean_error), err(max_error))
                      for test_key, mean_error, max_error in lines[2:]}
            return errors
    else:
        warnings.warn('No expected errors file found while testing {}: '
                      'file {} does not exist'.format(context, fname),
                      RuntimeWarning)
        return noerrors_expected


def save_errors(context, errors):
    DIR = sdepy._config.OUTPUT_DIR
    if DIR is not None:
        fname = os.path.join(DIR, context + '_err_realized.txt')
        with open(fname, 'w') as f:
            print('{:40} {:>12} {:>12}\n'.format(
                'TEST_ID', 'MEAN_ERROR', 'MAX_ERROR'), file=f)
            for test_key, (mean_error, max_error) in errors.items():
                print('{:40s} {:12.5g} {:12.5g}'.format(
                    test_key, mean_error, max_error), file=f)


def save_figure(fig, fig_id):
    DIR = sdepy._config.OUTPUT_DIR
    if DIR is not None:
        fname = os.path.join(DIR, fig_id + '.png')
        fig.savefig(fname, dpi=300)


def plot_histogram(hist, **kwargs):
    counts, bins = hist
    return plt.hist(bins[:-1], bins=bins, weights=counts, **kwargs)


# -----------------------
# assert_ for quant tests
# -----------------------

message = """
A quantitative test has failed, running with {} paths using numpy
legacy random number generation. This test relies on the exact reproducibility
of expected errors, once random numbers have been seeded with
`np.random.RandomState({})`, and its failure may not necessarily indicate
that the package is broken. Consider running tests as
`sdepy.test('full', rng=np.random.default_rng(), paths=100000, outdir='.')`
and inspecting the realized error summaries and plots saved in the current
directory.
"""


def assert_quant(flag):

    if (sdepy._config.TEST_RNG == 'legacy') and (not flag):
        assert_(flag, message.format(sdepy._config.PATHS, _LEGACY_SEED))
    else:
        # should never occur, non legacy tests run with expected errors
        # as `noerrors_expected`
        assert_(flag, 'Quantitative test dependant on random number '
                f'generation failed, using '
                f'rng={sdepy.infrastructure.default_rng}')
