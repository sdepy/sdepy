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
import sdepy as sp
from sdepy import _config


# ---------------------------
# set up plotting if required
# ---------------------------

if _config.PLOT:
    import matplotlib.pyplot as plt

    # figure size
    plt.rcParams['figure.figsize'] = 12, 6


# ----------------------
# common test parameters
# ----------------------

KFUNC = _config.KFUNC

# exact equality is tested up to the float resolution times EPS_FACTOR
# (see eps function below)
EPS_FACTOR = 16

# each testing routine should call np.random.seed(SEED)
SEED = 1234

# DIR is the directory used, in quantitative tests,
# to retrieve expected errors and to save realized errors and plots
PACKAGE_DIR = os.path.split(sp.__file__)[0]
DIR = os.path.join(PACKAGE_DIR, 'tests', 'cfr')

# Saving flags
# PLOT = False disables plots construction and saving
# (each testing module checks this flag and behaves accordingly).
# SAVE_ERRORS = False disables saving realized errors
# (see save_errors() below).
PLOT = _config.PLOT
SAVE_ERRORS = _config.SAVE_ERRORS

# testing flags
QUANT_TEST_FAIL = _config.QUANT_TEST_FAIL
VERBOSE = _config.VERBOSE

# Relative tolerance when checking expected vs. realized errors:
# the expected errors to be tested, returned by load_errors,
# are the stored expected error times (1 + EXPECTED_ERROR_RTOL).
EXPECTED_ERROR_RTOL = 0.01

# High or low definition of quantitative tests
QUANT_TEST_MODE = _config.QUANT_TEST_MODE
assert QUANT_TEST_MODE in {'HD', 'LD'}
PATHS_HD = 100*1000
PATHS_LD = 100


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
        return np.finfo(dtype).resolution * EPS_FACTOR
    else:
        return 0


def choice(list_):
    return list_[np.random.randint(len(list_))]


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
    fname = os.path.join(DIR, context + '_err_expected.txt')
    min_error = eps(float)

    def err(error):
        return max(min_error, abs(float(error)*(1 + EXPECTED_ERROR_RTOL)))

    if QUANT_TEST_FAIL:
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                lines = [line.split() for line in f.readlines()]
                errors = {test_key: (err(mean_error), err(max_error))
                          for test_key, mean_error, max_error in lines[2:]}
        else:
            warnings.warn('No expected errors file found while testing {}: '
                          'file {} does not exist'.format(context, fname),
                          RuntimeWarning)
            errors = noerrors_expected
    else:
        errors = noerrors_expected

    return errors


def save_errors(context, errors):
    if SAVE_ERRORS:
        fname = os.path.join(DIR, context + '_err_realized.txt')
        with open(fname, 'w') as f:
            print('{:40} {:>12} {:>12}\n'.format(
                'TEST_ID', 'MEAN_ERROR', 'MAX_ERROR'), file=f)
            for test_key, (mean_error, max_error) in errors.items():
                print('{:40s} {:12.5g} {:12.5g}'.format(
                    test_key, mean_error, max_error), file=f)


def save_figure(fig, fig_id):
    fname = os.path.join(DIR, fig_id + '.png')
    fig.savefig(fname, dpi=300)


def plot_histogram(hist, **kwargs):
    counts, bins = hist
    return plt.hist(bins[:-1], bins=bins, weights=counts, **kwargs)


# -----------------------
# assert_ for quant tests
# -----------------------

message = """
A quantitative test has failed, running in {} definition with {} paths.
This test relies on the reproducibility of expected errors, once
random numbers have been seeded with np.random.seed({}), and its failure
may not necessarily indicate that the package is broken.
Consider changing the test configuration in
{} by setting

    QUANT_TEST_FAIL = False
    QUANT_TEST_MODE = 'HD'
    PLOT = True

and inspecting the plots that are generated in the folder
{}."""


def assert_quant(flag):

    if not flag:
        definition, paths = (
            ('low', PATHS_LD) if _config.QUANT_TEST_MODE == 'LD'
            else ('high', PATHS_HD))

        warnings.warn(
            message.format(definition, paths, SEED,
                           os.path.join(PACKAGE_DIR, '_config.py'),
                           DIR),
            RuntimeWarning)

    assert_(flag)
