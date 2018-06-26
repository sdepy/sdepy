import shutil
import sys
import os
import importlib
import doctest
import pdb

import sdepy
from sdepy import test
from sdepy import _config


# -------------------
# inspect directories
# -------------------

def getdir(file):
    return os.path.dirname(os.path.abspath(file))

# this script runs either from . or from ./build/tests
SCRIPT_DIR = getdir(__file__)
ishome = os.path.exists(os.path.join(SCRIPT_DIR, 'doc', 'quickguide.rst'))
HOME_DIR = SCRIPT_DIR if ishome else os.path.join(
    SCRIPT_DIR, os.pardir, os.pardir)
assert os.path.exists(os.path.join(HOME_DIR, 'doc', 'quickguide.rst'))
TEST_DIR = os.path.join(HOME_DIR, 'build', 'tests')
if not ishome:
    assert os.path.samefile(SCRIPT_DIR, TEST_DIR)

# tests should run from the installed package, not from source
PACKAGE_DIR = getdir(sdepy.__file__)
issource = os.path.samefile(HOME_DIR,
                            os.path.join(PACKAGE_DIR, os.pardir))

print('running runtests.py with args', sys.argv[1:])
print('script dir = ', os.path.abspath(SCRIPT_DIR))
print('home dir =   ', os.path.abspath(HOME_DIR))
print('test dir =   ', os.path.abspath(TEST_DIR))
print('package dir =', os.path.abspath(PACKAGE_DIR))


# --------------
# setup and exit
# --------------

def setup_tests():
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    shutil.copyfile(os.path.join(HOME_DIR, __file__),
                    os.path.join(TEST_DIR, __file__))


def exit_tests():
    if os.path.exists(os.path.join(TEST_DIR, '.coverage')):
        shutil.copyfile(os.path.join(TEST_DIR, '.coverage'),
                        os.path.join(HOME_DIR, '.coverage'))


# ---------
# run tests
# ---------

def reload():

    # reimport sdepy
    importlib.reload(sdepy.infrastructure)
    importlib.reload(sdepy.integration)
    importlib.reload(sdepy.analytical)
    importlib.reload(sdepy.kfun)
    importlib.reload(sdepy.shortcuts)
    importlib.reload(sdepy)

    # reimport tests
    importlib.reload(sdepy.tests.shared)
    importlib.reload(sdepy.tests.test_analytical)
    importlib.reload(sdepy.tests.test_integrator)
    importlib.reload(sdepy.tests.test_kfunc)
    importlib.reload(sdepy.tests.test_montecarlo)
    importlib.reload(sdepy.tests.test_process)
    importlib.reload(sdepy.tests.test_processes)
    importlib.reload(sdepy.tests.test_quant)
    importlib.reload(sdepy.tests.test_source)
    importlib.reload(sdepy.tests)


def run_quickguide():
    # needs matplotlib.pyplot to be installed
    return doctest.testfile(
        os.path.join(HOME_DIR, 'doc', 'quickguide.rst'),
        module_relative=False
        ).failed


def run_fast():
    assert not issource
    res = test()
    return res.errors + res.failures


def run_full():
    assert not issource
    res = test('full', doctests=True)
    return res.errors + res.failures


def run_insane():
    # needs matplotlib.pyplot to be installed
    # saves realized errors and plots in PACKAGE_DIR/tests/cfr
    assert not issource
    res = []

    def run_tests(*var, **args):
        res.append(test(*var, **args))

    print('------------------')
    print('RUNNING FULL TESTS')
    print('------------------\n')

    # run default tests
    run_tests()

    # run quickguide doctests
    res_quickguide = run_quickguide()

    # run tests with maximum code coverage, in all
    # kfunc modes
    _config.PLOT = True
    _config.SAVE_ERRORS = True
    _config.VERBOSE = True
    _config.QUANT_TEST_MODE = 'LD'
    for k in ('shortcuts', 'all', None):
        _config.KFUNC = k
        reload()
        run_tests('full', doctests=True)

    # run quantitative tests with high resolution
    # and maximum code coverage
    _config.PLOT = True
    _config.SAVE_ERRORS = True
    _config.VERBOSE = True
    _config.QUANT_TEST_MODE = 'HD'
    _config.KFUNC = None
    reload()
    run_tests('quant or config')

    # summarize results
    count_errors = sum(len(x.errors) for x in res)
    count_failures = sum(len(x.failures) for x in res)

    print('\n--------------------')
    print('FULL TESTS COMPLETED')
    print('--------------------\n')

    print('Full tests results: {} errors, {} failures'
          .format(count_errors, count_failures))

    return count_errors + count_failures + res_quickguide


# --------------------------------------
# minimal setup as a command line script
# --------------------------------------

if __name__ == '__main__':
    for cmd in sys.argv[1:]:
        if eval(cmd):
            sys.exit(1)
