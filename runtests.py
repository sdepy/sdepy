import shutil
import sys
import os
import importlib
import doctest
import pdb

import sdepy
from sdepy import test
from sdepy import _config

from sys import version as python_version
from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version
from pytest import __version__ as pytest_version


# -------------------
# inspect directories
# -------------------

def getdir(file):
    return os.path.dirname(os.path.abspath(file))


# this script may run with current directory set
# either to the package home directory . or to ./build/tests
SCRIPT_DIR = getdir(__file__)
SCRIPT_NAME = os.path.split(__file__)[-1]
ishome = os.path.exists(os.path.join(SCRIPT_DIR, 'doc', 'quickguide.rst'))
HOME_DIR = SCRIPT_DIR if ishome else os.path.join(
    SCRIPT_DIR, os.pardir, os.pardir)
assert os.path.exists(os.path.join(HOME_DIR, 'doc', 'quickguide.rst'))
TEST_DIR = os.path.join(HOME_DIR, 'build', 'tests')
if ishome:
    assert os.path.samefile(SCRIPT_DIR, HOME_DIR)
else:
    assert os.path.samefile(SCRIPT_DIR, TEST_DIR)

# probe location of package to be tested
PACKAGE_DIR = getdir(sdepy.__file__)
issource = os.path.samefile(HOME_DIR,
                            os.path.join(PACKAGE_DIR, os.pardir))


def print_info():
    """
    Print directories and versions information
    """
    print('script dir = ', os.path.abspath(SCRIPT_DIR))
    print('home dir =   ', os.path.abspath(HOME_DIR))
    print('test dir =   ', os.path.abspath(TEST_DIR))
    print('sdepy dir =  ', os.path.abspath(PACKAGE_DIR))
    print('sdepy version = ', sdepy.__version__)
    print('python version =', python_version)
    print('numpy version = ', numpy_version)
    print('scipy version = ', scipy_version)
    print('pytest version =', pytest_version, '\n')
    return 0


def no_source():
    """
    Fail if sdepy was loaded from source, not from installed package
    """
    assert not issource
    return 0


# --------------
# setup and exit
# --------------

def setup_tests():
    """
    Make ./build and ./build/tests directories if not present,
    and copy ./runtests.py to ./build/tests
    """
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    shutil.copyfile(os.path.join(HOME_DIR, SCRIPT_NAME),
                    os.path.join(TEST_DIR, SCRIPT_NAME))
    return 0


def exit_tests():
    """
    Copy ./build/tests/.coverage, if any, to .
    """
    if os.path.exists(os.path.join(TEST_DIR, '.coverage')):
        shutil.copyfile(os.path.join(TEST_DIR, '.coverage'),
                        os.path.join(HOME_DIR, '.coverage'))
    return 0


# ---------
# run tests
# ---------

def reload():
    """
    Reload all sdepy and sdepy.tests modules,
    based on the configuration stored in sdepy._config variables
    (sdepy._config is not reloaded)
    """

    # reimport sdepy
    importlib.reload(sdepy.infrastructure)
    importlib.reload(sdepy.integration)
    importlib.reload(sdepy.analytical)
    importlib.reload(sdepy.kfun)
    importlib.reload(sdepy.shortcuts)
    importlib.reload(sdepy)

    # reimport sdepy.tests
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
    """
    Run doctest on ./doc/quickguide.rst
    """
    # needs matplotlib.pyplot to be installed
    print_info()
    return doctest.testfile(
        os.path.join(HOME_DIR, 'doc', 'quickguide.rst'),
        module_relative=False
        ).failed


def run(label='fast', doctests=False, warnings='pass'):
    """
    Run tests with given options
    """
    print_info()
    pytest_args = {'pass': (),
                   'fail': ('-W', 'error::Warning')}
    return int(not test(
        label=label, doctests=doctests,
        pytest_args=pytest_args[warnings]))


def run_insane(warnings='pass'):
    """
    Test package in its multiple configurations
    (time consuming, not part of the ci testing suite)
    """
    # needs matplotlib.pyplot to be installed
    # saves realized errors and plots in PACKAGE_DIR/tests/cfr
    print_info()
    results = []

    print('------------------')
    print('RUNNING FULL TESTS')
    print('------------------\n')

    # run default tests
    results.append(run('fast', warnings=warnings))

    # run quickguide doctests
    count_quickguide = run_quickguide()

    # run tests with maximum code coverage, in all
    # kfunc modes
    _config.PLOT = True
    _config.SAVE_ERRORS = True
    _config.VERBOSE = True
    _config.QUANT_TEST_MODE = 'LD'
    for k in ('shortcuts', 'all', None):
        _config.KFUNC = k
        reload()
        results.append(run('full', doctests=True, warnings=warnings))

    # run quantitative tests with high resolution
    # and maximum code coverage
    _config.PLOT = True
    _config.SAVE_ERRORS = True
    _config.VERBOSE = True
    _config.QUANT_TEST_MODE = 'HD'
    _config.KFUNC = None
    reload()
    results.append(run('quant or config', warnings=warnings))

    # summarize results
    count_failures = sum(results)

    print('\n--------------------')
    print('FULL TESTS COMPLETED')
    print('--------------------\n')

    return count_failures + count_quickguide


# --------------------------------------
# minimal setup as a command line script
# --------------------------------------

usage = """
Run tests for sdepy package.

python runtests.py command1 command2 ...

Available commands ('.' is the package home directory):

"""
for command in (setup_tests, exit_tests,
                print_info, no_source,
                run_quickguide, run, run_insane):
    usage += command.__name__ + '(): ' + command.__doc__ + '\n'

if __name__ == '__main__':
    cmds = sys.argv[1:]

    if not cmds or cmds[0] in ('-h', '--help'):
        print(usage)
        sys.exit(0)
    else:
        test_result = sum(eval(cmd) for cmd in cmds)
        if any(cmd[:3] == 'run' for cmd in cmds):
            print(cmds, 'FINAL RESULT (0 if all tests passed):', test_result)
        sys.exit(1 if test_result else 0)
