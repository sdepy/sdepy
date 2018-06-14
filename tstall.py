import shutil
import sys
import os
import importlib
import doctest
import pdb

# import from local package
import sdepy
from sdepy import test as tst
from sdepy import _config

# check which target is being tested
DIR = os.path.split(sdepy.__file__)[0]
print(DIR)


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


def quickguide_doctests():
    doctest.testfile(
        os.path.join('.', 'doc', 'quickguide.rst')
        )


def tstall():

    res = []

    def run_tests(*var, **args):
        res.append(tst(*var, **args))

    print('------------------')
    print('RUNNING FULL TESTS')
    print('------------------\n')

    # run default tests
    res.append(tst())

    # run quickguide doctests
    quickguide_doctests()

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

    return res
