import numpy as np
import scipy
import matplotlib.pyplot as plt
import warnings
import cProfile
import pstats
import pdb
import shutil
import sys
import os
import inspect
import importlib
import doctest
from numpy.testing import rundocs

# warnings.filterwarnings('error')
# warnings.filterwarnings('always')
# warnings.filterwarnings('default')

# import from local package
import sdepy
from sdepy import *

# check which target is being tested
DIR = os.path.split(sdepy.__file__)[0]
print(DIR)

# setup the tester
tst = sdepy.test


# --------------------------------------------
# import statements for running tests manually
# --------------------------------------------

from sdepy.tests.test_process import *
from sdepy.tests.test_quant import *
from sdepy.tests.test_processes import *
from sdepy.tests.test_integrator import *
from sdepy.tests.test_montecarlo import *
from sdepy.tests.test_analytical import *
from sdepy.tests.test_source import *
from sdepy.tests.test_kfunc import *


# -----------
# other tools
# -----------

def profile(f):
    """
    profiling helper
    """
    pr = cProfile.Profile()

    pr.enable()
    f()
    pr.disable()

    ps = pstats.Stats(pr)
    ps = ps.strip_dirs().sort_stats('tottime')

    ps.print_stats(10)
    return ps


def pl(x):
    for a in x:
        print(a)


def corr(a, b=None):
    if b is None:
        a, b = a[0], a[1]
    cov = np.cov(a, b)
    n = cov.shape[0]
    return np.asarray(
        [[cov[i, j]/sqrt(cov[i, i]*cov[j, j]) for i in range(n)]
         for j in range(n)])


def getcode(in_, out, mode='x'):
    header = \
'''# ------------------------------------------
# This file has been automatically generated
# from {}
# ------------------------------------------
'''.format(in_)

    with open(in_) as i, open(out, mode) as o:
        print(header, file=o)
        while True:
            a = i.readline()
            if a == '':
                break
            elif a.strip()[:3] in ('>>>', '...'):
                print(a.strip()[4:], file=o)


def quickguide_make():
    getcode(os.path.join('.', 'doc', 'quickguide.rst'),
            'quickguide.py',
            mode='w')


def quickguide_run():
    quickguide_make()
    os.system('python quickguide.py')


def quickguide_doctests():
    doctest.testfile(
        os.path.join('.', 'doc', 'quickguide.rst')
        )
