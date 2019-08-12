import numpy as np
import scipy
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

try:
    import matplotlib.pyplot as plt
except Exception:
    pass

# import from local package
import sdepy
from sdepy import *
import runtests
from runtests import *

# import statements for running tests manually
from sdepy.tests.test_process import *
from sdepy.tests.test_quant import *
from sdepy.tests.test_processes import *
from sdepy.tests.test_integrator import *
from sdepy.tests.test_montecarlo import *
from sdepy.tests.test_analytical import *
from sdepy.tests.test_source import *
from sdepy.tests.test_kfunc import *

# check which target is being tested
runtests.print_info()

# setup the tester
test = sdepy.test


# -----
# tools
# -----

def profile(f, show=10):
    """
    Profiling helper
    """
    pr = cProfile.Profile()

    pr.enable()
    f()
    pr.disable()

    ps = pstats.Stats(pr)
    ps = ps.strip_dirs().sort_stats('tottime')

    ps.print_stats(show)
    return ps


def corr(a, b=None):
    """
    Correlation matrix
    """
    if b is None:
        a, b = a[0], a[1]
    cov = np.cov(a, b)
    n = cov.shape[0]
    return np.asarray(
        [[cov[i, j]/sqrt(cov[i, i]*cov[j, j]) for i in range(n)]
         for j in range(n)])


def getcode(in_, out, mode='x'):
    """
    Parse doc file named 'in_', trasform doctests
    into executable code and all the rest into
    comments, and save output in file named 'out'
    """
    header = \
        '# ------------------------------------------\n'\
        '# This file has been automatically generated\n'\
        '# from {}                                   \n'\
        '# ------------------------------------------\n'\
        .format(in_)

    with open(in_) as i, open(out, mode) as o:
        print(header, file=o)
        sep = ''
        while True:
            a = i.readline()
            a, EOF = a.strip(), a == ''
            if EOF:
                break
            elif a[:3] in ('>>>', '...'):
                print(a[4:], file=o)
                sep = '\n\n'
            else:
                print((sep + '# ' + a) if a else sep + '#', file=o)
                sep = ''


def quickguide_make():
    """
    Generate ./quickguide.py from ./doc/quickguide.rst
    """
    getcode(os.path.join('.', 'doc', 'quickguide.rst'),
            'quickguide.py',
            mode='w')


def inspect_warnings():
    """
    Execute all tests in sdepy.tests module,
    printing warnings if any.
    """
    warnings.filterwarnings('default')
    for key, item in globals().items():
        if key[:5] == 'test_':
            print('\nrunning', key)
            item()
