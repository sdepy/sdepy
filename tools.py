import numpy as np
import scipy
import warnings
import cProfile
import pstats
import pdb
import shutil
import sys
import os
import pathlib
import nbformat as nbf
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


# used to generate quickguide.py in
# versions <= 1.1.0
def getcode(in_, out, mode='w', header=''):
    """
    Convert documentation file to python code.

    Parse doc file named 'in_', transform doctests
    into executable code and the rest into comments,
    and save output in file named 'out'.
    Inserts 'header' at the top.
    """
    in_, out = str(pathlib.Path(in_)), str(pathlib.Path(out))
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


def getnotebook(in_, out,
                skip=0, header=(), magic='',
                to_markdown=lambda x: x,
                to_code=lambda x: x):
    """
    Convert documentation file to jupyter notebook.

    Parse doc file named 'in_', transform doctests
    into code cells (omitting output) and the rest into
    markdown cells, and save output in a jupyter notebook named 'out'.
    Skips 'skip' initiali lines, inserts lines in 'header' at the top,
    puts 'magic' text at the beginning of the first code cell.

    Markdown and code text is preprocessed via 'to_markdown'
    and 'to_code' respectively.
    """
    in_, out = str(pathlib.Path(in_)), str(pathlib.Path(out))
    with open(in_) as f:
        z = list(header) + [x.rstrip() + '\n'
             for x in f.readlines()[skip:]]

    nb = nbf.v4.new_notebook()
    nb['cells'] = []
    while z:
        text = ''
        # parse markdown cell
        while z and z[0][:4] != '    ':
            text += z.pop(0)
        text = to_markdown(text)
        nb['cells'].append(
            nbf.v4.new_markdown_cell(text))

        text = ''
        # parse code cell
        while z and z[0][:4] in ('    ', '\n'):
            line = z.pop(0)
            if line == '\n':
                text += '\n'
            elif len(line[4:]) >= 3 and line[4:7] in ('>>>', '...'):
                text += line[8:]
            else:
                pass  # skip code output
        text = to_code(text)
        nb['cells'].append(nbf.v4.new_code_cell(text))

    # put %matplotlib inline in first code cell
    if magic.strip():
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                break
        cell['source'] = magic + '\n' + cell['source']
    nbf.write(nb, out)


def execute_notebook(in_, out):
    in_, out = [str(pathlib.Path(z).resolve())
                for z in (in_, out)]
    # when using relative paths, jupyter-nbconvert
    # resolves 'out' relative to 'in_', not to current dir
    os.system(
        'jupyter-nbconvert --to notebook --execute '
        '--output {out} {in_}'
        .format(in_=in_, out=out)
        )


def quickguide_make(execute=True):
    """
    Generate notebook from ./doc/quickguide.rst
    """
    pathlib.Path('./dist').mkdir(exist_ok=True)
    in_ = './doc/quickguide.rst'
    out_code = './dist/quickguide_code.ipynb'
    out_executed = './dist/quickguide.ipynb'
    header = (
        '*This file, part of the* [SdePy](https://github.com/sdepy/sdepy) '
        '*package* v{},\n'.format(sdepy.__version__),
        '*was automatically generated from* `{}`\n'.format(in_),
        '\n',
        '-----------------------------------------------\n')
    getnotebook(in_=in_, out=out_code,
                skip=1,  # skip initial '==========='
                header=header,
                magic='%matplotlib inline\n'
                "%config InlineBackend.figure_format = 'png'",
                to_markdown=lambda x: x.replace('::', ':'),
                to_code=lambda x:
                x.replace('# doctest: +SKIP', '').rstrip() + '\n',
                )
    if execute:
        execute_notebook(in_=out_code,
                         out=out_executed)


def inspect_warnings(action='error'):
    """
    Execute all tests in sdepy.tests module,
    printing warnings if any.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action)
        for key, item in globals().items():
            if key[:5] == 'test_':
                print('\nrunning', key)
                item()


# minimal setup as a command line script
if __name__ == '__main__':
    for cmd in sys.argv[1:]:
        eval(cmd)
    sys.exit(0)
