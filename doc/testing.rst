#######
Testing
#######


Tests have been set up within the ``numpy.testing`` framework,
and require the ``pytest`` package to be installed.
To launch tests, invoke ``sdepy.test()`` or ``sdepy.test('full')``.

The testing subpackage ``sdepy.tests`` was written in pursuit of
the following goals:

-  Maximize *case* coverage, by exposing the package functions and methods
   to a plurality of different input shapes, values, data types etc.,
   and of different combinations thereof, as may be encountered in practice.

-  Provide a quantitative validation of the algorithms, functions and
   processes covered in ``sdepy``.

-  Keep dependencies of the test code on the adopted testing framework
   to a bare minimum.

Most often, a number of testing cases is declared as a list or lists of
classes and inputs, a general testing procedure is set up, and
the latter is iteratively applied to the former. Unfortunately,
all this resulted in a thinly documented (if at all), hard to read, and hard
to maintain testing code base - sorry about that.

The quantitative validation of the package, via tests marked as ``'slow'`` and
``'quant'``, is done in two steps:

-  To validate a ``sdepy`` release, tests are run
   with 100_000 or more paths.
   Numerical integration results for the mean, standard deviation,
   probability distribution, and/or characteristic function are compared
   against their exact values computed analytically from the process
   parameters. Comparisons are then plotted and visually inspected, and
   the occasional larger than usual deviation is manually checked to be
   statistically acceptable, i.e. only so few standard deviations
   off the mark.

-  Each time ``sdepy.test('full')`` is invoked, to keep testing times
   manageable and the testing procedure non invasive, tests are run
   with 100 paths, using NumPy legacy random generation with a fixed seed,
   the realized errors are then compared and checked against the expected
   errors, as distributed with the package and stored in the
   ``./tests/cfr`` directory. Note that such default tests rely on
   the stable reproducibility of expected errors,
   across platforms and versions of Python, NumPy and SciPy.

In order to run tests using current NumPy random generation,
with a set number of paths (200_000 in the example below),
and inspect graphs and realized errors as saved in the current directory,
use the following statement (non backward compatible changes
to the testing interface may occur in the future,
without prior warning)::

   sdepy.test('full', rng=numpy.random.default_rng(),
              paths=200_000, plot=True, outdir='.')


For further details, see ``sdepy.test`` documentation:

.. autosummary::
   :toctree: generated/

   sdepy.test
