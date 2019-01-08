Testing
=======

Tests have been set up within the ``numpy.testing`` framework. 
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
   with both 100 and 100000 paths against a fixed random seed.
   Numerical integration results for the mean, standard deviation, 
   probability distribution, and/or characteristic function are compared 
   against their exact values computed analytically from the process
   parameters. Comparisons are then plotted and visually inspected, and
   the occasional larger than usual deviation is manually checked to be
   statistically acceptable, i.e. only so few standard deviations
   off the mark. The plots and the average and maximum errors are recorded 
   in png and text files located in the ``./tests/cfr`` directory, relative 
   to the package home directory where ``sdepy.__file__`` is located.
   
-  Each time ``sdepy.test('full')`` is invoked, to keep testing times
   manageable and the testing procedure uninvasive, tests are run
   with 100 paths against the same fixed random seed, without plotting
   or storing results. The realized errors
   are then compared and checked against the expected errors,
   as distributed with the package and stored in the 
   ``./tests/cfr`` directory.
   
Note that the tests rely on the reproducibility of expected errors, once
random numbers have been seeded with ``np.random.seed()``, across platforms
and versions of Python, NumPy and SciPy.

In order to reproduce the full tests and inspect the
graphs, change the following configuration settings in the file of the
``sdepy._config`` subpackage (private, not part of the API, may change
in the future)::

	PLOT = True
	SAVE_ERRORS = True
	QUANT_TEST_MODE = 'HD'
	
With these settings, tests are run with 100000 paths, and realized errors and
plots are stored in the ``./tests/cfr`` directory. In case some tests fail,
to carry out the whole procedure and get the failing errors and plots, set in
the same configuration file::
		
	QUANT_TEST_FAIL = False
