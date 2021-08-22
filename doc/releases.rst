
#############
Release Notes
#############

This section reports notable changes and additions in SdePy releases.

Not all releases are covered. For further information,
and a complete list of releases, please refer to the SdePy repository on
`GitHub <https://github.com/sdepy/sdepy/releases>`_.


===========
SdePy 1.2.0
===========


**New in this release**

SdePy was upgraded to the current NumPy pseudo-random numbers generation
framework, based on the PCG-64 algorithm.
The following additions were made:

- SDEs and stochasticity source classes now accept upon instantiation
  a new optional parameter ``rng``, to locally set the
  random number generator used by each instance.
  Such ``rng`` is expected to expose the interface
  used by ``numpy.random.Generator`` and ``numpy.random.RandomState``
  instances.

- In case no ``rng`` parameter is given, random number generation
  is delegated to a global ``numpy.random.default_rng()`` object,
  created upon import: absent any user intervention,
  unmodified pre-existing code will rely on PCG-64
  random number generation after upgrading to current
  NumPy and SdePy releases.

- To access the stably reproducible NumPy legacy random generation,
  it is recommended to instantiate sources and SDEs with the
  ``rng=numpy.random.RandomState(SEED)`` parameter. As a result,
  each object is served the random numbers stream formerly obtained
  from the global NumPy random state after a ``numpy.random.seed(SEED)``
  call.

- Compatibility with legacy NumPy versions was maintained, if now deprecated:
  in such case, SdePy silently falls back on the legacy
  global NumPy random state.

The present SdePy version safeguards functional compatibility with
code written for previous versions, note however that, if such code is run
with current NumPy and SdePy versions, it will
**break pseudo-random numbers reproducibility**:

- ``numpy.random.seed`` calls no longer affect SdePy objects.

- To reproduce the default output of former SdePy versions,
  ``numpy.random.seed(SEED)`` statements may be replaced
  with the following assignment::

     sdepy.infrastructure.default_rng = numpy.random.RandomState(SEED)

  Any other use of such global variable should rather be avoided,
  in favor of the ``rng`` keyword.


**Improvements**

- The testing suite was updated, and its interface ``sdepy.test()`` extended,
  for use beyond NumPy legacy random generation.
- GitHub Actions were updated
  accordingly, to perform CI tests using both legacy and current NumPy random
  generation.


**Changes**

Python 3.5 is no longer supported.


===========
SdePy 1.1.0
===========


**New in this release**

- The ``sdepy.process`` class acquired new methods
  ``vmin``, ``vmax``, ``vmean``, ``vvar``, ``vstd``
  to perform summary operations across values, for each
  time point and path.
- A piecewise constant process constructor was added as the
  ``piecewise`` function; it replaces, and improves upon,
  the private ``_piecewise_constant_process`` (undocumented,
  not part of the API), now deprecated.


**Bug-fixes**

- An incompatibility issue of the ``sdepy.process`` class with
  NumPy versions >= 1.16.0 was solved.
- A bug in the ``out`` parameter of process methods
  ``pmin``, ``pmax``, ``tmin``, ``tmax`` was fixed.
  No backward incompatible changes were made against the stated API,
  note however that code relying on the ``out`` parameter of the
  mentioned process methods might need fixing as well.
