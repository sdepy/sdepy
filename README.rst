=====
SdePy
=====
|ci|  |codecov|  |readthedocs|

The SdePy package provides tools to state and numerically
integrate Ito Stochastic Differential Equations (SDEs), including equations
with time-dependent parameters, time-dependent correlations, and
stochastic jumps, and to compute with, and extract statistics from,
their realized paths.

Several preset processes are provided, including lognormal,
Ornstein-Uhlenbeck, Hull-White n-factor, Heston, and jump-diffusion processes.

Computations are fully vectorized across paths, via NumPy and SciPy,
making live sessions with 100000 paths reasonably fluent
on single cpu hardware.

----------

This package came out of practical need, so expect a flexible tool
that gets real-life things done. On the other hand, not every part of it
is clean and polished, so expect rough edges, and the occasional
bug (please report!).

Developers are committed to the stability of the public API,
here again out of practical need to safeguard dependencies.

----------
Start here
----------

-  `Installation           <https://pypi.org/project/sdepy>`_: ``pip install sdepy``
-  `Quick Guide            <https://sdepy.readthedocs.io/en/latest/intro.html#id2>`_
   (as `notebook           <https://nbviewer.jupyter.org/github/sdepy/sdepy-doc/blob/master/quickguide.ipynb>`_)
-  `Documentation          <https://sdepy.readthedocs.io/en/latest>`_
   (as `pdf                <https://sdepy.readthedocs.io/_/downloads/en/latest/pdf/>`_)
-  `Source                 <https://github.com/sdepy/sdepy>`_
-  `License                <https://github.com/sdepy/sdepy/blob/master/LICENSE.txt>`_
-  `Bug Reports            <https://github.com/sdepy/sdepy/issues>`_


.. |readthedocs| image:: https://readthedocs.org/projects/sdepy/badge/?version=latest
   :target: https://sdepy.readthedocs.io
   :alt: Documentation Status

.. |ci| image:: https://github.com/sdepy/sdepy/workflows/CI/badge.svg?branch=master
    :target: https://github.com/sdepy/sdepy/actions

.. |codecov| image:: https://codecov.io/gh/sdepy/sdepy/blob/master/graph/badge.svg
  :target: https://codecov.io/gh/sdepy/sdepy
