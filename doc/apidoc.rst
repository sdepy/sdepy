#################
API Documentation
#################


Overview
========

.. automodule:: sdepy


Infrastructure
==============

.. autosummary::
   :toctree: generated/

   process
   montecarlo


Stochasticity Sources
=====================

.. autosummary::
   :toctree: generated/

   source
   wiener_source
   poisson_source
   cpoisson_source
   odd_wiener_source
   even_poisson_source
   even_cpoisson_source
   true_source
   true_wiener_source
   true_poisson_source
   true_cpoisson_source
   norm_rv
   uniform_rv
   exp_rv
   double_exp_rv
   rvmap


SDE Integration Framework
=========================

.. autosummary::
   :toctree: generated/

   paths_generator
   integrator
   SDE
   SDEs
   integrate


Stochastic Processes
====================

.. autosummary::
   :toctree: generated/

   wiener_process
   lognorm_process
   ornstein_uhlenbeck_process
   hull_white_process
   hull_white_1factor_process
   cox_ingersoll_ross_process
   full_heston_process
   heston_process
   jumpdiff_process
   merton_jumpdiff_process
   kou_jumpdiff_process
   wiener_SDE
   lognorm_SDE
   ornstein_uhlenbeck_SDE
   hull_white_SDE
   cox_ingersoll_ross_SDE
   full_heston_SDE
   heston_SDE
   jumpdiff_SDE
   merton_jumpdiff_SDE
   kou_jumpdiff_SDE


Analytical Results
==================

.. autosummary::
   :toctree: generated/

   wiener_mean
   wiener_var
   wiener_std
   wiener_pdf
   wiener_cdf
   wiener_chf

   lognorm_mean
   lognorm_var
   lognorm_std
   lognorm_pdf
   lognorm_cdf
   lognorm_log_chf

   oruh_mean
   oruh_var
   oruh_std
   oruh_pdf
   oruh_cdf

   hw2f_mean
   hw2f_var
   hw2f_std
   hw2f_pdf
   hw2f_cdf

   cir_mean
   cir_var
   cir_std
   cir_pdf

   heston_log_mean
   heston_log_var
   heston_log_std
   heston_log_pdf
   heston_log_chf

   mjd_log_pdf
   mjd_log_chf

   kou_mean
   kou_log_pdf
   kou_log_chf

   bsd1d2
   bscall
   bscall_delta
   bsput
   bsput_delta


Shortcuts
=========

Stochasticity sources and preset processes may be addressed
using the following shortcuts:

====================================  ==============
Full name                             Shortcut
====================================  ==============
``wiener_source``                       ``dw``
``poisson_source``                      ``dn``
``cpoisson_source``                     ``dj``
``odd_wiener_source``                   ``odd_dw``
``even_poisson_source``                 ``even_dn``
``even_cpoisson_source``                ``even_dj``
``true_wiener_source``                  ``true_dw``
``true_poisson_source``                 ``true_dn``
``true_cpoisson_source``                ``true_dj``
``wiener_process``                      ``wiener``
``lognorm_process``                     ``lognorm``
``ornstein_uhlenbeck_process``          ``oruh``
``hull_white_process``                  ``hwff``
``hull_white_1factor_process``          ``hw1f``
``cox_ingersoll_ross_process``          ``cir``
``full_heston_process``                 ``heston_xy``
``heston_process``                      ``heston``
``jumpdiff_process``                    ``jumpdiff``
``merton_jumpdiff_process``             ``mjd``
``kou_jumpdiff_process``                ``kou``
====================================  ==============

Shortcuts have been wrapped as "kfuncs", objects with managed
keyword arguments (see ``kfunc`` decorator documentation below).

Analytical results are named according to the shortcut
of the corresponding process (e.g. ``lognorm_mean``, ``lognorm_cdf`` etc.
from the ``lognorm`` shortcut) and are wrapped as kfuncs as well.

.. autosummary::
   :toctree: generated/

   kfunc
   iskfunc

   
.. Testing
   =======
   
.. include:: testing.rst