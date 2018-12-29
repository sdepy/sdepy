"""
=================================
SHORTCUTS FOR INTERACTIVE
SESSIONS (INCLUDING KFUNCS SETUP)
=================================

*  Shortcuts for stochasticity sources and process realizations,
*  ``kfunc`` wrappings, based on _config.KFUNC setting
"""

from . import _config
from .infrastructure import *
from .integration import *
from .analytical import *
from .kfun import *


###############################
# Shortcuts and kfunc wrappings
# for sources and processes
###############################

if _config.KFUNC == 'all':
    kfunc_all = kfunc

    def kfunc_short(f): return f

elif _config.KFUNC == 'shortcuts':

    def kfunc_all(f): return f

    kfunc_short = kfunc

elif _config.KFUNC is None:

    def kfunc_all(f): return f

    def kfunc_short(f): return f

else:
    raise ValueError(
        'unsupported value {} for KFUNC configuration parameter'
        .format(_config.KFUNC)
        )

# handle sources and processes full names
#   'all': wrap as kfuncs
#   None: do nothing
wiener_source = kfunc_all(wiener_source)
poisson_source = kfunc_all(poisson_source)
cpoisson_source = kfunc_all(cpoisson_source)

odd_wiener_source = kfunc_all(odd_wiener_source)
even_poisson_source = kfunc_all(even_poisson_source)
even_cpoisson_source = kfunc_all(even_cpoisson_source)

true_wiener_source = kfunc_all(true_wiener_source)
true_poisson_source = kfunc_all(true_poisson_source)
true_cpoisson_source = kfunc_all(true_cpoisson_source)

wiener_process = kfunc_all(wiener_process)
lognorm_process = kfunc_all(lognorm_process)
ornstein_uhlenbeck_process = kfunc_all(ornstein_uhlenbeck_process)
hull_white_process = kfunc_all(hull_white_process)
hull_white_1factor_process = kfunc_all(hull_white_1factor_process)
cox_ingersoll_ross_process = kfunc_all(cox_ingersoll_ross_process)
full_heston_process = kfunc_all(full_heston_process)
heston_process = kfunc_all(heston_process)
jumpdiff_process = kfunc_all(jumpdiff_process)
merton_jumpdiff_process = kfunc_all(merton_jumpdiff_process)
kou_jumpdiff_process = kfunc_all(kou_jumpdiff_process)

# handle shortcuts for sources and processes:
#   'all': shortcuts refer to wrapped classes, kfunc_short does nothing
#   None: shortcuts refer to unwrapped classes, kfunc_short wraps them
dw = kfunc_short(wiener_source)
dn = kfunc_short(poisson_source)
dj = kfunc_short(cpoisson_source)

odd_dw = kfunc_short(odd_wiener_source)
even_dn = kfunc_short(even_poisson_source)
even_dj = kfunc_short(even_cpoisson_source)

true_dw = kfunc_short(true_wiener_source)
true_dn = kfunc_short(true_poisson_source)
true_dj = kfunc_short(true_cpoisson_source)

wiener = kfunc_short(wiener_process)
lognorm = kfunc_short(lognorm_process)
oruh = kfunc_short(ornstein_uhlenbeck_process)
hwff = kfunc_short(hull_white_process)
hw1f = kfunc_short(hull_white_1factor_process)  # with 'f' factors
cir = kfunc_short(cox_ingersoll_ross_process)
# returns both x (the process) and y (its stochastic volatility)
heston_xy = kfunc_short(full_heston_process)
heston = kfunc_short(heston_process)
jumpdiff = kfunc_short(jumpdiff_process)
mjd = kfunc_short(merton_jumpdiff_process)
kou = kfunc_short(kou_jumpdiff_process)


##########################
# kfunc wrappings
# for analytical functions
##########################

# handle analytical functions
if _config.KFUNC in ('all', 'shortcuts'):
    # no wrapping if KFUNC is None

    wiener_mean = kfunc(nvar=1)(wiener_mean)
    wiener_var = kfunc(nvar=1)(wiener_var)
    wiener_std = kfunc(nvar=1)(wiener_std)
    wiener_pdf = kfunc(nvar=2)(wiener_pdf)
    wiener_cdf = kfunc(nvar=2)(wiener_cdf)
    wiener_chf = kfunc(nvar=2)(wiener_chf)

    lognorm_mean = kfunc(nvar=1)(lognorm_mean)
    lognorm_var = kfunc(nvar=1)(lognorm_var)
    lognorm_std = kfunc(nvar=1)(lognorm_std)
    lognorm_pdf = kfunc(nvar=2)(lognorm_pdf)
    lognorm_cdf = kfunc(nvar=2)(lognorm_cdf)
    lognorm_log_chf = kfunc(nvar=2)(lognorm_log_chf)

    oruh_mean = kfunc(nvar=1)(oruh_mean)
    oruh_var = kfunc(nvar=1)(oruh_var)
    oruh_std = kfunc(nvar=1)(oruh_std)
    oruh_pdf = kfunc(nvar=2)(oruh_pdf)
    oruh_cdf = kfunc(nvar=2)(oruh_cdf)

    hw1f_mean = kfunc(nvar=1)(hw1f_mean)
    hw1f_var = kfunc(nvar=1)(hw1f_var)
    hw1f_std = kfunc(nvar=1)(hw1f_std)
    hw1f_pdf = kfunc(nvar=2)(hw1f_pdf)
    hw1f_cdf = kfunc(nvar=2)(hw1f_cdf)

    hw2f_mean = kfunc(nvar=1)(hw2f_mean)
    hw2f_var = kfunc(nvar=1)(hw2f_var)
    hw2f_std = kfunc(nvar=1)(hw2f_std)
    hw2f_pdf = kfunc(nvar=2)(hw2f_pdf)
    hw2f_cdf = kfunc(nvar=2)(hw2f_cdf)

    cir_mean = kfunc(nvar=1)(cir_mean)
    cir_var = kfunc(nvar=1)(cir_var)
    cir_std = kfunc(nvar=1)(cir_std)
    cir_pdf = kfunc(nvar=2)(cir_pdf)

    heston_log_mean = kfunc(nvar=1)(heston_log_mean)
    heston_log_var = kfunc(nvar=1)(heston_log_var)
    heston_log_std = kfunc(nvar=1)(heston_log_std)
    heston_log_chf = kfunc(nvar=2)(heston_log_chf)
    heston_log_pdf = kfunc(nvar=2)(heston_log_pdf)

    mjd_mean = kfunc(nvar=1)(mjd_mean)
    mjd_log_chf = kfunc(nvar=2)(mjd_log_chf)
    mjd_log_pdf = kfunc(nvar=2)(mjd_log_pdf)

    kou_mean = kfunc(nvar=1)(kou_mean)
    kou_log_chf = kfunc(nvar=2)(kou_log_chf)
    kou_log_pdf = kfunc(nvar=2)(kou_log_pdf)

    bsd1d2 = kfunc(nvar=2)(bsd1d2)
    bscall = kfunc(nvar=2)(bscall)
    bscall_delta = kfunc(nvar=2)(bscall_delta)
    bsput = kfunc(nvar=2)(bsput)
    bsput_delta = kfunc(nvar=2)(bsput_delta)

# clean up the namespace
del kfunc_all, kfunc_short
