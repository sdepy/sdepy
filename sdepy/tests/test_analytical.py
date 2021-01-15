"""
==================================
FORMAL TESTS ON ANALYTICAL RESULTS
==================================
"""
from .shared import *
import math


# -------------------------------
# lists of functions to be tested
# -------------------------------


# restate bsd1d2 as functions returning 1 result
# ----------------------------------------------

def bsd1(K, t, **args):
    return sp.bsd1d2(K, t, **args)[0]


def bsd2(K, t, **args):
    return sp.bsd1d2(K, t, **args)[1]


# all functions excluding hw2f stats and numerically integrated pdfs
# ------------------------------------------------------------------
f_var1 = (
    sp.wiener_mean, sp.wiener_var, sp.wiener_std,
    sp.lognorm_mean, sp.lognorm_var, sp.lognorm_std,
    sp.oruh_mean, sp.oruh_var, sp.oruh_std,
    sp.hw1f_mean, sp.hw1f_var, sp.hw1f_std,
    sp.cir_mean, sp.cir_var, sp.cir_std,
    sp.heston_log_mean, sp.heston_log_var, sp.heston_log_std
)
f_var2 = (
    bsd1, bsd2, sp.bscall, sp.bscall_delta,
    sp.bsput, sp.bsput_delta,
    sp.wiener_pdf, sp.wiener_cdf, sp.wiener_chf,
    sp.lognorm_pdf, sp.lognorm_cdf, sp.lognorm_log_chf,
    sp.oruh_pdf, sp.oruh_cdf,
    sp.hw1f_pdf, sp.hw1f_cdf,
    sp.cir_pdf,
    sp.heston_log_chf,
    sp.mjd_log_chf,
    sp.kou_log_chf
)


# hw2f stats
# -------------------------------
hw2f_var1 = (sp.hw2f_mean, sp.hw2f_var, sp.hw2f_std)
hw2f_var2 = (sp.hw2f_pdf, sp.hw2f_cdf)

# numerically integrated pdfs
# ---------------------------
pdfs_var2 = (sp.heston_log_pdf,
             sp.mjd_log_pdf,
             sp.kou_log_pdf)

# default args for all functions
# ------------------------------
ARGS = {}
ARGS[bsd1] = \
    ARGS[bsd2] = \
    ARGS[sp.bscall] = \
    ARGS[sp.bscall_delta] = \
    ARGS[sp.bsput] = \
    ARGS[sp.bsput_delta] = {'x0': 1.0, 'r': 0.0, 'q': 0.0, 'sigma': 1.0}

ARGS[sp.wiener_mean] = \
    ARGS[sp.wiener_var] = \
    ARGS[sp.wiener_std] = \
    ARGS[sp.wiener_pdf] = \
    ARGS[sp.wiener_cdf] = \
    ARGS[sp.wiener_chf] = {'x0': 0.0, 'mu': 0.0, 'sigma': 1.0}

ARGS[sp.lognorm_mean] = \
    ARGS[sp.lognorm_var] = \
    ARGS[sp.lognorm_std] = \
    ARGS[sp.lognorm_pdf] = \
    ARGS[sp.lognorm_cdf] = \
    ARGS[sp.lognorm_log_chf] = {'x0': 1.0, 'mu': 0.0, 'sigma': 1.0}

ARGS[sp.oruh_mean] = \
    ARGS[sp.oruh_var] = \
    ARGS[sp.oruh_std] = \
    ARGS[sp.oruh_pdf] = \
    ARGS[sp.oruh_cdf] = {'x0': 0.0, 'theta': 0.0, 'k': 1.0, 'sigma': 1.0}

ARGS[sp.hw1f_mean] = \
    ARGS[sp.hw1f_var] = \
    ARGS[sp.hw1f_std] = \
    ARGS[sp.hw1f_pdf] = \
    ARGS[sp.hw1f_cdf] = {'x0': 0.0, 'theta': 0.0, 'k': 1.0, 'sigma': 1.0}

ARGS[sp.hw2f_mean] = \
    ARGS[sp.hw2f_std] = \
    ARGS[sp.hw2f_var] = \
    ARGS[sp.hw2f_pdf] = \
    ARGS[sp.hw2f_cdf] = {'x0': (0.0, 0.0), 'theta': (0.0, 0.0),
                         'k': (1.0, 1.0), 'sigma': (1.0, 1.0),
                         'rho': 0.0}

ARGS[sp.cir_mean] = \
    ARGS[sp.cir_var] = \
    ARGS[sp.cir_std] = \
    ARGS[sp.cir_pdf] = {'x0': 1.0, 'theta': 1.0, 'k': 1.0, 'xi': 1.0}

ARGS[sp.heston_log_mean] = \
    ARGS[sp.heston_log_var] = \
    ARGS[sp.heston_log_std] = \
    ARGS[sp.heston_log_pdf] = \
    ARGS[sp.heston_log_chf] = {'x0': 1.0, 'y0': 1.0, 'mu': 0.0, 'sigma': 1.0,
                               'theta': 1.0, 'k': 1.0, 'xi': 1.0, 'rho': 0.5}

ARGS[sp.mjd_log_pdf] = \
    ARGS[sp.mjd_log_chf] = {'x0': 1.0, 'mu': 0.0, 'sigma': 1.0, 'lam': 1.0,
                            'a': 0.0, 'b': 1.0}

ARGS[sp.kou_log_pdf] = \
    ARGS[sp.kou_log_chf] = {'x0': 1.0, 'mu': 0.0, 'sigma': 1.0, 'lam': 1.0,
                            'pa': 0.5, 'a': 0.5, 'b': 0.5}


# ------------------------
# tests
# ------------------------

def test_warnings():
    if False:
        # raises a warning in python 3.9
        # (used to test runtests.py script with warnings=='fail')
        math.factorial(10.)


def test_analytical():
    np.random.seed(SEED)

    # all functions excluding hw2f stats
    # ----------------------------------

    tshape = [(), (3,), (2, 3), (1, 1, 3), (1, 3, 1)]
    xushape = [(), (3,), (2, 3), (1, 1, 3), (1, 3, 1)]
    params_shape = [(), (3,), (2, 3), (1, 1, 3), (1, 3, 1)]

    # params and rho shaped (), either t or x/u of any shape
    for tshape, xushape in zip((tshape, xushape[:1]), (tshape[:1], xushape)):
        do(analytical, f_var1, tshape, xushape, params_shape[:1],
           nvar=1, hw2f=False)
        do(analytical, f_var2, tshape, xushape, params_shape[:1],
           nvar=2, hw2f=False)

    # t and x/u shaped (), params of any shape
    do(analytical, f_var1, tshape[:1], xushape[:1], params_shape,
       nvar=1, hw2f=False)
    do(analytical, f_var2, tshape[:1], xushape[:1], params_shape,
       nvar=2, hw2f=False)

    # shape combinations
    do(analytical, f_var2, [(3, 1)], [(1, 2)], [()],
       nvar=2, hw2f=False)
    do(analytical, f_var2, [(3, 1)], [(1, 2)], [(3, 2)],
       nvar=2, hw2f=False)
    do(analytical, f_var2, [(2, 1, 1)], [(1, 3, 5)], [()],
       nvar=2, hw2f=False)
    do(analytical, f_var2, [(2, 1, 1)], [(1, 3, 5)], [(2, 1, 5)],
       nvar=2, hw2f=False)

    # hw2f stats
    # -------------------------------

    params_shape = [(), (2,), (3, 2, 1)]

    # params and rho shaped (), either t or x/u of any shape
    for tshape, xushape in zip((tshape, xushape[:1]), (tshape[:1], xushape)):
        do(analytical, hw2f_var1, tshape, xushape, params_shape[:1],
           nvar=1, hw2f=True)
        do(analytical, hw2f_var2, tshape, xushape, params_shape[:1],
           nvar=2, hw2f=True)

    # t and x/u shaped (), params of any shape
    do(analytical, hw2f_var1, tshape[:1], xushape[:1], params_shape,
       nvar=1, hw2f=True)
    do(analytical, hw2f_var2, tshape[:1], xushape[:1], params_shape,
       nvar=2, hw2f=True)

    # shape combinations
    do(analytical, hw2f_var1, [(3, 1)], [(1, 5)], [(2,)],
       nvar=1, hw2f=True)
    do(analytical, hw2f_var2, [(3, 1)], [(1, 5)], [(2,)],
       nvar=2, hw2f=True)
    do(analytical, hw2f_var1, [(3, 1, 7)], [(1, 5, 7)], [(2, 7)],
       nvar=1, hw2f=True)
    do(analytical, hw2f_var2, [(3, 1, 7)], [(1, 5, 7)], [(2, 7)],
       nvar=2, hw2f=True)
    do(analytical, hw2f_var1, [(3, 1, 1)], [(3, 1, 1)], [(3, 2, 1)],
       nvar=1, hw2f=True)
    do(analytical, hw2f_var2, [(3, 1, 1)], [(3, 1, 1)], [(3, 2, 1)],
       nvar=2, hw2f=True)

    # numerically integrated pdfs
    # ---------------------------
    tshape = [(3,)]
    xushape = [()]
    params_shape = [()]
    do(analytical, pdfs_var2, tshape, xushape, params_shape,
       nvar=2, hw2f=False)
    do(analytical, pdfs_var2, xushape, tshape, params_shape,
       nvar=2, hw2f=False)


# do cases
def analytical(f, tshape, xushape, params_shape, nvar, hw2f):
    assert 1 <= nvar <= 2
    args = ARGS[f]

    # generate variables and parameters values
    def make(shape):
        return 0.1*np.random.random(shape)
    t = 1 + make(tshape)
    xu = 1 + make(xushape)
    if hw2f:
        params = {k: default[0] + make(params_shape)
                  for k, default in args.items() if k != 'rho'}
        params['rho'] = 1.
        pass
    else:
        params = {k: default + 0.1*make(params_shape)
                  for k, default in args.items()}

    # handle 1 vs 2 variables
    if nvar == 1:
        var = (t,)
        fvalue_shape = tshape
    else:
        var = (t, xu)
        fvalue_shape = (t + xu).shape

    # compute f with default scalar parameters
    v = f(*var)
    assert_(v.shape == fvalue_shape)
    assert_(np.isfinite(v).all())

    # compute fvalue_shape for non scalar parameters,
    # handling hw2f parameters consisting of couples of values
    if hw2f:
        if len(params_shape) == 1:
            fvalue_shape = fvalue_shape
            params['rho'] = make(params_shape[:-1])
        elif len(params_shape) > 1:
            fvalue_shape = (make(fvalue_shape) +
                            make(params_shape[:-2] + (1,))).shape
            params['rho'] = make(params_shape[:-2] + (1,))
    else:
        fvalue_shape = (make(fvalue_shape) + make(params_shape)).shape

    # compute f and check shape
    v = f(*var, **params)
    assert_(v.shape == fvalue_shape)
    assert_(np.isfinite(v).all())
