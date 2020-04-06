"""
=====================================
FORMAL TESTS ON STOCHASTICITY SOURCES
=====================================
"""
from .shared import *
from numpy import sqrt
import scipy.stats

source = sp.source
wiener_source = sp.wiener_source
poisson_source = sp.poisson_source
cpoisson_source = sp.cpoisson_source

odd_wiener_source = sp.odd_wiener_source
even_poisson_source = sp.even_poisson_source
even_cpoisson_source = sp.even_cpoisson_source

true_wiener_source = sp.true_wiener_source
true_poisson_source = sp.true_poisson_source
true_cpoisson_source = sp.true_cpoisson_source

dw = sp.dw
odd_dw = sp.odd_dw
true_dw = sp.true_dw
dn = sp.dn
even_dn = sp.even_dn
true_dn = sp.true_dn
dj = sp.dj
even_dj = sp.even_dj
true_dj = sp.true_dj

norm_rv = sp.norm_rv
uniform_rv = sp.uniform_rv
exp_rv = sp.exp_rv
double_exp_rv = sp.double_exp_rv

true_source = sp.true_source
true_wiener_source = sp.true_wiener_source


# ---------------------
# sources general tests
# ---------------------

# main test
def test_source_general():
    np.random.seed(SEED)

    # do cases
    paths = [10]
    float_dtype = [None, np.float32]
    int_dtype = [int, np.int16]

    source_and_params = [
        (wiener_source, dict(vshape=())),
        (wiener_source, dict(vshape=(2, 3))),
        (wiener_source, dict(vshape=(2,), corr=((1., -.5), (-.5, 1.)))),
        (wiener_source, dict(vshape=(3, 2,), corr=((1., -.5), (-.5, 1.)))),
        (true_wiener_source, dict(vshape=())),
        (true_wiener_source, dict(vshape=(2, 3))),
        (true_wiener_source, dict(vshape=(2,), corr=((1., -.5), (-.5, 1.)))),
        (true_wiener_source, dict(vshape=(3, 2,),
                                  corr=((1., -.5), (-.5, 1.)))),
        (odd_wiener_source, dict(vshape=())),
        (odd_wiener_source, dict(vshape=(2, 3))),
        (odd_wiener_source, dict(vshape=(2,), corr=((1., -.5), (-.5, 1.)))),
        (odd_wiener_source, dict(vshape=(3, 2,), corr=((1., -.5), (-.5, 1.)))),
        (poisson_source, dict(vshape=())),
        (poisson_source, dict(vshape=(2,), lam=1.)),
        (poisson_source, dict(vshape=(2,), lam=((1.,), (2.,)))),
        (poisson_source, dict(vshape=(3, 2), lam=((1.,), (2.,)))),
        (poisson_source, dict(vshape=(2, 3, 5),
                              lam=np.arange(15.).reshape(3, 5, 1))),
        (cpoisson_source, dict(vshape=())),
        (cpoisson_source, dict(vshape=(3, 2), lam=((1.,), (2.,)))),
        (cpoisson_source,
         dict(vshape=(3, 2), lam=((1.,), (2.,)),
              y=norm_rv(a=1., b=5.), ptype=np.int16)),
        (cpoisson_source,
         dict(vshape=(3, 2), lam=((1.,), (2.,)),
              y=uniform_rv(a=1., b=5.))),
        (cpoisson_source,
         dict(vshape=(3, 2), lam=((1.,), (2.,)),
              y=double_exp_rv(pa=.3, a=1., b=5.))),
        (cpoisson_source,
         dict(vshape=(3, 2), lam=((1.,), (2.,)),
              y=scipy.stats.trapz(c=0.2, d=0.4))),
        (cpoisson_source,
         dict(vshape=(3, 2), lam=((1.,), (2.,)),
              y=scipy.stats.expon(scale=2.))),
        (true_cpoisson_source, dict(vshape=())),
        (true_cpoisson_source, dict(vshape=(3, 2), lam=((1.,), (2.,)))),
        (true_cpoisson_source,
         dict(vshape=(3, 2), lam=((1.,), (2.,)),
              y=norm_rv(a=1., b=5.))),
        (true_cpoisson_source,
         dict(vshape=(3, 2), lam=((1.,), (2.,)),
              y=uniform_rv(a=1., b=5.))),
        (true_cpoisson_source,
         dict(vshape=(3, 2), lam=((1.,), (2.,)),
              y=double_exp_rv(pa=.3, a=1., b=5.))),
        (true_cpoisson_source,
         dict(vshape=(3, 2), lam=((1.,), (2.,)),
              y=scipy.stats.trapz(c=0.2, d=0.4))),
        (true_cpoisson_source,
         dict(vshape=(3, 2), lam=((1.,), (2.,)),
              y=scipy.stats.expon(scale=2.)))
        ]

    source_and_params_int = [
        (poisson_source, dict(vshape=())),
        (poisson_source, dict(vshape=(2,), lam=1.)),
        (poisson_source, dict(vshape=(2,), lam=((1.,), (2.,)))),
        (poisson_source, dict(vshape=(3, 2), lam=((1.,), (2.,)))),
        (poisson_source, dict(vshape=(2, 3, 5),
                              lam=np.arange(15.).reshape(3, 5, 1))),
        (true_poisson_source, dict(vshape=())),
        (true_poisson_source, dict(vshape=(2,), lam=1.)),
        (true_poisson_source, dict(vshape=(2,), lam=((1.,), (2.,)))),
        (true_poisson_source, dict(vshape=(3, 2), lam=((1.,), (2.,)))),
        (true_poisson_source, dict(vshape=(2, 3, 5),
                                   lam=np.arange(15.).reshape(3, 5, 1))),
        ]

    source_and_params_shortcuts = [
        (dw, dict(vshape=())),
        (dw, dict(vshape=(3, 2,), corr=((1., -.5), (-.5, 1.)))),
        (odd_dw, dict(vshape=())),
        (odd_dw, dict(vshape=(3, 2,), corr=((1., -.5), (-.5, 1.)))),
        (true_dw, dict(vshape=())),
        (true_dw, dict(vshape=(3, 2,), corr=((1., -.5), (-.5, 1.)))),
        (dj, dict(vshape=())),
        (dj, dict(vshape=(3, 2), lam=((1.,), (2.,)),
                  y=scipy.stats.expon(scale=2.))),
        (even_dj, dict(vshape=())),
        (even_dj, dict(vshape=(3, 2), lam=((1.,), (2.,)),
                       y=scipy.stats.expon(scale=2.))),
        (true_dj, dict(vshape=())),
        (true_dj, dict(vshape=(3, 2), lam=((1.,), (2.,)),
                       y=scipy.stats.expon(scale=2.))),
        ]

    source_and_params_shortcuts_int = [
        (dn, dict(vshape=())),
        (dn, dict(vshape=(2, 3, 5),
                  lam=np.arange(15.).reshape(3, 5, 1))),
        (even_dn, dict(vshape=())),
        (even_dn, dict(vshape=(2, 3, 5),
                       lam=np.arange(15.).reshape(3, 5, 1))),
        (true_dn, dict(vshape=())),
        (true_dn, dict(vshape=(2, 3, 5),
                       lam=np.arange(15.).reshape(3, 5, 1))),
        ]

    do(source_general, paths, float_dtype, source_and_params)
    do(source_general, paths, int_dtype, source_and_params_int)

    do(source_general, paths, float_dtype, source_and_params_shortcuts)
    do(source_general, paths, int_dtype, source_and_params_shortcuts_int)


# cases
def source_general(paths, dtype, source_and_params):
    cls, params = source_and_params

    src = cls(paths=paths, dtype=dtype, **params)
    assert_(src.paths == paths)
    assert_(src.vshape == params['vshape'])

    for dt in [2., (5.,), (0, 1.), np.linspace(0.5, 4., 12),
               np.arange(30.).reshape(2, 3, 5)]:
        s = src(0*np.asarray(dt), dt)
        assert_(s.shape == np.asarray(dt).shape +
                src.vshape + (src.paths,))
        assert_(s.dtype == dtype)

    if isinstance(src, true_source):
        for t in [2., (5.,), (0, 1.), np.linspace(0.5, 4., 12),
                  np.arange(30.).reshape(2, 3, 5)]:
            s = src(t)
            assert_(s.shape == np.asarray(t).shape +
                    src.vshape + (src.paths,))
            assert_(s.dtype == dtype)
            s = src(t, np.asarray(t)/10)
            sz = src.size
            assert_(sz > 1)
            st = src.t
            assert_(len(st) > 1)
    elif isinstance(src, source):
        assert_(src.size == 0)
        assert_(src.t.size == 0)
        assert_raises(TypeError, lambda: src(0.))
    else:
        assert_(False)


# ---------------------
# source specific tests
# ---------------------

# main test
def test_source_specific():
    np.random.seed(SEED)

    # wiener tests
    src = wiener_source(vshape=3, paths=5)
    assert_(src(0., 1.).shape == (3, 5))
    assert_raises(ValueError, wiener_source,
                  vshape=3, paths=10, corr=((1., .5), (.5, 1.)))

    # antithetics tests
    for src_class, sign in ((odd_wiener_source, -1),
                            (even_poisson_source, 1),
                            (even_cpoisson_source, 1)):
        src = src_class(vshape=(2, 3), paths=10)
        s = src(0., 1.)
        assert_array_equal(s[..., :5], sign*s[..., 5:])
        assert_raises(ValueError, src_class, paths=3)

    # poission tests
    assert_raises(ValueError, poisson_source,
                  vshape=(3, 2), lam=(1., 2., 3.))

    # compound poisson tests
    val = 2.

    class const_rv:
        def rvs(self, size):
            return np.full(size, fill_value=val)

    src = cpoisson_source(lam=1., paths=100, ptype=np.int16,
                          y=const_rv())
    s, n = src(0, 1), src.dn_value
    assert_allclose(n*val, s, rtol=eps(s.dtype))
    s, n = src(0, 100), src.dn_value
    assert_allclose(n*val, s, rtol=eps(s.dtype))

    # true sources tests
    for cls in (true_wiener_source, true_poisson_source, true_cpoisson_source,
                true_dw, true_dn, true_dj):
        src = cls(vshape=(2, 3), paths=10, t0=1., z0=2.)
        size1 = src.size
        assert_((src(1.) == 2.).all())
        s = src(2.)
        assert_(src.size >= 2*size1)
        assert_array_equal(src(2.), s)
        for dt in [-0.1, -1e-6, 0, 1e-6, 0.1]:
            z = src(2 + dt)
            w = src(2) + src(2, dt)
            assert_allclose(z, w, rtol=eps(z.dtype))
            if cls is true_wiener_source:
                nsigma = 5
                assert_((np.abs(z-s) <= sqrt(np.abs(dt))*nsigma).all())
        assert_(src[0, 0](2.).shape == (10,))
        assert_(src[:, :2](2., .1).shape == (2, 2, 10))
        assert_allclose(src[:1](2., .1), src(2., .1)[:1], rtol=eps(src.dtype))
        src2 = src[:, :, np.newaxis]
        assert_(src2.paths == src.paths)
        assert_(src2.dtype == src.dtype)
        assert_(src2.vshape == (2, 3, 1))
        assert_(src2(1).shape == (2, 3, 1, 10))

    # more true source tests
    for cls in (true_poisson_source, true_cpoisson_source,
                true_dn, true_dj):
        # check poisson sources do not store points between constant values
        src = cls(lam=0.)
        src(1.)
        size = src.size
        src(np.linspace(0, 1, 10))
        assert_(src.size == size)


# ------------------------------------------
# tests for time-dependent source parameters
# ------------------------------------------

# sources with time-dependent parameters are tested
# in test_processes, as sources of processes with
# time-dependent parameters

def corr(a, b=None):
    if b is None:
        a, b = a[0], a[1]
    cov = np.cov(a, b)
    n = cov.shape[0]
    return np.asarray(
        [[cov[i, j]/sqrt(cov[i, i]*cov[j, j]) for i in range(n)]
         for j in range(n)])


@slow
@quant
def test_source_true_wiener():

    # test exactness of true_wiener with correlations linarly
    # dependent on t
    rho0, irho0 = (.3, lambda t, t0: .3)
    rho1, irho1 = (lambda t: -0.1 - t/6, lambda t, t0: -0.1 - (t + t0)/12)
    rho2, irho2 = (lambda t: 0.1 + t/6, lambda t, t0: 0.1 + (t + t0)/12)
    rho3, irho3 = (None, lambda t, t0: 0)

    np.random.seed(SEED)
    for t0 in (0, 1):
        for rho, irho in ((rho0, irho0), (rho1, irho1),
                          (rho2, irho2), (rho3, irho3)):
            for vshape, i in zip(((2,), (3, 2)),
                                 (np.index_exp[...], np.index_exp[0])):
                tw = true_dw(rho=rho, vshape=vshape, paths=100000, t0=t0)
                for tt in (((6, 2, 4), (-6, -2, -4))):
                    tw(tt)
                    for s in tt:
                        x = tw(s)[i]
                        c = corr(x)
                        v = np.var(x)
                        assert_allclose(c[0, 1], irho(s, t0),
                                        rtol=0.02, atol=0.01)
                        assert_allclose(v, abs(s - t0), rtol=0.02, atol=0.01)
                        print('.', sep='', end='')
                for s1, s2 in ((4, 6), (2, 6), (2, 4),
                               (-4, -6), (-2, -6), (-2, -4)):
                    c = corr((tw(s2) - tw(s1))[i][0], tw(s1)[i][0])
                    v = np.var((tw(s2) - tw(s1))[i][0])
                    assert_allclose(c[0, 1], 0, rtol=0.02, atol=0.01)
                    assert_allclose(v, abs(s2 - s1), rtol=0.02, atol=0.01)
                    print('.', sep='', end='')
