"""
====================================
FORMAL TESTS ON PROCESS REALIZATIONS
====================================
"""
from .shared import *

from numpy import sqrt, exp
import scipy.stats

process = sp.process

wiener_source = sp.wiener_source
poisson_source = sp.poisson_source
cpoisson_source = sp.cpoisson_source

odd_wiener_source = sp.odd_wiener_source
even_poisson_source = sp.even_poisson_source
even_cpoisson_source = sp.even_cpoisson_source

true_wiener_source = sp.true_wiener_source
true_poisson_source = sp.true_poisson_source
true_cpoisson_source = sp.true_cpoisson_source

norm_rv = sp.norm_rv
uniform_rv = sp.uniform_rv
exp_rv = sp.exp_rv
double_exp_rv = sp.double_exp_rv
rvmap = sp.rvmap

integrator = sp.integrator
integrate = sp.integrate
iskfunc = sp.iskfunc

wiener_process = sp.wiener_process
lognorm_process = sp.lognorm_process
ornstein_uhlenbeck_process = sp.ornstein_uhlenbeck_process
hull_white_process = sp.hull_white_process
hull_white_1factor_process = sp.hull_white_1factor_process
cox_ingersoll_ross_process = sp.cox_ingersoll_ross_process
heston_process = sp.heston_process
full_heston_process = sp.full_heston_process
jumpdiff_process = sp.jumpdiff_process
merton_jumpdiff_process = sp.merton_jumpdiff_process
kou_jumpdiff_process = sp.kou_jumpdiff_process

wiener = sp.wiener
lognorm = sp.lognorm
oruh = sp.oruh
hwff = sp.hwff
hw1f = sp.hw1f
cir = sp.cir
heston = sp.heston
heston_xy = sp.heston_xy
jumpdiff = sp.jumpdiff
mjd = sp.mjd
kou = sp.kou


# -----------------------
# processes general tests
# -----------------------

all_cls = (wiener_process, lognorm_process,
           ornstein_uhlenbeck_process,
           hull_white_process, hull_white_1factor_process,
           cox_ingersoll_ross_process, full_heston_process, heston_process,
           jumpdiff_process, merton_jumpdiff_process,
           kou_jumpdiff_process)

all_shortcuts = (wiener, lognorm,
                 oruh,
                 hwff, hw1f,
                 cir, heston, heston_xy,
                 jumpdiff, mjd,
                 kou)


# enumerate test cases with constant parameters and launch tests
# --------------------------------------------------------------
def test_processes():
    np.random.seed(SEED)

    # setup some parameter test values
    # --------------------------------
    CM2 = ((1, .2), (.2, 1))
    CM3 = ((1, .2, -.3), (.2, 1, .1), (-.3, .1, 1))
    CM6 = np.eye(6) + 0.1*np.random.random((6, 6))
    CM6 = (CM6 + CM6.T)/2

    # processes to act as sources
    Z2 = wiener_process(sigma=.1, paths=11, vshape=2
                        )(np.linspace(0, 1, 100))
    Z23 = wiener_process(sigma=.1, paths=11, vshape=(2, 3)
                         )(np.linspace(0, 1, 100))
    Z32 = wiener_process(sigma=.1, paths=11, vshape=(3, 2)
                         )(np.linspace(0, 1, 100))

    # a 'bare' object with source protocol
    def S2(s, ds):
        return np.random.normal(size=(2, 11))*sqrt(np.abs(ds))

    S2.vshape = (2,)
    S2.paths = 11

    # a generic scipy random variable
    rv = scipy.stats.trapz(c=.1, d=.2)

    # test all classes with default parameters
    # and with generic dw
    # ----------------------------------------
    cls = all_cls + all_shortcuts
    paths, dtype = [1], [None]
    params = (dict(), dict(vshape=()), dict(vshape=(2,)))
    do(processes, cls, params, paths, dtype)

    paths, dtype = [3], [np.float16]
    params = (dict(),)
    do(processes, cls, params, paths, dtype)

    paths, dtype = [11], [None]
    params = (dict(vshape=2, dw=Z2),
              dict(vshape=2, dw=S2),
              )
    cls = list(cls)
    for p in (heston_process, heston,
              full_heston_process, heston_xy,
              hull_white_process, hwff):
        cls.remove(p)
    do(processes, cls, params, paths, dtype)

    # test each class with specific parameters
    # ----------------------------------------
    paths, dtype = [11], [None]

    cls = (wiener_process, lognorm_process,
           wiener, lognorm)
    params = (
        dict(vshape=2, x0=.1, mu=.2, sigma=.3),
        dict(vshape=2, x0=((.1,), (.11,)),
             mu=((.2,), (.21,)), sigma=((.3,), (.31,))),
        dict(vshape=(2,), corr=CM2),
        dict(vshape=(2, 3)),
        dict(vshape=(2, 3), corr=CM3),
        dict(vshape=(2, 3), dw=wiener_source(paths=11, vshape=(2, 3))),
        dict(vshape=(), dw=true_wiener_source(paths=11, vshape=())),
        dict(vshape=(2, 3), dw=Z23),
        )
    do(processes, cls, params, paths, dtype)

    cls = (ornstein_uhlenbeck_process, oruh)
    params = (
        dict(vshape=2, x0=.1, theta=.2, k=.4, sigma=.5),
        dict(vshape=(2,), corr=CM2),
        dict(vshape=(2,), dw=Z2),
        dict(vshape=(2, 3), dw=wiener_source(paths=11, vshape=(2, 3))),
        dict(vshape=(2, 3), dw=Z23),
        dict(vshape=(), dw=true_wiener_source(paths=11, vshape=()))
    )
    do(processes, cls, params, paths, dtype)

    cls = (hull_white_process, hwff)
    params = (
        dict(vshape=()),
        dict(vshape=(), factors=2),
        dict(vshape=(), factors=2,
             x0=((.1,), (.2,)), theta=((.4,), (.5,)), k=((.6,), (.7,)),
             sigma=((.8,), (.9,)), rho=0.25),
        dict(vshape=(), factors=2,
             dw=wiener_source(paths=11, vshape=(2,),
                              corr=((1, .25,), (.25, 1)))),
        dict(vshape=(3,), factors=2, dw=Z32),
        dict(vshape=(), factors=3,
             dw=wiener_source(paths=11, vshape=(3,), corr=CM3)),
        dict(vshape=(2, 3), factors=5,
             sigma=np.arange(2*3*5).reshape(2, 3, 5, 1)/(300)),
        dict(vshape=2, factors=3, dw=Z23),
        dict(vshape=(), factors=2, dw=Z2),
        dict(vshape=(), factors=2, dw=S2),
    )
    do(processes, cls, params, paths, dtype)

    cls = (hull_white_1factor_process, hw1f)
    params = (
        dict(vshape=2, x0=.1, theta=.2, k=.4, sigma=.5),
        dict(vshape=(2, 3), dw=wiener_source(paths=11, vshape=(2, 3))),
        dict(vshape=(), dw=true_wiener_source(paths=11, vshape=())),
    )
    do(processes, cls, params, paths, dtype)

    cls = (cox_ingersoll_ross_process, cir)
    params = (
        dict(vshape=2, x0=.1, theta=.2, k=.3, xi=.4),
        dict(vshape=(2,), corr=CM2),
        dict(vshape=(2,), rho=-.1),
        dict(vshape=(2, 3), dw=wiener_source(paths=11, vshape=(2, 3))),
        dict(vshape=(), dw=true_wiener_source(paths=11, vshape=())),
        dict(vshape=(2, 3), dw=Z23)
    )
    do(processes, cls, params, paths, dtype)

    cls = (full_heston_process, heston_xy)
    params = (
        dict(vshape=(1,)),
        dict(vshape=(3,), x0=.1, mu=.2, sigma=.3,
             y0=.4, theta=.5, k=.6, xi=.7),
        dict(vshape=(3,), x0=.1, mu=((.2,), (.3,), (.4,)), sigma=.3,
             y0=.4, theta=.5, k=.6, xi=.7),
        dict(vshape=(3,), corr=CM6),
        dict(vshape=(3,), rho=(.1, .2, .3)),
        dict(vshape=(5,), dw=wiener_source(paths=11, vshape=(10,))),
        dict(vshape=10, dw=true_wiener_source(paths=11, vshape=(20,))),
        dict(vshape=(), dw=Z2),
        dict(vshape=(), dw=S2),
    )
    do(processes, cls, params, paths, dtype)

    # correlation matrix should be even-dimensional and matching vshape
    assert_raises(ValueError, full_heston_process,
                  vshape=(2,), corr=CM6)
    assert_raises(ValueError, full_heston_process,
                  vshape=(2,), corr=np.asarray(CM6)[:5, :5])

    cls = (heston_process, heston)
    params = (
        dict(vshape=(1,)),
        dict(vshape=2, x0=.1, mu=.2, sigma=.3,
             y0=.4, theta=.5, k=.6, xi=.7),
        dict(vshape=(), corr=CM2),
        dict(vshape=(1,), corr=CM2),
        dict(vshape=(2, 3, 1), rho=.25),
        dict(vshape=(3), dw=wiener_source(paths=11, vshape=(6,))),
        dict(vshape=(2, 3, 1),
             dw=true_wiener_source(paths=11, vshape=(2, 3, 2))),
        dict(vshape=(), dw=Z2),
        dict(vshape=(), dw=S2),
    )
    do(processes, cls, params, paths, dtype)

    cls = (jumpdiff_process, jumpdiff,
           merton_jumpdiff_process, mjd,
           kou_jumpdiff_process, kou,
           )
    params = (
        dict(vshape=2, x0=.1, mu=.2, sigma=.3,
             lam=.4),
        dict(vshape=(2,), corr=CM2),
        dict(vshape=(2, 3), dw=wiener_source(paths=11, vshape=(2, 3))),
        dict(vshape=(3,), dw=true_wiener_source(paths=11, vshape=(3,)),
             dj=cpoisson_source(paths=11, vshape=(3,))),
        dict(vshape=(3,), dw=true_wiener_source(paths=11, vshape=(3,)),
             dj=true_cpoisson_source(paths=11, vshape=(3,))),
        dict(vshape=2, dj=poisson_source(paths=11, vshape=2)),
        dict(vshape=2, dj=true_poisson_source(paths=11, vshape=2)),
        dict(vshape=2, dj=cpoisson_source(paths=11, vshape=2, y=rv)),
        dict(vshape=2, dj=true_cpoisson_source(paths=11, vshape=2, y=rv)),
        dict(vshape=2, dw=Z2, dj=Z2),
        dict(vshape=2, dw=S2, dj=S2),
    )
    do(processes, cls, params, paths, dtype)

    cls = (jumpdiff_process, jumpdiff)
    params = (
        dict(vshape=2, y=norm_rv(a=.1, b=.2)),
        dict(vshape=2, y=uniform_rv(a=.1, b=.2)),
        dict(vshape=2, y=double_exp_rv(pa=.1, a=.2, b=.3)),
        dict(vshape=2, y=rv),
    )
    do(processes, cls, params, paths, dtype)

    cls = (merton_jumpdiff_process, mjd)
    params = (
        dict(vshape=2, x0=.1, mu=.2, sigma=.3,
             lam=.4, a=.5, b=.6),
    )
    do(processes, cls, params, paths, dtype)

    cls = (kou_jumpdiff_process, kou)
    params = (
        dict(vshape=()),
        dict(vshape=2, x0=.1, mu=.2, sigma=.3,
             lam=.4, pa=.5, a=.6, b=.7),
    )
    do(processes, cls, params, paths, dtype)


# enumerate test cases with time-dependent parameters and launch tests
# --------------------------------------------------------------------
def test_processes_local():
    np.random.seed(SEED)

    # setup some parameter test values
    # --------------------------------
    paths = [11]
    dtype = [None, float, np.float16]

    t = np.linspace(0, 2, 10)
    A = process(t, v=.1 + .2*t)
    B = A*A

    def C(t): return 0.1 + .1*t

    def D(t): return .1 + .5*t/10

    def CM2(t): return ((1, .2*t/10), (.2*t/10, 1))

    def CM6(t):
        C = np.eye(6) + t*0.1*np.random.random((6, 6))
        return (C + C.T)/2

    def R(t): return 0.5 - 0.1*t

    def R3(t): return (.1*C(t), .2*C(t), .3*C(t))

    CM2P = process((1., 2, 3), v=(CM2(1), CM2(2), CM2(3)))
    RP = A/10
    R3P = process((1., 2, 3), v=(R3(1), R3(2), R3(3)))

    # launch tests
    # --------------------------------
    cls = (wiener_process, lognorm_process)
    params = (
        dict(vshape=(), x0=.1, mu=A, sigma=B),
        dict(vshape=(2,), mu=A, sigma=C, corr=CM2),
        dict(vshape=(2,), mu=A, sigma=C, corr=CM2P),
        dict(vshape=(2,), mu=A, sigma=C, rho=R),
        dict(vshape=(2,), mu=A, sigma=C, rho=RP),
        dict(vshape=(6,), mu=A, sigma=C, corr=CM6),
        dict(vshape=(6,), mu=A, sigma=C, rho=R3P),
        dict(vshape=(6,), mu=A, sigma=C,
             dw=true_wiener_source(paths=11, vshape=6, rho=R3P)),
        )
    do(processes, cls, params, paths, dtype)

    cls = (ornstein_uhlenbeck_process, )
    params = (
        dict(vshape=(), x0=.1, theta=A, k=B, sigma=C),
        dict(vshape=(2,), theta=A, k=B, sigma=C, corr=CM2),
        dict(vshape=(2,), theta=A, k=B, sigma=C,
             dw=true_wiener_source(paths=11, vshape=2, corr=CM2)),
    )
    do(processes, cls, params, paths, dtype)

    cls = (hull_white_process, )
    params = (
        dict(vshape=(), factors=2, x0=((.1,), (.2,)),
             theta=lambda t: (A(t), B(t)), k=lambda t: ((C(t),), (D(t),)),
             sigma=lambda t: ((C(t),), (C(t),)), rho=C),
    )
    do(processes, cls, params, paths, dtype)

    cls = (cox_ingersoll_ross_process,)
    params = (
        dict(vshape=(), x0=.1, theta=A, k=B, xi=C),
        dict(vshape=(2,), theta=A, k=B, xi=C, corr=CM2)
    )
    do(processes, cls, params, paths, dtype)

    cls = (heston_process,)
    params = (
        dict(vshape=(), x0=.1, mu=A, sigma=B,
             y0=.2, theta=C, k=B, xi=A, rho=B),
        dict(vshape=(3,), theta=A, k=B, xi=C, corr=CM6),
        dict(vshape=(3,), theta=A, k=B, xi=C, rho=R3P)
    )
    do(processes, cls, params, paths, dtype)

    cls = (full_heston_process,)
    params = (
        dict(vshape=(), x0=.1, mu=A, sigma=B,
             y0=.2, theta=A, k=B, xi=C, rho=B),
        dict(vshape=(3,), theta=A, k=B, xi=C, corr=CM6)
    )
    do(processes, cls, params, paths, dtype)

    cls = (jumpdiff_process,)
    params = (
        dict(vshape=(), x0=.1, mu=A, sigma=B, lam=D),
        dict(vshape=(2,), mu=C, sigma=B, lam=A, corr=CM2),
        dict(vshape=(2,), mu=C, sigma=A, lam=B, corr=CM2P),
        dict(vshape=2, mu=A, sigma=B, y=norm_rv(a=D, b=C))
    )
    do(processes, cls, params, paths, dtype)

    cls = (merton_jumpdiff_process,)
    params = (
        dict(vshape=(), x0=.1, mu=A, sigma=B,
             lam=C, a=D, b=A),
        dict(vshape=2, mu=A, sigma=B, lam=C, corr=CM2),
    )
    do(processes, cls, params, paths, dtype)

    cls = (kou_jumpdiff_process,)
    params = (
        dict(vshape=(), x0=.1, mu=A, sigma=B,
             lam=C, pa=A, a=D, b=A),
        dict(vshape=(2,), mu=A, sigma=B, lam=C, corr=CM2)
    )
    do(processes, cls, params, paths, dtype)


# case testing
def processes(cls, params, paths, dtype):

    # print(cls.__name__, sep='', end='')
    P = cls(**params, paths=paths, dtype=dtype)

    vshape = params.get('vshape', P.vshape)
    if not isinstance(vshape, tuple):
        vshape = (vshape,)

    tlist = [(0.,), (1, 2), np.linspace(0, 1, 11)]
    stepslist = [None, 5, (1, 1.5, 2.5)]
    i0list = [0, 3, -2]
    for t in tlist:
        for steps in stepslist:
            for i0 in i0list:
                tt = np.asarray(t).reshape(-1)
                i0 = min(i0, tt.size - 1)
                if i0 + tt.size < 0:
                    i0 = 0
                if iskfunc(cls):
                    ps = P(t, i0=i0, steps=steps)
                else:
                    ps = cls(**params, paths=paths, dtype=dtype,
                             steps=steps, i0=i0)(t)
                if not isinstance(ps, (tuple, list)):
                    ps = (ps,)
                for p in ps:
                    assert_(isinstance(p, process))
                    assert_array_equal(p.t, tt)
                    assert_(p.vshape == vshape)
                    assert_(p.paths == paths)
                    assert_(p.dtype == dtype)
                    # no inf or nan
                    assert_(np.isfinite(p).all())
                    # initial values are the same for all paths
                    assert_((p[i0] == p[i0, ..., 0][..., np.newaxis]).all())

    # check initial conditions on the last computed p
    if cls in {wiener_process, lognorm_process,
               ornstein_uhlenbeck_process,
               cox_ingersoll_ross_process,
               jumpdiff_process,
               merton_jumpdiff_process,
               kou_jumpdiff_process}:
        assert_(len(ps) == 1)
        p = ps[0]
        p0, x0 = np.broadcast_arrays(p[i0], P.args['x0'])
        assert_allclose(p0, x0, rtol=eps(p.dtype))

    elif cls in {hull_white_process}:
        assert_(len(ps) == 1)
        p = ps[0]
        x0 = np.broadcast_to(P.args['x0'],
                             p.vshape + (P.args['factors'],) + (1,))
        p0, x0 = np.broadcast_arrays(p[i0], x0.sum(axis=-2))
        assert_allclose(p0, x0, rtol=eps(p.dtype))

    elif cls in {full_heston_process}:
        assert_(len(ps) == 2)
        p, q = ps
        p0, x0 = np.broadcast_arrays(p[i0], P.args['x0'])
        q0, y0 = np.broadcast_arrays(q[i0], P.args['y0'])
        assert_allclose(p0, x0, rtol=eps(p.dtype))
        assert_allclose(q0, y0, rtol=eps(p.dtype))


# stand alone test
def test_processes_exceptions():

    # timeline must be 1-dimensional
    for cls in all_cls:
        P = cls if iskfunc(cls) else cls()
        assert_raises(ValueError, P, np.arange(6.).reshape(2, 3))


def test_jumps():
    np.random.seed(SEED)

    # jump diffusion process integrated as is
    # (jumpdiff_process integrates the logarithm)
    @integrate(q=0, sources={'dt', 'dw', 'dj'})
    def jumpdiff_process2(s, x, mu=0., sigma=1.):
        return {'dt': mu*x,
                'dw': sigma*x,
                'dj': x}

    jdp1 = jumpdiff_process
    jdp2 = jumpdiff_process2
    t = (1, 1.5, 2)

    # compare using same stochasticity sources
    # (using almost deterministic y - easiest way to
    # synchronize dj1 and dj2 y values...)
    PATHS = 10
    eps = .000000001
    y1 = uniform_rv(.1, .1 + eps)
    y2 = rvmap(lambda y: exp(y) - 1, y1)
    dn = true_poisson_source(lam=10, paths=PATHS)
    dw = true_wiener_source(paths=PATHS)
    dj1 = true_cpoisson_source(dn=dn, paths=PATHS, y=y1)
    dj2 = true_cpoisson_source(dn=dn, paths=PATHS, y=y2)

    args = dict(sigma=.2, mu=0.2, paths=PATHS, steps=501)
    X1 = jdp1(**args, dj=dj1, dw=dw)
    X2 = jdp2(**args, dj=dj2, dw=dw)
    x1, x2 = X1(t), X2(t)
    # print(np.abs((x1-x2)/x1).max())
    # print(x1(2),x2(2))
    assert_(np.abs(1-x2/x1).max() < 0.02)

    # same as above using time dependence
    PATHS = 10
    eps = .000000001
    y1 = uniform_rv(lambda t: .05 + t*.05,
                    lambda t: .05 + t*.05 + eps)
    y2 = rvmap(lambda y: exp(y) - 1, y1)
    dn = true_poisson_source(lam=lambda t: t*5, paths=PATHS)
    dw = true_wiener_source(paths=PATHS)
    dj1 = true_cpoisson_source(dn=dn, paths=PATHS, y=y1)
    dj2 = true_cpoisson_source(dn=dn, paths=PATHS, y=y2)

    args = dict(sigma=lambda t: (.2 - t*.1), mu=lambda t: 0.1*t,
                paths=PATHS, steps=501)
    X1 = jdp1(**args, dj=dj1, dw=dw)
    X2 = jdp2(**args, dj=dj2, dw=dw)
    x1, x2 = X1(t), X2(t)
    # print(np.abs((x1-x2)/x1).max())
    # print(x1(2),x2(2))
    assert_(np.abs(1-x2/x1).max() < 0.03)

    # fully stochastic
    PATHS = 10000
    y1 = uniform_rv(0.05, 0.10)
    y2 = rvmap(lambda y: exp(y) - 1, y1)
    args = dict(sigma=.05, mu=-.1, lam=5, paths=PATHS, steps=501)
    X1 = jdp1(**args, y=y1)
    X2 = jdp2(**args, y=y2)
    x1, x2 = X1(t), X2(t)
    m1, s1 = x1[-1].mean(), x1[-1].std()
    m2, s2 = x2[-1].mean(), x2[-1].std()
    # print(m1, m2, s1, s2)
    assert_(abs(1-m2/m1) < 0.01)
    assert_(abs(1-s2/s1) < 0.03)


def test_processes_misc():

    # test exactness of wiener_process and lognorm_process
    # with constant parameters

    np.random.seed(SEED)
    paths = 31
    x0 = 10
    mu, sigma = .2, .7
    dw = true_wiener_source(paths=paths)
    t0, DT = 1, 3

    pw = wiener_process(x0=x0, mu=mu, sigma=sigma, paths=paths, dw=dw)
    pl = lognorm_process(x0=x0, mu=mu, sigma=sigma, paths=paths, dw=dw)

    t1 = (t0, t0 + DT/2, t0 + DT)
    t2 = np.linspace(t0, t0 + DT, 100)

    xw_exact = x0 + mu*DT + sigma*dw(t0, DT)
    xl_exact = x0*exp((mu - sigma*sigma/2)*DT + sigma*dw(t0, DT))
    xw1, xw2 = pw(t1), pw(t2)
    xl1, xl2 = pl(t1), pl(t2)

    rtol = eps(xw1.dtype)
    assert_allclose(xw_exact, xw1[-1], rtol=rtol)
    assert_allclose(xw_exact, xw2[-1], rtol=rtol)
    assert_allclose(xl_exact, xl1[-1], rtol=rtol)
    assert_allclose(xl_exact, xl2[-1], rtol=rtol)
