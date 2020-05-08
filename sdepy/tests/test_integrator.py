"""
===============================================
Formal tests on the paths_generator, integrator
and SDE/SDEs classes
===============================================
"""
from .shared import *

process = sp.process
kfunc = sp.kfunc
paths_generator, integrator = sp.paths_generator, sp.integrator
SDE, SDEs = sp.SDE, sp.SDEs
integrate = sp.integrate

norm_rv = sp.norm_rv
uniform_rv = sp.uniform_rv
exp_rv = sp.exp_rv
double_exp_rv = sp.double_exp_rv

true_wiener_source = sp.true_wiener_source


# -----------------------------
# paths_generator general tests
# -----------------------------

# a generator subclass, to be tested
@kfunc
class generator_cls(paths_generator):

    depth = 2

    def __init__(self, *, exitf=1., vshape=(),
                 xw0=None, **args):
        self.exitf, self.xw0 = exitf, xw0
        self.vshape = vshape
        super().__init__(xshape=vshape, wshape=vshape,
                         **args)

    def begin(self):
        super().begin()
        iv = self.itervars
        iv['xw'][0][...] = np.asarray(self.xw0)

    def next(self):
        super().next()
        iv = self.itervars
        sw, xw = iv['sw'], iv['xw']
        x = xw[0]
        ds = sw[1] - sw[0]
        xw[1][...] = x + ds

    def store(self, i, k):
        super().store(i, k)
        iv = self.itervars
        iv['xx'][i] = iv['xw'][k]

    def exit(self, tt, xx):
        return process(tt, x=xx * self.exitf)


# tests with output timeline == integration timeline
def test_generator_t():
    np.random.seed(SEED)

    integr = generator_cls(paths=1, vshape=(), xw0=1)
    t = np.linspace(0, 10, 11)

    # test integration
    p = integr(t)
    assert_allclose(p[:, 0], 1 + t, rtol=eps(p.dtype))

    # test post processing with exit_integrate post-processing
    p = integr(t, exitf=10)
    assert_allclose(p[:, 0], 10 * (1 + t), rtol=eps(p.dtype))

    # test post processing with exit_integrate providing
    # explicit result
    p = integr(t, exitf=10)
    assert_allclose(p[:, 0], 10 * (1 + t), rtol=eps(p.dtype))

    # test multiple paths
    p = integr(t, paths=7)
    assert_allclose(p, np.full(t.shape + (7,), 1.) + t[:, np.newaxis],
                    rtol=eps(p.dtype))

    # test integration backwards
    p = integr(t)
    assert_allclose(p[:, 0], 1 + t, rtol=eps(p.dtype))

    # test exceptions
    assert_raises(ValueError, integr, t.reshape(-1, 1))  # bad shape
    assert_raises(ValueError, integr, (1, 2, 2, 3))  # repeated values
    assert_raises(ValueError, integr, (3, 2, 1))  # not sorted
    assert_raises(IndexError, integr, t, i0=Ellipsis)  # bad i0
    assert_raises(IndexError, integr, t, i0=12)  # bad i0
    assert_raises(IndexError, integr, t, i0=[1, 2, 3])  # bad i0
    integr2 = integr(vshape=())
    integr2.depth = 1
    assert_raises(ValueError, integr2, t)

    # test process dtype
    p = integr(t, dtype=np.float32)
    assert_(p.dtype == np.float32)
    assert_allclose(p[:, 0], 1 + t, rtol=eps(p.dtype))
    p = integr(t, dtype=np.float64)
    assert_(p.dtype == np.float64)
    assert_allclose(p[:, 0], 1 + t, rtol=eps(p.dtype))
    ti = np.arange(10)
    p = integr(ti, steps=37)
    assert_allclose(p[:, 0], 1 + ti, rtol=eps(p.dtype))


# tests with output timeline != integration timeline
def test_generator_steps():
    np.random.seed(SEED)

    integr = generator_cls(paths=1, vshape=(), xw0=1,
                           info={})
    t = np.linspace(0, 10, 11)

    # test integration and info
    p1 = integr(t)
    i1 = integr.info.copy()
    p2 = integr(t, steps=37)
    i2 = integr.info.copy()
    assert_allclose(p1, p2, rtol=eps(p1.dtype))
    assert_array_equal(p1.t, p2.t)
    assert_(i1['computed_steps'] == 10)
    assert_(i1['stored_steps'] == 10)
    assert_(i2['computed_steps'] >= 37)
    assert_(i2['stored_steps'] == 10)

    # test vshape same as wshape
    xw0 = np.arange(3*5*7).reshape(3, 5, 7)
    z = integr(paths=7, vshape=(3, 5), xw0=xw0)
    p1, p2 = z(t), z(t, steps=149)
    assert_allclose(p1, p2, rtol=16*eps(p1.dtype))


# testing depth > 2
class deep_cls(paths_generator):

    depth = 4

    def begin(self):
        super().begin()
        iv = self.itervars
        for k in range(3):
            iv['xw'][k][...] = (k + 1)*10

    def next(self):
        super().next()
        iv = self.itervars
        sw, xw = iv['sw'], iv['xw']
        x = xw[-2]
        ds = sw[-1] - sw[-2]
        xw[-1][...] = x + ds

    def store(self, i, k):
        super().store(i, k)
        iv = self.itervars
        iv['xx'][i] = iv['xw'][k]

    def exit(self, tt, xx):
        return process(tt, x=xx)


def test_generator_depth():

    np.random.seed(SEED)
    integr = deep_cls(dtype=int)
    x = integr((1, 2, 3))
    assert_((x == np.array((10, 20, 30)).reshape(-1, 1)).all())
    x = integr((1, 2, 3, 4, 5))
    assert_((x == np.array((10, 20, 30, 31, 32)).reshape(-1, 1)).all())
    assert_raises(ValueError, integr, (1, 2))
    assert_raises(ValueError, integr, (1,))


# ------------------------------
# integrator and integrate tests
# ------------------------------

def test_SDE():
    np.random.seed(SEED)

    paths = 3
    vshapes = ((), 2, (2, 5))
    t = (1., 2, 3, 4, 5, 6, 7)

    # 1-equation system
    # -----------------
    def f(t, x):
        return ({'dt': t+x, 'dw': x},)
    x = integrate(f)(x0=0.)(t)  # auto-detect q and sources
    assert_(len(x) == 1 and x[0].shape == (7, 1))

    # a minimal 2-equations system
    # ----------------------------
    def f(t, x, y=0.):
        return {'dt': 1, 'dw': 1}, {'dt': 1, 'dw': 1}

    # use defaults
    x, y = integrate(f)(x0=(1., 1.))(t)  # auto-detect
    for u in (x, y):
        assert_(u.shape == (7, 1))
    x, y = integrate(f, q=2, sources={'dt', 'dw'}  # explicit
                     )(x0=(1, 1))(t)
    for u in (x, y):
        assert_(u.shape == (7, 1))

    # evaluate with different shapes
    for vshape in vshapes:
        tst_vshape = (vshape,) if isinstance(vshape, int) else vshape
        x, y = integrate(f, q=2, sources={'dt', 'dw'})(
            x0=(1., 1.), paths=paths, vshape=vshape)(t)
        for z in (x, y):
            assert_(z.shape == (7,) + tst_vshape + (3,))

    # 3-equations system, parameters, different calling styles
    # --------------------------------------------------------
    def f(t, x=0, y=0, z=0, sigma=1., a=2.):
        return ({'dt': a*x+z, 'dw': sigma*y}, {'dt': t*y, 'dw': y+sigma*z},
                {'dt': a+x, 'dw': y-sigma*z})
    x, y, z = integrate(f)(x0=(1, 1, 1))(t)  # auto-detect
    for u in (x, y, z):
        assert_(u.shape == (7, 1))

    @integrate
    def f_process(t, x=0, y=0, z=0, sigma=1., a=2.):
        return f(t, x, y, z, sigma, a)

    x, y, z = f_process(sigma=2., x0=(1, 1, 1))(t)
    for u in (x, y, z):
        assert_(u.shape == (7, 1))

    @integrate(q=3, sources=('dt', 'dw'))
    def f_process(t, x=0, y=0, z=0, sigma=1., a=2.):
        return f(t, x, y, z, sigma, a)
    x, y, z = f_process(sigma=2., x0=(1, 1, 1))(t)
    for u in (x, y, z):
        assert_(u.shape == (7, 1))

    # same equation, calling with 'method' keyword
    # --------------------------------------------
    x, y, z = integrate(f)(method='euler', x0=(1, 1, 1))(t)
    for u in (x, y, z):
        assert_(u.shape == (7, 1))

    # same equation, time-dependent parameters, more parameters
    # ---------------------------------------------------------
    x, y, z = f_process(sigma=2., a=lambda t: 1+t,
                        vshape=5, paths=11, steps=30,
                        x0=(1, 1, 1))(t)
    for u in (x, y, z):
        assert_(u.shape == (7, 5, 11))

    # x0 and sigma with different values on the two elements of vshape
    x, y, z = integrate(f)(x0=(((1,), (1.1,)), ((2,), (2.1,)), ((3,), (3.1,))),
                           sigma=((.1,), (.2,)), a=lambda t: ((1+t,), (2*t,)),
                           vshape=2, paths=11, steps=30)(t)
    for u in (x, y, z):
        assert_(u.shape == (7, 2, 11))

    # path-dependent initial conditions, backward-forward integration
    xinit = np.linspace(1., 2., 11)
    x, y, z = f_process(sigma=((.1,), (.2,)),
                        x0=(xinit, 2*xinit, 3*xinit),
                        vshape=2, paths=11, steps=30, i0=5)(t)

    # same equation, time-dependent parameters,
    # with correlated wiener processes
    # -----------------------------------------
    args = dict(x0=(1, 2, 3), sigma=((.1,), (.2,)),
                a=lambda t: ((1+t,), (2*t,)),
                vshape=2, paths=11, steps=30)
    x, y, z = integrate(f)(**args, rho=lambda t: (.01*t, .2, .3))(t)
    for u in (x, y, z):
        assert_(u.shape == (7, 2, 11))
    x, y, z = integrate(f)(
        **args, corr=lambda t:
        ((1,   .01*t,  .2,  .3,  .4,  .5),
         (.01*t,   1, -.1, -.2, -.3, -.4),
         (.2,    -.1,   1,   0,  .1,   0),
         (.3,    -.2,   0,   1,  .1,  .2),
         (.4,    -.3,  .1,  .1,   1,   0),
         (.5,    -.4,   0,  .2,   0,   1))
        )(t)
    for u in (x, y, z):
        assert_(u.shape == (7, 2, 11))

    # 2 equations with poisson jumps
    # ------------------------------
    def f(t, x=0, y=0, k=1):
        return ({'dt': x, 'dn': k}, {'dt': 1, 'dn': k*y})
    x, y = integrate(f)(x0=(0., 0.), k=lambda t: .01*t,
                        lam=lambda t: .2*t)(t)
    for u in (x, y):
        assert_(u.shape == (7, 1))

    @integrate(q=2, sources={'dt', 'dn'})
    def f_process(t, x, y, k):
        return f(t, x, y, k)

    x, y = f_process(x0=(0., 0.), k=lambda t: .01*t,
                     lam=lambda t: .2*t, steps=30)(t)
    for u in (x, y):
        assert_(u.shape == (7, 1))

    # 2 equations with poisson jumps and wiener increments
    # ----------------------------------------------------
    def f(t, x=0, y=0, k=1):
        return ({'dt': x, 'dn': k, 'dw': y}, {'dt': 1, 'dn': k*y, 'dw': x-y})
    x, y = integrate(f)(x0=(0., 0.), k=lambda t: .01*t,
                        lam=lambda t: .2*t, rho=-.5)(t)
    for u in (x, y):
        assert_(u.shape == (7, 1))

    @integrate(q=2, sources={'dt', 'dn', 'dw'})
    def f_process(t, x, y, k):
        return f(t, x, y, k)

    x, y = f_process(x0=(0., 0.), k=lambda t: .01*t,
                     lam=lambda t: .2*t, rho=lambda t: -.01*t,
                     steps=30)(t)
    for u in (x, y):
        assert_(u.shape == (7, 1))

    # 2 equations with compound poisson jumps
    # ---------------------------------------
    def f(t, x=0, y=0, k=1):
        return ({'dt': x, 'dj': k}, {'dt': 1, 'dj': k*y})
    x, y = integrate(f)(x0=(0., 0.), k=lambda t: .01*t,
                        lam=lambda t: .2*t,
                        y=uniform_rv(a=-1, b=lambda t: t))(t)
    for u in (x, y):
        assert_(u.shape == (7, 1))

    @integrate(q=2, sources={'dt', 'dj'})
    def f_process(t, x, y, k):
        return f(t, x, y, k)

    x, y = f_process(x0=(0., 0.), k=lambda t: .01*t, lam=lambda t: .2*t,
                     y=exp_rv(a=-1), steps=30)(t)
    for u in (x, y):
        assert_(u.shape == (7, 1))

    # 2 equations with poisson jumps and wiener increments
    # ----------------------------------------------------
    def f(t, x=0, y=0, k=1):
        return ({'dt': x, 'dj': k, 'dw': y}, {'dt': 1, 'dj': k*y, 'dw': x-y})
    x, y = integrate(f)(x0=(0., 0.), k=lambda t: .01*t,
                        lam=lambda t: .2*t, rho=-.5,
                        y=norm_rv(a=1, b=lambda t: 2+t))(t)
    for u in (x, y):
        assert_(u.shape == (7, 1))

    @integrate(q=2, sources={'dt', 'dj', 'dw'})
    def f_process(t, x, y, k):
        return f(t, x, y, k)

    x, y = f_process(x0=(0., 0.), k=lambda t: .01*t,
                     lam=lambda t: .2*t, rho=lambda t: -.01*t,
                     y=double_exp_rv(a=1, b=2, pa=.1), steps=30)(t)
    for u in (x, y):
        assert_(u.shape == (7, 1))

    # minimal 4 equations with no 'dt' term
    # -------------------------------------
    def f(t, x=0, y=0, z=0, w=0):
        return ({'dw': x}, {'dw': y}, {'dw': z}, {'dw': w})
    rm4 = np.random.random((4, 4))
    corr4 = np.eye(4) + 0.1 * (rm4 + rm4.T)
    xs = integrate(f)(x0=(1,)*4, corr=corr4, paths=11, steps=30)(t)
    for u in xs:
        assert_(u.shape == (7, 11))

    # as above, with a true_wiener source
    # -----------------------------------
    tw = true_wiener_source(vshape=4, paths=11,
                            corr=corr4)
    xs1 = integrate(f)(x0=(1.,)*4, dw=tw, paths=11, steps=30)(t)
    for u in xs1:
        assert_(u.shape == (7, 11))
    xs2 = integrate(f)(x0=(1.,)*4, dw=tw, paths=11, steps=30)(t)
    for u, v in zip(xs1, xs2):
        assert_allclose(xs1, xs2, rtol=eps(u.dtype))

    # 4 equations with terms partially omitted
    # (formerly a cause of error)
    # ----------------------------------------
    @integrate
    def f_process(t, x=0, y=0, z=0, w=0):
        return ({'dt': 1, 'dw': 1}, {'dt': 1}, {'dw': 1}, {})
    xs = f_process(x0=(1,)*4, paths=11, steps=30)(t)

    # SDE class from integrate
    # ------------------------
    # without test evaluation (single equation specified by 'q=0')
    @integrate(q=0, sources={'dw', 'dt'})
    def f_process(t, x):
        return {'dt': 1, 'dw': 1}
    assert_(issubclass(f_process, SDE))
    assert_(not issubclass(f_process, SDEs))
    x = f_process(x0=1, paths=11, steps=30)(t)

    # with test evaluation
    @integrate
    def f_process(t, x):
        return {'dt': 1, 'dw': 1}
    assert_ (issubclass(f_process, SDE))
    assert_(not issubclass(f_process, SDEs))
    x = f_process(x0=1, paths=11, steps=30)(t)

    # test errors
    # -----------

    # SDE: ok
    class f_process(SDE, integrator):
        def sde(self, t, x):
            return {'dt': 1, 'dw': 1}
    x = f_process(x0=1, paths=11, steps=30)(t)

    # SDE: wrong type
    f_process.sde = lambda self, t, x: x
    assert_raises(TypeError, f_process(x0=1, paths=11, steps=30), t)

    # SDE: wrong sde entry
    f_process.sde = lambda self, t, x: {'dt': 1, 'dzzz': 1}
    assert_raises(KeyError, f_process(x0=1, paths=11, steps=30), t)

    # SDEs: ok
    @integrate(q=2, sources=('dt', 'dw'))
    def f_process(t, x, y):
        return {'dt': 1, 'dw': 1}, {'dt': 1}
    assert f_process.q == 2
    x, y = f_process(x0=(1,)*2, paths=11, steps=30)(t)

    # SDEs: wrong type
    f_process.sde = lambda self, t, x, y: {'dt': 1, 'dw': 1}
    assert f_process.q == 2
    assert_raises(TypeError, f_process(x0=(1,)*2, paths=11, steps=30), t)

    # SDEs: wrong number of equations
    f_process.sde = lambda self, t, x, y: ({'dt': 1, 'dw': 1},)
    assert_raises(ValueError, f_process(x0=(1,)*2, paths=11, steps=30), t)

    # SDEs: wrong sde entry
    f_process.sde = lambda self, t, x, y: ({'dt': 1}, {'dzzz': 1})
    assert_raises(KeyError, f_process(x0=(1,)*2, paths=11, steps=30), t)

    # integrate errors
    # ----------------
    # no test evaluation of f_process
    @integrate(q=0, sources={'dt', 'dw'})
    def f_process(t, x):
        raise ValueError

    # test evaluation ok
    @integrate(q=2)
    def f_process(t, x, y=1.):
        return {'dt': 1}, {'dw': 1}
    xs = f_process(x0=(1,)*2, paths=11, steps=30)(t)

    # test evaluation inconsistent with declared q or sources
    def err():
        @integrate(q=1)
        def f_process(t, x=1., y=1.):
            return {'dt': 1}, {'dw': 1}
    assert_raises(TypeError, err)
    def err():
        @integrate(sources={'dw'})
        def f_process(t, x=1., y=1.):
            return {'dt': 1}, {'dw': 1}
    assert_raises(TypeError, err)

    # test evaluation fails
    def err():
        @integrate
        def f_process(t, x):
            raise ValueError
    assert_raises(TypeError, err)
