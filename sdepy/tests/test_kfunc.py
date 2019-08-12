"""
========================
TESTS ON THE KFUNC CLASS
========================
"""
from .shared import *
kfunc = sp.kfunc
iskfunc = sp.iskfunc


def test_aaa_config():
    """testall.py: print realized configuration"""
    if VERBOSE:
        print('\n****', __name__, 'configuration:')
        print('KFUNC, iskfunc(wiener), iskfunc(wiener_process) =',
              KFUNC, sp.iskfunc(sp.wiener), sp.iskfunc(sp.wiener_process))
        print('PLOT, SAVE_ERRORS =',
              PLOT, SAVE_ERRORS)
        print('VERBOSE, QUANT_TEST_MODE =',
              VERBOSE, QUANT_TEST_MODE)


test_aaa_config.config = True


# -----------
# test kfunc
# -----------

def test_kfunc():

    # test kfunc definition errors
    # ----------------------------

    class fail1:
        def __new__(cls):
            pass

    class fail2:
        def __init__(self):
            pass

    class fail3:
        def __call__(self):
            pass

    class fail4(fail3):
        def __init__(self, x):
            pass

    class fail5:
        def __init__(self, *, x):
            pass

        def __call__(self, x):
            pass

    class warn1(fail2, fail3):
        params = 0

    @kfunc
    class warn2_parent(fail2, fail3):
        pass

    class warn2(warn2_parent):
        pass

    assert_raises(TypeError, kfunc, fail1)
    assert_raises(TypeError, kfunc, fail2)
    assert_raises(TypeError, kfunc, fail3)
    assert_raises(TypeError, kfunc, fail4)
    assert_raises(TypeError, kfunc, fail5)

    assert_warns(RuntimeWarning, kfunc, warn1)
    with assert_warns(RuntimeWarning):
        warn2()

    # define a test kfunc class
    # -------------------------
    @kfunc
    class Fbase:
        """
        kfunc with parmeters: a, b and info,
        variables: x, y
        info is used to return evaluation additional info
        """

        def __init__(self, *, a=1, b=2, info=None):
            # store in attributes what was passed to __init__
            self.a, self.b, self.info = a, b, info
            # initialize a info attribute
            if self.info is None:
                self.info = {}

        def __call__(self, x=11, y=22):
            self.info['value'] = (x, y)  # a test value returned by reference
            return (x, y, self.a, self.b)  # some test return values

    # define test subclasses of kfuncs
    # --------------------------------
    @kfunc
    class Finit(Fbase):

        def __init__(self, *, a=1, b=2, info=None):
            self.aa, self.bb = a, b
            super().__init__(a=a, b=b, info=info)

    @kfunc
    class Fcall(Fbase):

        def __call__(self, x=11, y=22):
            self.x, self.y = x, y
            return super().__call__(x, y)

    @kfunc
    class Fboth(Finit, Fcall):
        pass

    # tests an F instance against given expected args values
    # ------------------------------------------------------
    def F_check(f: 'F instance',
                aa: 'expected a', bb: 'expected b',
                ii: 'info' = None):

        # check args and attributes
        assert_(f.a == aa and f.b == bb)
        assert_(f.params == dict(a=aa, b=bb, info=ii))

        # check evaluation with default args
        assert_(f() == (11, 22, aa, bb))
        assert_(f.info['value'] == (11, 22))
        assert_(f('z', 'w') == ('z', 'w', aa, bb))
        assert_(f.info['value'] == ('z', 'w'))

        # check evaluation changing args
        # x=110, y is default, set a=3
        assert_(f(110, a=3) == (110, 22, 3, bb))
        assert_(f.a == aa and f.b == bb)  # defaults preserved
        if ii is not None:
            assert_(f.info is ii)  # info preserved and updated
            assert_(f.info['value'] == (110, 22))
        # x=110, set b=4
        assert_(f(1100, b=4) == (1100, 22, aa, 4))
        assert_(f.a == aa and f.b == bb)  # defaults preserved
        if ii is not None:
            assert_(f.info is ii)
            assert_(f.info['value'] == (1100, 22))
        # x=110, y=220, set a=3 and b=4, force info
        f_value = f(110, 220, a=3, b=4, info=f.info)  # set info dict
        assert_(f_value == (110, 220, 3, 4))
        assert_(f.a == aa and f.b == bb)  # defaults preserved
        assert_(f.info['value'] == (110, 220))  # info updated

    # do tests on F
    # -------------
    for F in (Fbase, Finit, Fcall, Fboth):

        # F instance with default args
        f = F()
        F_check(f, 1, 2)
        assert_(f._kfunc_parent is None)

        # F instance with non-default args
        f = F(a=11)
        F_check(f, 11, 2)
        f = F(b=13)
        F_check(f, 1, 13)
        f = F(a=11, b=13)
        F_check(f, 11, 13)

        # direct call of the class to instantiate and evaluate in one go
        assert_(F(110) == (110, 22, 1, 2))
        assert_(F('z', 'w') == ('z', 'w', 1, 2))
        assert_(F(110, a=3) == (110, 22, 3, 2))
        assert_(F(110, b=4) == (110, 22, 1, 4))
        assert_(F(110, a=3, b=4) == (110, 22, 3, 4))
        assert_(F('z', 'w', a=3, b=4) == ('z', 'w', 3, 4))

        # instantiaton from an instance, with modified args
        f = F(a=7)
        F_check(f, 7, 2)
        g = f(a=11, info=f.info)
        F_check(g, 11, 2, f.info)
        assert_(g._kfunc_parent is f)
        g = f(a=11, b=13, info=f.info)
        F_check(g, 11, 13, f.info)
        assert_(g._kfunc_parent is f)

        # type error in case of unexpected argument
        assert_raises(TypeError, F, z=1)  # at construction
        assert_raises(TypeError, F(), z=1, w=2)  # at instance from instance
        assert_raises(TypeError, F(), 11, 22, a=1, w=2)  # during evaluation
        assert_raises(TypeError, F, 11, 22, b=1, w=2)  # during evaluation

    # do tests on kfunc from functions
    # --------------------------------

    # define a kfunc from a function 1 variable and no parameters
    @kfunc(nvar=1)
    def G(x):
        return x

    assert_(G(5) == 5)
    assert_(G()(5) == 5)

    # define a kfunc from a function with 2 variables and parameters
    @kfunc(nvar=2)
    def G(x, y=0, *, p1='1', p2='2'):
        return x, y, p1, p2

    assert_(G(p1='x').params == dict(p1='x', p2='2'))
    assert_(G(1) == (1, 0, '1', '2'))
    assert_(G(1, 2) == (1, 2, '1', '2'))
    assert_(G(1, 2, p2='y') == (1, 2, '1', 'y'))
    assert_(G(1, p2='y') == (1, 0, '1', 'y'))
    assert_raises(TypeError, G(p3='z'), 1)

    g = G(p1='x', p2='y')
    assert_(g.params == dict(p1='x', p2='y'))
    assert_(g(1) == (1, 0, 'x', 'y'))
    assert_(g(1, p1='xx') == (1, 0, 'xx', 'y'))
    assert_(g(p1='xx')(1) == (1, 0, 'xx', 'y'))
    assert_(g(p1='xx')(1, 2) == (1, 2, 'xx', 'y'))
    assert_(g(p1='xx').params == dict(p1='xx', p2='y'))
    assert_raises(TypeError, g(p3='z'), 1)


def test_kfunc_exceptions():

    def fail():

        @kfunc(nvar=2)
        class A:
            pass

    assert_raises(SyntaxError, fail)

    def fail():

        @kfunc
        def f(x, y):
            pass

    assert_raises(SyntaxError, fail)

    def fail():

        @kfunc(nvar=3)  # nvar out of range
        def f(x, y):
            pass

    assert_raises(ValueError, fail)

    def fail():

        @kfunc(nvar=2)  # non-keyword parameter
        def f(x, y, z):
            pass

    assert_raises(TypeError, fail)


def test_kfunc_params():

    # sources
    # -------

    def kf(cls):
        if iskfunc(cls):
            return cls
        else:
            return kfunc(cls)

    for X in (sp.wiener_source, sp.dw):
        assert_(kf(X)(paths=2).params ==
                {'paths': 2, 'vshape': (), 'dtype': None,
                 'corr': None, 'rho': None})

    for X in (sp.odd_wiener_source, sp.odd_dw):
        # defaults beyond paths and vshape are missed
        # due to _antithetics wrapping
        assert_(kf(X)(paths=2).params ==
                {'paths': 2, 'vshape': ()})

    for X in (sp.true_wiener_source, sp.true_dw):
        assert_(kf(X)(paths=2).params ==
                {'paths': 2, 'vshape': (), 'dtype': None,
                 'rtol': 'max', 't0': 0., 'z0': 0.,
                'corr': None, 'rho': None})

    for X in (sp.poisson_source, sp.dn):
        assert_(kf(X)(lam=2, dtype=np.float32).params ==
                {'paths': 1, 'vshape': (),
                 'dtype': np.float32, 'lam': 2})

    # processes
    # ---------

    for X in (sp.wiener_process, sp.wiener):
        assert_(kf(X)(x0=2).params ==
                {'paths': 1, 'vshape': (), 'dtype': None,
                 'steps': None, 'i0': 0, 'info': None,
                 'getinfo': True, 'method': 'euler',
                 'dw': None, 'corr': None, 'rho': None,
                 'x0': 2, 'mu': 0.0, 'sigma': 1.0})

    for X in (sp.lognorm_process, sp.lognorm):
        assert_(kf(X)(x0=2).params ==
                {'paths': 1, 'vshape': (), 'dtype': None,
                 'steps': None, 'i0': 0, 'info': None,
                 'getinfo': True, 'method': 'euler',
                 'dw': None, 'corr': None, 'rho': None,
                 'x0': 2, 'mu': 0.0, 'sigma': 1.0})

    for X in (sp.ornstein_uhlenbeck_process, sp.oruh):
        assert_(kf(X)(theta=2, steps=20).params ==
                {'paths': 1, 'vshape': (), 'dtype': None,
                 'steps': 20, 'i0': 0, 'info': None,
                 'getinfo': True, 'method': 'euler', 'theta': 2,
                 'dw': None, 'corr': None, 'rho': None,
                 'x0': 0., 'k': 1., 'sigma': 1.})

    for X in (sp.hull_white_process, sp.hwff):
        assert_(kf(X)(vshape=3, factors=2).params ==
                {'paths': 1, 'vshape': 3, 'dtype': None,
                 'steps': None, 'i0': 0, 'info': None,
                 'getinfo': True, 'method': 'euler', 'theta': 0,
                 'dw': None, 'corr': None, 'rho': None,
                 'x0': 0., 'k': 1., 'sigma': 1.,
                 'factors': 2})

    for X in (sp.hull_white_1factor_process, sp.hw1f):
        assert_(kf(X)(vshape=3).params ==
                {'paths': 1, 'vshape': 3, 'dtype': None,
                 'steps': None, 'i0': 0, 'info': None,
                 'getinfo': True, 'method': 'euler', 'theta': 0.,
                 'dw': None, 'corr': None, 'rho': None,
                 'x0': 0., 'k': 1., 'sigma': 1.})

    for X in (sp.cox_ingersoll_ross_process, sp.cir):
        assert_(kf(X)(vshape=3).params ==
                {'paths': 1, 'vshape': 3, 'dtype': None,
                 'steps': None, 'i0': 0, 'info': None,
                 'getinfo': True, 'method': 'euler',
                 'dw': None, 'corr': None, 'rho': None,
                 'x0': 1., 'theta': 1., 'k': 1., 'xi': 1.})

    for X in (sp.heston_process, sp.heston,
              sp.full_heston_process, sp.heston_xy):
        assert_(kf(X)(vshape=2).params ==
                {'paths': 1, 'vshape': 2, 'dtype': None,
                 'steps': None, 'i0': 0, 'info': None,
                 'getinfo': True, 'method': 'euler',
                 'dw': None, 'corr': None, 'rho': None,
                 'x0': 1., 'y0': 1., 'mu': 0., 'sigma': 1.,
                 'theta': 1., 'k': 1., 'xi': 1.})

    for X in (sp.jumpdiff_process, sp.jumpdiff):
        assert_(kf(X)(vshape=2).params ==
                {'paths': 1, 'vshape': 2, 'dtype': None,
                 'steps': None, 'i0': 0, 'info': None,
                 'getinfo': True, 'method': 'euler', 'ptype': int,
                 'dw': None, 'corr': None, 'rho': None,
                 'dj': None, 'dn': None, 'lam': 1.,
                 'y': None, 'x0': 1., 'mu': 0., 'sigma': 1.})

    for X in (sp.merton_jumpdiff_process, sp.mjd):
        assert_(kf(X)(vshape=2).params ==
                {'paths': 1, 'vshape': 2, 'dtype': None,
                 'steps': None, 'i0': 0, 'info': None,
                 'getinfo': True, 'method': 'euler', 'ptype': int,
                 'dw': None, 'corr': None, 'rho': None,
                 'dj': None, 'dn': None, 'lam': 1.,
                 'a': 0., 'b': 1., 'x0': 1., 'mu': 0., 'sigma': 1.})

    for X in (sp.kou_jumpdiff_process, sp.kou):
        assert_(kf(X)(vshape=2).params ==
                {'paths': 1, 'vshape': 2, 'dtype': None,
                 'steps': None, 'i0': 0, 'info': None,
                 'getinfo': True, 'method': 'euler', 'ptype': int,
                 'dw': None, 'corr': None, 'rho': None,
                 'dj': None, 'dn': None, 'lam': 1.,
                 'a': 0.5, 'b': 0.5, 'pa': 0.5,
                 'x0': 1., 'mu': 0., 'sigma': 1.})
