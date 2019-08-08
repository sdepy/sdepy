"""
=================================
FORMAL TESTS ON THE PROCESS CLASS
=================================
"""
from .shared import *
process = sp.process
piecewise = sp.piecewise
_piecewise_constant_process = sp.infrastructure._piecewise_constant_process


def test_import():
    import sdepy as sdepy1
    assert_(sp.process is sdepy1.process)


# -----------------------------
# test the process constructors
# -----------------------------

# enumerate test cases and launch tests
def test_constructor():
    np.random.seed(SEED)

    # do cases
    t = [2., (5.,), (0, 1.), np.linspace(0., 4., 12)]
    paths = [1, 10]
    vshape = [(), (2,), (3, 2)]
    dtype = [None, float, int, np.float64, np.float32, np.float16]
    do(process_constructor, t, paths, vshape, dtype)

    # dtype of timeline is preserved
    p = process(np.arange(5, dtype=int), c=10., dtype=np.float16)
    assert_(p.t.dtype == np.dtype(int))
    assert_(p.dtype == np.dtype(np.float16))

    # incompatible timeline and body
    assert_raises(ValueError, process, (1, 2), x=np.zeros((3, 2)))


# case testing
def process_constructor(t, paths, vshape, dtype):
    tt = np.asarray(t)
    if tt.ndim == 0:
        tt = tt.reshape(1)

    # x constructor
    x = np.random.random(tt.shape + vshape + (paths,))
    p = process(t, x=x, dtype=dtype)
    assert_array_equal(p.t, tt)
    assert_(isinstance(p.t, np.ndarray))
    assert_(p.shape == p.t.shape + p.vshape + (p.paths,))
    assert_array_equal(p, x.astype(dtype))
    assert_(p.dtype == np.dtype(dtype))

    # v constructor
    v = np.random.random(tt.shape + vshape)
    p = process(t, v=v, dtype=dtype)
    assert_array_equal(p.t, tt)
    assert_(isinstance(p.t, np.ndarray))
    assert_(p.shape == p.t.shape + p.vshape + (p.paths,))
    assert_array_equal(p, v[..., np.newaxis].astype(dtype))
    assert_(p.dtype == np.dtype(dtype))

    # c constructor
    c = np.random.random(vshape)
    p = process(t, c=c, dtype=dtype)
    assert_array_equal(p.t, tt)
    assert_(isinstance(p.t, np.ndarray))
    assert_(p.shape == p.t.shape + p.vshape + (p.paths,))
    assert_(p.shape == (max(1, tt.size),) + vshape + (1,))
    for i in range(tt.size):
        assert_array_equal(p[i, ..., 0], c.astype(dtype))
        pass
    assert_(p.dtype == np.dtype(dtype))

    # only one of x, v, c
    assert_raises(ValueError, process, t, x=x, v=v)
    assert_raises(ValueError, process, t, x=x, c=c)
    assert_raises(ValueError, process, t, c=c, v=v)
    assert_raises(ValueError, process, t, x=x, v=v, c=c)

    # only zero or one dimensional timeline
    assert_raises(ValueError, process, tt[np.newaxis, np.newaxis], x=x)

    # test interp_kind attribute
    assert_(process.interp_kind == 'linear')
    assert_(p.interp_kind == 'linear')
    p = process((1., 2.), v=(10., 20.))
    assert_allclose(p(1.2), np.array([12.]), rtol=eps(float))
    p.interp_kind = 'nearest'
    assert_allclose(p(1.2), p[0], rtol=eps(float))


# -----------------
# test broadcasting
# -----------------

# enumerate test cases and launch tests
def test_broadcasting():
    np.random.seed(SEED)

    # do cases
    t = [np.linspace(0., 4., 12)]
    paths = [5]
    vshape = [(), (2,), (3, 2)]
    assert_((3, 2) in vshape)  # expected by process_broadcasting
    do(process_broadcasting, t, paths, vshape)


# case testing
def process_broadcasting(t, paths, vshape):
    p = process(t, x=np.random.random(t.shape + vshape + (paths,)))
    q = process(t.copy(),
                x=np.random.random(t.shape + vshape + (paths,)))

    # same processes
    a1 = p + q
    assert_(a1.shape == p.shape)

    # many vs 1 path
    a1 = p + q['p', 0]
    assert_(a1.shape == p.shape)
    a1 = q['p', 0] + p
    assert_(a1.shape == p.shape)

    # many vs 1 value
    if vshape != ():
        a1 = p + q['v', :1]
        assert_(a1.shape == p.shape)
        a1 = q['v', :1] + p
        assert_(a1.shape == p.shape)

    # two constant processes (based on different t values)
    p1, q1 = p['t', 0], q['t', 1]
    assert_(p1.t[0] != q1.t[0])
    a1 = p1 + q1
    assert_(a1.shape == (1,) + p.vshape + (p.paths,))

    # one constant and one non constant process
    a = p + q['t', 0]
    assert_(a.t.shape == p.t.shape)
    assert_(a.t is p.t)
    a = p['t', 0] + q
    assert_(a.t.shape == q.t.shape)
    assert_(a.t is q.t)

    # multiple inputs
    if vshape != ():
        a = p + q['p', 0] + p['t', 0] + q + p['v', :1]
        assert(a.shape == p.shape)

    # operations mixing processes and arrays
    if vshape != ():
        u, v = p.x[0], q.x[..., :1]
        a = u*p + v*q['p', 0] + p['t', 0] + q.x + p['v', :1]*2 + v
        assert(a.shape == p.shape)
        a = q['p', 0]*v + p*u + q.x + p['v', :1]*2 + v + p['t', 0]
        assert(a.shape == p.shape)

    # addition test for assert_raises
    def f(p, q):
        return p + q

    # failing: different paths number
    assert_raises(ValueError, f, p['p', :2], q['p', 2:5])

    # failing: different timelines
    assert_raises(ValueError, f, p['t', :2], q['t', 2:4])

    if vshape == (3, 2):
        # failing: vshapes of different length
        p1, q1 = p, q['v', :2, 0]
        assert_(p1.vshape == (3, 2) and q1.vshape == (2,))
        assert_raises(ValueError, f, p1, q1)

        # failing: incompatible vshapes of same length
        p1, q1 = p, q['v', :2]
        assert_(p1.vshape == (3, 2) and q1.vshape == (2, 2))
        assert_raises(ValueError, f, p1, q1)

        # failing: vshapes of different length,
        # broadcasting of corresponding arrays is ok
        p1, q1 = p['t', 0]['v', :1, :1], q['v', :, 0]
        assert_(p1.shape == (1, 1, 1, paths) and
                q1.shape == t.shape + (3, paths))
        x = p1.x + q1.x  # ok as arrays
        assert_(x.shape == (1,) + t.shape + vshape[:1] + (paths,))
        assert_raises(ValueError, f, p1, q1)  # ko as processes


# -----------
# test interp
# -----------

# enumerate test cases and launch tests
def test_interp():
    np.random.seed(SEED)

    # do cases on t, s, paths and vshape
    t = [2+np.zeros(1), np.linspace(1., 4., 12)]
    s = [1., (2,), (1, 2, 3.), np.linspace(-3, 10, 4),
         np.arange(5*7).reshape(5, 7)/10]
    paths = [1, 10]
    vshape = [(), (2,), (3, 2)]
    dtype = [None]
    kind = [None]
    do(process_interp, t, s, paths, vshape, dtype, kind)

    # do cases on dtype and interpolation kind
    t = [np.linspace(1., 4., 12)]
    s = [(2,)]
    paths = [10]
    vshape = [(3,)]
    dtype = [None, float, np.float64, np.float32, np.float16]
    kind = [None, 'linear', 'nearest', 'quadratic', 'cubic']
    do(process_interp, t, s, paths, vshape, dtype, kind)


# case testing
def process_interp(t, s, paths, vshape, dtype, kind):

    # create a process with nontrivial values to interpolate
    p = process(t, x=np.empty(t.shape + vshape + (paths,), dtype=dtype))
    p[:] = p.tx**2
    p[:] *= (1 + np.arange(p[0].size)).reshape((1,) + vshape + (paths,))

    # interpolation along the process timeline
    y = p(t, kind=kind)
    assert_allclose(y, p, rtol=eps(dtype))

    # generic interpolation - formal tests
    y = p(s, kind=kind)
    z = p.interp(kind=kind)(s)
    assert_array_equal(y, z)
    assert_(isinstance(y, np.ndarray) and not isinstance(y, process))
    assert_(y.shape == np.asarray(s).shape + vshape + (paths,))
    assert_(y.dtype == p.dtype)

    # increments - formal tests
    ds = 0.1
    w = p(s, ds, kind=kind)
    assert_array_equal(w, p(np.asarray(s) + ds, kind=kind) - p(s, kind=kind))

    # boundaries
    tmin, tmax = p.t.min(), p.t.max()
    assert_allclose(p(tmin, kind=kind), p(tmin - 1, kind=kind),
                    rtol=eps(dtype))
    assert_allclose(p(tmax, kind=kind), p(tmax + 1, kind=kind),
                    rtol=eps(dtype))

    # rebase
    s = np.asarray(s)
    if s.ndim <= 1:
        q = p.rebase(s, kind=kind)
        assert_(isinstance(q, process))
        assert_(q.vshape == p.vshape and q.paths == p.paths)
        if np.asarray(s).ndim > 0:
            assert_allclose(q, p(s, kind=kind), rtol=eps(dtype))


# simple stand alone test on interpolated values
def test_interp_values():
    np.random.seed(SEED)

    # check interpolated values
    p = process(t=(1, 2, 3.), v=(1, 4, 9.))
    assert_allclose(p((1.5, 2.5)),
                    np.array(((1 + 4)/2, (4 + 9)/2)).reshape(2, 1),
                    rtol=eps(p.dtype))
    assert_allclose(p(1, 0.5), (1 + 4)/2 - 1, rtol=eps(p.dtype))


# ----------------
# test 't' getitem
# ----------------

# enumerate test cases and launch tests
def test_t_getitem():
    np.random.seed(SEED)

    # do cases
    t = [np.linspace(1., 4., 12)]
    paths = [1, 10]
    vshape = [(), (2,), (3, 2)]
    do(process_t_getitem, t, paths, vshape)


# case testing
def process_t_getitem(t, paths, vshape):
    p = process(t, x=np.random.random(t.shape + vshape + (paths,)))

    def verify(q, newt, newx):
        assert_(isinstance(q, process))
        assert_array_equal(q.t, newt)
        assert_array_equal(q.x, newx)

    q = p['t', ()]
    verify(q, p.t, p)
    q = p['t', 0]
    verify(q, p.t[:1], p[:1])
    q = p['t', 0:2]
    verify(q, p.t[0:2], p[0:2])
    q = p['t', :-2]
    verify(q, p.t[:-2], p[:-2])
    q = p['t', 2:-2]
    verify(q, p.t[2:-2], p[2:-2])
    q = p['t', ::2]
    verify(q, p.t[::2], p[::2])
    i = [0, 3, 5]
    q = p['t', i]
    verify(q, p.t[i], p[i])
    i = (p.t > 2)
    q = p['t', i]
    verify(q, p.t[i], p[i])


# ----------------
# test 'p' getitem
# ----------------

# enumerate test cases and launch tests
def test_p_getitem():
    np.random.seed(SEED)

    # do cases
    t = [np.zeros(1) + 2, np.linspace(1., 4., 12)]
    paths = [10]
    vshape = [(), (2,), (3, 2)]
    do(process_p_getitem, t, paths, vshape)


# case testing
def process_p_getitem(t, paths, vshape):
    p = process(t, x=np.random.random(t.shape + vshape + (paths,)))

    def verify(q, newt, newx):
        assert_(isinstance(q, process))
        assert_array_equal(q.t, newt)
        assert_array_equal(q.x, newx)

    q = p['p', ()]
    verify(q, p.t, p)
    q = p['p', 0]
    verify(q, p.t, p[..., :1])
    q = p['p', 0:2]
    verify(q, p.t, p[..., 0:2])
    q = p['p', :-2]
    verify(q, p.t, p[..., :-2])
    q = p['p', 2:-2]
    verify(q, p.t, p[..., 2:-2])
    q = p['p', ::2]
    verify(q, p.t, p[..., ::2])
    i = [0, 3, 5]
    q = p['p', i]
    verify(q, p.t, p[..., i])
    ii = np.array(i)
    q = p['p', ii]
    verify(q, p.t, p[..., i])
    i = (np.arange(p.paths) % 2 == 0)
    q = p['p', i]
    verify(q, p.t, p[..., i])


# ----------------
# test 'v' getitem
# ----------------

# enumerate test cases and launch tests
def test_v_getitem():
    np.random.seed(SEED)

    # do cases
    t = [np.zeros(1) + 2, np.linspace(1., 4., 11)]
    paths = [13]
    do(process_v_getitem, t, paths)


# case testing
def process_v_getitem(t, paths):

    def verify(q, newt, newx):
        assert_(isinstance(q, process))
        assert_array_equal(q.t, newt)
        assert_array_equal(q.x, newx)

    def iall(p):
        for i in np.ndindex(p.vshape):
            verify(p[('v',) + i], p.t, p[(slice(None),) + i + (slice(None),)])

    # vshape = ()
    p = process(t, x=np.random.random(t.shape + () + (paths,)))
    q = p['v', ()]
    verify(q, p.t, p)
    iall(p)

    def ierr():
        return p['zz']
    assert_raises(IndexError, ierr)

    # vshape = (5,)
    p = process(t, x=np.random.random(t.shape + (5,) + (paths,)))
    iall(p)
    q = p['v', 0:2]
    verify(q, p.t, p[:, 0:2, :])
    q = p['v', :-1]
    verify(q, p.t, p[:, :-1, :])
    q = p['v', 1:-1]
    verify(q, p.t, p[:, 1:-1, :])
    q = p['v', ::2]
    verify(q, p.t, p[:, ::2, :])
    i = [0, 2, 3]
    q = p['v', i]
    verify(q, p.t, p[:, i, :])
    ii = np.array(i)
    q = p['v', ii]
    verify(q, p.t, p[:, i, :])
    i = (np.arange(5) % 2 == 0)
    q = p['v', i]
    verify(q, p.t, p[:, i, :])

    # vshape = (5, 7)
    p = process(t, x=np.random.random(t.shape + (5, 7) + (paths,)))
    iall(p)
    q = p['v', 0:2]
    verify(q, p.t, p[:, 0:2, :])
    q = p['v', :-1]
    verify(q, p.t, p[:, :-1, :])
    q = p['v', 1:-1]
    verify(q, p.t, p[:, 1:-1, :])
    q = p['v', ::2]
    verify(q, p.t, p[:, ::2, :])
    i = [0, 2]
    q = p['v', i]
    verify(q, p.t, p[:, i, :])
    ii = np.array(i)
    q = p['v', ii]
    verify(q, p.t, p[:, i, :])
    j = (np.arange(5) % 2 == 0)
    q = p['v', j]
    verify(q, p.t, p[:, j, :])
    q = p['v', 0:2, -1]
    verify(q, p.t, p[:, 0:2, -1, :])
    q = p['v', 0:2, 1:-1]
    verify(q, p.t, p[:, 0:2, 1:-1, :])

    ii = [[2, 3], [4, 0]]
    jj = [[6, 5], [4, 3]]
    q = p['v', ii, jj]
    verify(q, p.t, p[:, ii, jj, :])
    iii, jjj = np.array(ii), np.array(jj)
    q = p['v', iii, jjj]
    verify(q, p.t, p[:, ii, jj, :])

    jj = (np.arange(5*7).reshape(5, 7) % 2 == 0)
    q = p['v', jj]
    verify(q, p.t, p[:, jj, :])


# ----------------
# test properties
# ----------------

# enumerate test cases and launch tests
def test_properties():
    np.random.seed(SEED)
    p = process(t=(1., 2, 4), x=np.random.random((3, 5, 7, 11)))

    def sharemem(a, b):
        return a.__array_interface__['data'][0] == \
               b.__array_interface__['data'][0]

    assert_array_equal(p, p.x)
    assert_(sharemem(p, p.x))
    assert_(not hasattr(p.x, 't'))
    assert_(not isinstance(p.x, process))

    assert_(p.paths == 11)
    assert_(p.vshape == (5, 7))
    assert_(p.shape == p.t.shape + p.vshape + (p.paths,))

    assert_(p.tx.shape == p.t.shape + (1, 1, 1))
    dt = p.dt
    assert_array_equal(dt, np.array((1., 2)))
    assert_array_equal(p.dtx, dt.reshape(2, 1, 1, 1))

    # do cases
    t = [2., (5.,), (0, 1.), np.linspace(0., 4., 12)]
    paths = [1, 10]
    vshape = [(), (2,), (3, 2)]
    dtype = [None, float, int, np.float64, np.float32, np.float16]
    do(process_properties, t, paths, vshape, dtype)


# case testing
def process_properties(t, paths, vshape, dtype):
    tt = np.asarray(t)
    if tt.ndim == 0:
        tt = tt.reshape(1)

    p = process(t, x=np.random.random(tt.shape + vshape + (paths,)),
                dtype=dtype)
    assert_(p.shape == p.t.shape + p.vshape + (p.paths,))
    assert_array_equal(p.t, p.tx.flatten())
    assert_array_equal(p.dt, p.dtx.flatten())
    assert_array_equal(p.dt, np.diff(p.t))

    # check broadcasting
    a = p + p.tx
    assert_(a.shape == p.shape)
    b = np.diff(p, axis=0) + p.dtx
    assert_(b.shape == p[:-1].shape)


# ----------------
# test shapeas
# ----------------

def test_shapeas():
    np.random.seed(SEED)

    def mkp(vshape):
        return process(t=(0, 1), x=np.random.random((2,) + vshape + (3,)))

    for vshape in ((), (5,), (5, 7)):
        p = mkp(vshape)
        assert_(p.shapeas(vshape).shape == p.shape)
        assert_(p.shapeas(p).shape == p.shape)

    p = mkp(())
    assert_(p.shapeas((5,)).shape == (2, 1, 3))
    assert_(p.shapeas((5, 7)).shape == (2, 1, 1, 3))

    p = mkp((5,))
    assert_(p.shapeas((5, 7)).shape == (2, 1, 5, 3))
    assert_(p.shapeas((5, 7, 11)).shape == (2, 1, 1, 5, 3))

    p = mkp((5, 7))
    assert_(p.shapeas((5, 7, 11)).shape == (2, 1, 5, 7, 3))

    p = mkp((1, 1,))
    assert_(p.shapeas(()).shape == (2, 3))

    p = mkp((1, 5))
    assert_(p.shapeas((5,)).shape == (2, 5, 3))
    assert_raises(ValueError, lambda: p.shapeas(()))

    p = mkp((1, 1,))
    for vshape in ((), (5,), (5, 7), (5, 7, 2)):
        assert(p.shapeas(vshape).shapeas(p.vshape).shape == p.shape)


# ----------------
# test copying
# ----------------

def test_copy():
    np.random.seed(SEED)

    p = process(t=(1, 2, 3), x=np.random.random((3, 5, 7, 11)))
    q = p.pcopy()
    qt = p.tcopy()
    qx = p.xcopy()

    def sharemem(a, b):
        return a.__array_interface__['data'][0] == \
               b.__array_interface__['data'][0]

    assert_(not sharemem(q.t, p.t))
    assert_(not sharemem(q.x, p.x))

    assert_(not sharemem(qt.t, p.t))
    assert_(sharemem(qt.x, p.x))

    assert_(sharemem(qx.t, p.t))
    assert_(not sharemem(q.x, p.x))


# -----------------------
# test summary operations
# -----------------------

def test_summary():
    np.random.seed(SEED)

    p = process(t=(1, 2, 3), x=1 + np.random.random((3, 5, 7, 11)),
                dtype=np.float64)

    funcs = ('min', 'max', 'sum', 'mean', 'var', 'std')

    # summary across paths
    for f in funcs:
        # test values
        a = getattr(p, 'p' + f)()
        b = getattr(np, f)(p, axis=-1)[..., np.newaxis]
        assert_allclose(a, b, rtol=eps(p.dtype))
        # test out parameter
        y = np.full((3, 5, 7, 1), np.nan)
        a = getattr(p, 'p' + f)(out=y)
        assert_(a.base is y)
        assert_allclose(y, b, rtol=eps(p.dtype))
    for f in ('sum', 'mean', 'var', 'std'):
        # test dtype parameter
        a = getattr(p, 'p' + f)(dtype=np.float32)
        assert_(a.dtype == np.float32)
    # test ddof parameter for pvar and pstd
    a = p.pvar(ddof=1)
    assert_allclose(a, p.var(ddof=1, axis=-1)[..., np.newaxis],
                    rtol=eps(p.dtype))
    a = p.pstd(ddof=1)
    assert_allclose(a, p.std(ddof=1, axis=-1)[..., np.newaxis],
                    rtol=eps(p.dtype))

    # summary across values
    for f in funcs:
        # test values
        a = getattr(p, 'v' + f)()
        b = getattr(np, f)(p, axis=(1, 2))
        assert_allclose(a, b, rtol=eps(p.dtype))
        # test out parameter
        y = np.full((3, 11), np.nan)
        a = getattr(p, 'v' + f)(out=y)
        assert_(a.base is y)
        assert_allclose(b, y, rtol=eps(p.dtype))
    for f in ('sum', 'mean', 'var', 'std'):
        # test dtype parameter
        a = getattr(p, 'v' + f)(dtype=np.float32)
        assert_(a.dtype == np.float32)
    # test ddof parameter for pvar and pstd
    a = p.vvar(ddof=1)
    assert_allclose(a, p.var(ddof=1, axis=(1, 2)),
                    rtol=eps(p.dtype))
    a = p.vstd(ddof=1)
    assert_allclose(a, p.std(ddof=1, axis=(1, 2)),
                    rtol=eps(p.dtype))

    # summary along the timeline
    for f in funcs:
        # test values
        a = getattr(p, 't' + f)()
        b = getattr(np, f)(p, axis=0)[np.newaxis, ...]
        assert_allclose(a, b, rtol=eps(p.dtype))
        # test out parameter
        y = np.full((1, 5, 7, 11), np.nan)
        a = getattr(p, 't' + f)(out=y)
        assert_(a.base is y)
        assert_allclose(y, b, rtol=eps(p.dtype))
    for f in ('sum', 'mean', 'var', 'std'):
        # test dtype parameter
        a = getattr(p, 't' + f)(dtype=np.float32)
        assert_(a.dtype == np.float32)
    # test ddof parameter for tvar and tstd
    a = p.tvar(ddof=1)
    assert_allclose(a, p.var(ddof=1, axis=0)[np.newaxis, ...],
                    rtol=eps(p.dtype))
    a = p.tstd(ddof=1)
    assert_allclose(a, p.std(ddof=1, axis=0)[np.newaxis, ...],
                    rtol=eps(p.dtype))

    # test tcumsum values
    a = p.tcumsum()
    b = np.cumsum(p, axis=0)
    assert_allclose(a, b, rtol=3*eps(p.dtype))
    # test tcumsum out parameter
    y = np.full((3, 5, 7, 11), np.nan)
    a = p.tcumsum(out=y)
    assert_(a.base is y)
    assert_allclose(y, b, rtol=3*eps(p.dtype))
    # test tcumsum dtype parameter
    a = p.tcumsum(dtype=np.float32)
    assert_(a.dtype == np.float32)

    # numpy summary operations
    for f in funcs:
        a = getattr(p, f)()
        b = getattr(np, f)(p)
        c = getattr(np, f)(p.x)
        assert_(isinstance(a, np.ndarray) and not isinstance(a, process))
        assert_(isinstance(b, np.ndarray) and not isinstance(b, process))
        assert_allclose(a, c)
        assert_allclose(b, c)


# ----------------------------------------
# test increments, derivative and integral
# ----------------------------------------

def test_increments():
    np.random.seed(SEED)

    t = np.linspace(1, 2, 100)
    t *= t
    p = process(t, x=1 + np.random.random(t.shape + (2, 3) + (5,)),
                dtype=np.float64)

    dp = p.tdiff()
    assert_array_equal(dp.t, p.t[:-1])
    assert_allclose(dp, np.diff(p, axis=0), rtol=eps(p.dtype))

    dp = p.tdiff(fwd=False, dt_exp=3)
    assert_array_equal(dp.t, p.t[1:])
    assert_allclose(dp, np.diff(p, axis=0) /
                    np.diff(p.t).reshape(-1, 1, 1, 1)**3,
                    rtol=eps(p.dtype))

    dp = p.tder()
    assert_allclose(dp, p.tdiff(dt_exp=1), rtol=eps(p.dtype))
    q = dp.tint()
    assert_allclose(q, p['t', :-1] - p[:1], rtol=eps(p.dtype)*10,
                    atol=eps(p.dtype)*10)
    ip = p.tint()
    q = ip.tder()
    assert_allclose(q, p['t', :-1], rtol=eps(p.dtype)*10)

    t = np.linspace(1, 2, 1000)
    tx = t.reshape(-1, 1, 1)
    p = process(t, x=1 + tx**1.5)
    assert_allclose(p.tder(), 1.5*tx[:-1]**(0.5),
                    rtol=0.001)
    assert_allclose(p.tint(), (tx + tx**2.5/2.5) - (1 + 1/2.5),
                    rtol=0.001)


# ----------------
# test chf and cdf
# ----------------

# enumerate test cases and launch tests
def test_chf_cdf():
    np.random.seed(SEED)

    t = [2., (5.,), (0, 1.), np.linspace(0., 4., 17)]
    paths = [1, 19]
    vshape = [(), (3,), (2, 3)]
    s = [1, (2, 3), np.linspace(0, 1, 5*7).reshape(5, 7)]
    u = [11, (22, 33), np.linspace(0, 1, 11*13).reshape(11, 13)]
    do(process_chf_cdf, t, s, u, paths, vshape)


# case testing
def process_chf_cdf(t, s, u, paths, vshape):
    tt = np.asarray(t)
    if tt.ndim == 0:
        tt = tt.reshape(1)
    ss, uu = np.asarray(s), np.asarray(u)

    p = process(t, x=np.random.random(tt.shape + vshape + (paths,)))

    f = p.chf(u)
    assert_(f.shape == tt.shape + uu.shape + vshape)
    f = p.chf(u=u)
    assert_(f.shape == tt.shape + uu.shape + vshape)
    g = p.chf(p.t, u)
    assert_allclose(f, g, rtol=eps(f.dtype))
    h = p.chf(s, u)
    assert_(h.shape == ss.shape + uu.shape + vshape)
    assert_raises(TypeError, p.chf)

    f = p.cdf(u)
    assert_(f.shape == tt.shape + uu.shape + vshape)
    f = p.cdf(x=u)
    assert_(f.shape == tt.shape + uu.shape + vshape)
    g = p.cdf(p.t, u)
    assert_allclose(f, g, rtol=eps(f.dtype))
    h = p.cdf(s, u)
    assert_(h.shape == ss.shape + uu.shape + vshape)
    assert_array_equal(p.cdf(x=-0.001), 0)
    assert_array_equal(p.cdf(x=1.001), 1)
    assert_raises(TypeError, p.cdf)


# -------------------------------
# test piecewise constant process
# -------------------------------

# enumerate test cases and launch tests
def test_piecewise():
    np.random.seed(SEED)

    dtype = [np.float64, np.float32, np.float16]
    paths = [None, 5]
    vshape = [(3,), (2, 3)]
    mode = [None, 'mid', 'forward', 'backward']
    shift = [0, -2, -4]
    do(tst_piecewise, [np.float], paths, vshape, mode, shift)
    do(tst_piecewise,      dtype, paths,   [()], mode,   [0])

    with assert_raises(ValueError):
        p = piecewise((1, 2), v=(10, 20), mode='zzz')
    with assert_warns(DeprecationWarning):
        p = _piecewise_constant_process((1, 2), v=(10, 20))


# case testing
def tst_piecewise(dtype, paths, vshape, mode, shift):

    t = np.array((1, 2, 3), dtype=dtype) + shift
    if paths is None:
        paths = 1
        v = vv = 1 + np.random.random((3,) + vshape).astype(dtype)
        x = v.reshape(v.shape + (1,))
        xx = None
    else:
        x = xx = 1 + np.random.random((3,) + vshape + (paths,)).astype(dtype)
        vv = None

    if mode is None:
        p = piecewise(t, x=xx, v=vv)  # default mode is 'mid'
    else:
        p = piecewise(t, x=xx, v=vv, mode=mode)

    # common checks
    q = p*p + 2
    assert_(isinstance(q, process))
    assert_(p.interp_kind == 'nearest')
    assert_(p.paths == paths and p.vshape == vshape)
    assert_(p(0).dtype == dtype)

    # test values
    if mode in (None, 'mid'):
        s = np.array((0.50,    1, 1.49, 1.51, 2.49, 2.51,    3,  3.5))
        y = np.stack((x[0], x[0], x[0], x[1], x[1], x[2], x[2], x[2]))
        assert_allclose(p(s + shift), y)
    elif mode == 'forward':
        s = np.array((0.50,    1,  1.5,    2, 2.01,  2.5,    3, 3.01,  3.5))
        y = np.stack((x[0], x[0], x[0], x[0], x[1], x[1], x[1], x[2], x[2]))
        assert_allclose(p(s + shift), y)
    elif mode == 'backward':
        s = np.array((0.50,    1, 1.01,  1.5,    2, 2.01,  2.5,    3,  3.5))
        y = np.stack((x[0], x[0], x[1], x[1], x[1], x[2], x[2], x[2], x[2]))
        assert_allclose(p(s + shift), y)

    # test degenerate case of single time point
    q = piecewise(t=0., x=x[:1])
    assert_allclose(q((-3, 0, 3)), np.stack((x[0], x[0], x[0])))


# ---------------------------
# test no override commitment
# ---------------------------

def get_override(subclass, parent):
    """Return set of attributes and methods of parent
    overridden by subclass"""
    assert_(issubclass(subclass, parent))
    return set(vars(parent)).intersection(vars(subclass))


def test_no_override():

    allowed = {
        '__doc__',
        '__new__',
        '__array_finalize__',
        '__getitem__',
        '__array_prepare__',
        '__array_priority__',
        '__array_wrap__',
    }

    # sdepy.process should not override any numpy.ndarray
    # method other than those listed above
    assert_(get_override(process, np.ndarray) == allowed)

    # validate test
    class failing(process):

        def sum(self, *args):
            pass

    assert_('sum' in get_override(failing, np.ndarray))
