"""
===============================================
QUANTITATIVE (SLOW) TESTS ON REALIZED PROCESSES
===============================================

PATHS_HD or PATHS_LD paths are computed for each
process, according to the QUANT_TEST_MODE flag
(see 'shared.py' module), on a timeline of 50 points,
using 50 or 200 integration steps dependant on the process type.

Tests on pdf and cdf labeled 'pdf' and 'cdf1'
cumulate 10 runs using update, pdf and cdf methods
of the montecarlo class, while tests labeled 'cdf2' rely
on a single run and on the cdf method of the process class.

Plots for paths inspection:
30 paths, 1000 time points.
"""
from .shared import *
from numpy import exp, log
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


# --------------------------
# functions for common tasks
# --------------------------

def bfig():
    return plt.figure(figsize=(12, 6))


def sfig():
    return plt.figure(figsize=(6, 3))


def check_values(context, Xid, t, *tests,
                 fig_id='',
                 err_expected=None, err_realized=None,
                 mode=None, xlabel='t', brk=None, PATHS=1):
    """
    Common tasks:
       - get a list of expected and realized values depending on t;
       - compute realized errors;
       - compare realized vs expected errors;
       - plot results in a single plot with subplots;
       - save the plot;
       - store realized errors in 'err_realized' dict.
    """
    prefix = context + '_' + Xid

    for id, x_realized, x_expected in tests:
        assert_(x_expected.shape == x_realized.shape)
        key = prefix + '_' + id
        if mode == 'rel':
            delta = np.abs(x_expected - x_realized)/x_expected
        elif mode == 'wabs':
            delta = np.abs(x_expected - x_realized) / \
                np.abs(x_expected).max()
        elif mode == 'abs':
            delta = np.abs(x_expected - x_realized)
        else:
            raise ValueError('unrecognized mode')
        mean_err = delta.mean()
        max_err = delta.max()
        if VERBOSE:
            print('{:45}{:10.6f} {:10.6f}'
                  .format(key + ' (mean err)', mean_err, err_expected[key][0]))
            print('{:45}{:10.6f} {:10.6f}'
                  .format(key + ' (max err)', max_err, err_expected[key][1]))
        assert_quant(mean_err < err_expected[key][0])
        assert_quant(max_err < err_expected[key][1])
        err_realized[key] = (mean_err, max_err)
        if brk == key:
            raise RuntimeError('a break was set at {}'.format(key))
    if PLOT:
        print('plotting...')
        fig = bfig()
        plots = ((111,), (121, 122), (221, 222, 223,),
                 (221, 222, 223, 224))
        for plot, (id, x_realized, x_expected) in \
                zip(plots[len(tests)-1], tests):
            plt.subplot(plot)
            plt.title('{} {}: simulation with {} paths'.format(Xid, id, PATHS))
            plt.xlabel(xlabel)
            if 'stats' in fig_id:
                plt.axis(ymin=0, ymax=1.1*x_expected.max())
            if 'chf' in fig_id:
                a, b = x_expected.max(), x_expected.min()
                plt.axis(ymin=1.1*min(0, b), ymax=1.1*max(0, a))
            plt.plot(t, x_expected, label=id + ' expected')
            plt.plot(t, x_realized, '--', label=id + ' realized')
            plt.legend()
        plt.tight_layout()
        save_figure(fig, prefix + '_' + fig_id)
        plt.close(fig)


def hw2f_wrapped(**args):
    X = sp.hull_white_process(**args, factors=2)

    def f(*var, **args):
        return X(*var, **args)

    return f


def full_heston_wrapped(**args):
    X = sp.full_heston_process(**args)

    def f(*var, **args):
        return X(*var, **args)[0]

    return f


def hw2f_F_setup(x0, theta, k, sigma):
    x0, theta, k, sigma = \
        [np.asarray(z) for z in (x0, theta, k, sigma)]

    # alert if params stated in this module do not comply
    # with expected shape
    assert_(all(z.ndim == 0 or (z.ndim >= 1 and z.shape[-1] == 1)
                for z in (x0, theta, k, sigma)))

    # remove paths last axis from x0, theta, k, sigma
    x0, theta, k, sigma = \
        [z if z.ndim <= 1 else z[..., 0]
         for z in (x0, theta, k, sigma)]
    return x0, theta, k, sigma


def hw2f_F_nopaths(F):
    def G(*var, **args):
        x = F(*var, **args)
        assert_(x.shape[-1] == 1)
        return x[..., 0]
    return F


def hw2f_F_wrap1(F):
    def G(t, x0, theta, k, sigma, rho):
        return F(t, *hw2f_F_setup(x0, theta, k, sigma), rho=rho)
    return F


def hw2f_F_wrap2(F):
    def G(t, x, x0, theta, k, sigma, rho):
        return F(t, x, *hw2f_F_setup(x0, theta, k, sigma), rho=rho)
    return F


# ---------------------------------
# Tests with fixed parameter values
# context = 'quant'
# ---------------------------------

# main test
@slow
@quant
def test_quant():
    if QUANT_TEST_MODE == 'HD':
        tst_quant(context='quant5', PATHS=PATHS_HD)
    elif QUANT_TEST_MODE == 'LD':
        tst_quant(context='quant2', PATHS=PATHS_LD)
    else:
        raise ValueError()


def tst_quant(context, PATHS):
    """Setup test cases, do actual testing invoking 'quant',
    save results.
    """
    err_expected = load_errors(context)
    err_realized = {}

    # setup test cases
    # ----------------
    wiener_args = dict(sigma=0.3, mu=0.1, x0=log(10))
    lognorm_args = dict(sigma=0.3, mu=0.1, x0=10)
    oruh_args = dict(x0=0.20, theta=0.10, k=1, sigma=.03,
                     steps=200)
    hw2f_args = dict(x0=((0.30,), (0.05,)),
                     theta=((0.10,), (0.20,)), k=((1.,), (0.3,)),
                     sigma=((0.05,), (0.08,)), rho=-0.5,
                     steps=200)
    cir_args = dict(x0=0.30, theta=0.5, k=3., xi=.50,
                    steps=200)

    heston_args = dict(x0=10., sigma=0.2, mu=0.1, y0=.5, theta=1.5,
                       k=3., xi=2., rho=-0.5,
                       steps=200)
    mjd_args = dict(sigma=0.3, mu=0.1, x0=6, lam=2., a=-0.2, b=0.1,
                    steps=200)
    kou_args = dict(sigma=0.3, mu=0.1, x0=6, lam=2., pa=.80, a=0.2, b=0.3,
                    steps=200)

    wiener_F = (sp.wiener_mean, sp.wiener_var, sp.wiener_std,
                sp.wiener_pdf, sp.wiener_cdf, sp.wiener_chf)
    lognorm_F = (sp.lognorm_mean, sp.lognorm_var, sp.lognorm_std,
                 sp.lognorm_pdf, sp.lognorm_cdf, sp.lognorm_log_chf)
    oruh_F = (sp.oruh_mean, sp.oruh_var, sp.oruh_std,
              sp.oruh_pdf, sp.oruh_cdf, None)
    hw2f_F = (hw2f_F_nopaths(sp.hw2f_mean),
              hw2f_F_nopaths(sp.hw2f_var),
              hw2f_F_nopaths(sp.hw2f_std),
              hw2f_F_nopaths(sp.hw2f_pdf),
              hw2f_F_nopaths(sp.hw2f_cdf), None)
    cir_F = (sp.cir_mean, sp.cir_var, sp.cir_std,
             sp.cir_pdf, None, None)
    heston_F = (sp.heston_log_mean, sp.heston_log_var, sp.heston_log_std,
                sp.heston_log_pdf, None, sp.heston_log_chf)
    mjd_F = (sp.mjd_mean, None, None,
             sp.mjd_log_pdf, None, sp.mjd_log_chf)
    kou_F = (sp.kou_mean, None, None,
             sp.kou_log_pdf, None, sp.kou_log_chf)

    cases = (
        ('wiener2', sp.wiener_process, wiener_F, {**wiener_args,
                                                  **dict(steps=50)}),
        ('lognorm2', sp.lognorm_process, lognorm_F, {**lognorm_args,
                                                     **dict(steps=50)}),
        ('oruh', sp.ornstein_uhlenbeck_process, oruh_F, oruh_args),
        ('hw2f', hw2f_wrapped, hw2f_F, hw2f_args),
        ('cir', sp.cox_ingersoll_ross_process, cir_F, cir_args),
        ('heston1', full_heston_wrapped, heston_F, heston_args),
        ('heston2', sp.heston_process, heston_F, heston_args),
        ('mjd', sp.merton_jumpdiff_process, mjd_F, mjd_args),
        ('kou', sp.kou_jumpdiff_process, kou_F, kou_args),
        )

    # do the tests
    # ------------
    do(quant_case, cases,
       context=context,
       err_expected=err_expected,
       err_realized=err_realized, PATHS=PATHS)

    # save results
    # ------------
    save_errors(context, err_realized)

    # for key, (meanerr, maxerr) in err_realized.items():
    #     print('{:40s} {:12.5g} {:12.5g}'.format(key, meanerr, maxerr))


# test cases
def quant_case(case, context, err_expected, err_realized, PATHS):
    """
    Actual testing of a single case:
    - realize a given process for given fixed parameters
    - check mean, var, std, pdf, cdf, chf against analytic formulae
    """
    Xid, Xclass, F, args = case
    prefix = context + '_' + Xid

    def check(t, *tests, fig_id='', mode='rel', xlabel='t', brk=None):
        return check_values(context, Xid, t, *tests,
                            fig_id=fig_id,
                            err_expected=err_expected,
                            err_realized=err_realized,
                            mode=mode, xlabel=xlabel, brk=brk, PATHS=PATHS)

    # testing process
    # ---------------
    t0, t1 = 1, 3
    tt = np.linspace(t0, t1, 1000)
    t = np.linspace(t0, t1, 50)
    s = (t0, (t0+t1)/2, t1)

    print(Xid, sep=' ')
    if iskfunc(Xclass):
        X = Xclass(**args)
    else:
        def X(t, **p):
            return Xclass(**args, **p)(t)
    Xmean, Xvar, Xstd = F[:3]
    Xpdf, Xcdf, Xchf = F[3:]
    fargs = args.copy()
    fargs.pop('steps', None)

    # plot paths for visual inspection (no testing)
    if PLOT:
        print('plotting...')
        np.random.seed(SEED)
        fig = bfig()
        plt.title('{}: 30 sample paths'.format(Xid))
        plt.xlabel('t')
        plt.plot(tt, X(tt, paths=30))
        save_figure(fig, prefix + '_paths')
        plt.close(fig)

    # check mean, std and variance
    if Xmean is not None:
        np.random.seed(SEED)
        x = X(t, paths=PATHS)
        if 'heston' in Xid:
            log(x, out=x)
        if Xstd is not None:
            check(t[1:],
                  ('mean', x.pmean()[1:, ..., 0], Xmean(t[1:]-t[0], **fargs)),
                  ('std', x.pstd()[1:, ..., 0], Xstd(t[1:]-t[0], **fargs)),
                  ('var', x.pvar()[1:, ..., 0], Xvar(t[1:]-t[0], **fargs)),
                  fig_id='stats', mode='rel', brk=False)
        else:
            check(t[1:],
                  ('mean', x.pmean()[1:, ..., 0], Xmean(t[1:]-t[0], **fargs)),
                  fig_id='stats', mode='rel', brk=False)
        del x

    # check pdf and cdf
    np.random.seed(SEED)
    a = sp.montecarlo(bins=200)
    for i in range(10):
        x = X(s, paths=PATHS)
        xx = log(x) if 'log_pdf' in Xpdf.__name__ else x
        a.update(xx(t1))
    y = np.linspace(xx(t1).min(), xx(t1).max(), 200)
    if Xcdf is not None:
        check(y,
              ('pdf', a.pdf(y), Xpdf(t1-t0, y, **fargs)),
              ('cdf_1', a.cdf(y), Xcdf(t1-t0, y, **fargs)),
              ('cdf_2', xx.cdf(t1, y), Xcdf(t1-t0, y, **fargs)),
              fig_id='pdf_cdf', mode='wabs', xlabel='x')
    else:
        check(y, ('pdf', a.pdf(y), Xpdf(t1-t0, y, **fargs)),
              fig_id='pdf', mode='wabs', xlabel='x')
    del a, xx, y  # last computed x is used below

    # check chf
    if Xchf is not None:
        u = np.linspace(-5, 5, 100)
        A = (log(x) if 'log_chf' in Xchf.__name__ else x).chf(t1, u)
        B = Xchf(t1-t0, u, **fargs)
        check(u,
              ('chf_re', np.real(A), np.real(B)),
              ('chf_im', np.imag(A), np.imag(B)),
              fig_id='chf', mode='abs', xlabel='u')
        del u, A, B

    del x


# -----------------------------------
# Tests with varying parameter values
# context = 'params'
# -----------------------------------

@slow
@quant
def test_params():
    if QUANT_TEST_MODE == 'HD':
        tst_params(context='params5', PATHS=PATHS_HD)
    elif QUANT_TEST_MODE == 'LD':
        tst_params(context='params2', PATHS=PATHS_LD)
    else:
        raise ValueError()


# main test
def tst_params(context, PATHS):
    """Setup test cases, do actual testing invoking 'params()',
    save results.
    """
    err_expected = load_errors(context)
    err_realized = {}

    # setup test cases
    # ----------------
    K = 20

    SIGMA = np.linspace(0.01, 0.5, K).reshape(K, 1)
    SIGMA2 = np.concatenate(
        (SIGMA.reshape((K, 1, 1)),
         np.full(K, fill_value=0.10).reshape(K, 1, 1)),
        axis=-2)

    KAPPA = np.linspace(0.1, 20, K).reshape(K, 1)
    XI = np.linspace(0.1, 4., K).reshape(K, 1)
    LAM = np.linspace(0.1, 50., K).reshape(K, 1)
    # A = np.linspace(0.01, 0.5, K)
    # PA = np.linspace(0, 1, K)

    wiener_args_sigma = dict(vshape=K, sigma=SIGMA, mu=0.1, x0=log(10))
    lognorm_args_sigma = dict(vshape=K, sigma=SIGMA, mu=0.1, x0=10)
    oruh_args_sigma = dict(vshape=K, sigma=SIGMA, x0=0.20,
                           theta=0.10, k=1, steps=200)
    hw2f_args_sigma = dict(vshape=K, sigma=SIGMA2,
                           x0=((0.30,), (0.05,)),
                           theta=((0.10,), (0.20,)),
                           k=((1.,), (0.3,)), rho=-0.5,
                           steps=200)
    cir_args_k = dict(vshape=K, k=KAPPA, x0=0.30, theta=0.5, xi=.50,
                      steps=200)
    # heston parameters need an extra axis, to avoid
    # conflict with the axis representing the two components
    # of the sde
    heston_args_sigma = dict(vshape=(K, 1), sigma=SIGMA[..., np.newaxis],
                             x0=10., mu=0.1,
                             y0=.5, theta=1.5, k=3., xi=2., rho=-0.5,
                             steps=200)
    heston_args_k = dict(vshape=(K, 1), k=KAPPA[..., np.newaxis], sigma=0.2,
                         x0=10., mu=0.1,
                         y0=.5, theta=1.5, rho=-0.5,
                         steps=200)
    heston_args_xi = dict(vshape=(K, 1), xi=XI[..., np.newaxis], sigma=0.2,
                          x0=10., mu=0.1,
                          y0=.5, theta=1.5, k=3., rho=-0.5,
                          steps=200)
    mjd_args_sigma = dict(vshape=K, sigma=SIGMA, mu=0.1, x0=6, lam=2.,
                          a=-0.2, b=0.1, steps=200)
    mjd_args_lam = dict(vshape=K, lam=LAM, sigma=0.3, mu=0.1, x0=6,
                        a=-0.2, b=0.1, steps=200)
    kou_args_sigma = dict(vshape=K, sigma=SIGMA, mu=0.1, x0=6, lam=2.,
                          pa=.80,  a=0.2, b=0.3, steps=200)
    kou_args_lam = dict(vshape=K, lam=LAM, sigma=0.3, mu=0.1, x0=6,
                        pa=.80,  a=0.2, b=0.3, steps=200)

    wiener_F = (sp.wiener_mean, sp.wiener_var, sp.wiener_std,
                sp.wiener_pdf, sp.wiener_cdf, sp.wiener_chf)
    lognorm_F = (sp.lognorm_mean, sp.lognorm_var, sp.lognorm_std,
                 sp.lognorm_pdf, sp.lognorm_cdf, sp.lognorm_log_chf)
    oruh_F = (sp.oruh_mean, sp.oruh_var, sp.oruh_std,
              sp.oruh_pdf, sp.oruh_cdf, None)
    hw2f_F = (sp.hw2f_mean,
              sp.hw2f_var,
              sp.hw2f_std,
              sp.hw2f_pdf,
              sp.hw2f_cdf, None)
    cir_F = (sp.cir_mean, sp.cir_var, sp.cir_std,
             sp.cir_pdf, None, None)
    heston_F = (sp.heston_log_mean, sp.heston_log_var, sp.heston_log_std,
                sp.heston_log_pdf, None, sp.heston_log_chf)
    mjd_F = (None, None, None,
             sp.mjd_log_pdf, None, sp.mjd_log_chf)
    kou_F = (None, None, None,
             sp.kou_log_pdf, None, sp.kou_log_chf)

    cases_sigma = (
        ('wiener2_sigma', sp.wiener_process, wiener_F,
         {**wiener_args_sigma, **dict(steps=50)}, 'sigma'),
        ('lognorm2_sigma', sp.lognorm_process, lognorm_F,
         {**lognorm_args_sigma, **dict(steps=50)}, 'sigma'),
        ('oruh_sigma', sp.ornstein_uhlenbeck_process, oruh_F,
         oruh_args_sigma, 'sigma'),
        ('hw2f_sigma', hw2f_wrapped, hw2f_F,
         hw2f_args_sigma, 'sigma'),
        ('heston1_sigma', full_heston_wrapped, heston_F,
         heston_args_sigma, 'sigma'),
        ('heston2_sigma', sp.heston_process, heston_F,
         heston_args_sigma, 'sigma'),
        ('mjd_sigma', sp.merton_jumpdiff_process, mjd_F,
         mjd_args_sigma, 'sigma'),
        ('kou_sigma', sp.kou_jumpdiff_process, kou_F,
         kou_args_sigma, 'sigma'),
        )

    cases_other = (
        ('cir_k', sp.cox_ingersoll_ross_process, cir_F,
         cir_args_k, 'k'),
        ('heston1_k', full_heston_wrapped, heston_F,
         heston_args_k, 'k'),
        ('heston1_xi', full_heston_wrapped, heston_F,
         heston_args_xi, 'xi'),
        ('mjd_lam', sp.merton_jumpdiff_process, mjd_F,
         mjd_args_lam, 'lam'),
        ('kou_lam', sp.kou_jumpdiff_process, kou_F,
         kou_args_lam, 'lam'),
        )

    # do the tests
    # ------------
    do(params_case, cases_sigma + cases_other,
       context=context,
       err_expected=err_expected,
       err_realized=err_realized, PATHS=PATHS)

    # save results
    # ------------
    save_errors(context, err_realized)

    # for key, (meanerr, maxerr) in err_realized.items():
    #     print('{:40s} {:12.5g} {:12.5g}'.format(key, meanerr, maxerr))


# test cases
def params_case(case, context, err_expected, err_realized, PATHS):
    """
    Actual testing of a single case:
    - realize a given process for a range of values of a parameter;
    - check mean and std dependency on the given parameter, against
      analytical formulae.
    """
    Xid, Xclass, F, args, param = case

    def check(t, *tests, fig_id='', mode='rel', xlabel='t', brk=None):
        return check_values(context, Xid, t, *tests,
                            fig_id=fig_id,
                            err_expected=err_expected,
                            err_realized=err_realized,
                            mode=mode, xlabel=xlabel, brk=brk, PATHS=PATHS)

    # testing dependence on parameter
    # -------------------------------
    t0, t1 = 1, 3
    s = (t0, (t0+t1)/2, t1)

    print(Xid, sep=' ')
    Xmean, Xvar, Xstd = F[:3]
    Xpdf, Xcdf, Xchf = F[3:]

    fargs = args.copy()
    fargs.pop('steps', None)
    fargs.pop('vshape', None)

    if 'hw2f' in Xid or 'heston' in Xid:
        pvalues = args[param][..., 0, :]
    else:
        pvalues = args[param]

    # process to be tested
    np.random.seed(SEED)
    if iskfunc(Xclass):
        # test kfunc call with parameters
        x = Xclass(**args)(s, paths=PATHS)
    else:
        x = Xclass(**args, paths=PATHS)(s)

    if 'heston' in Xid:
        # handle extra axis in tested parameter
        K = x.size//len(s)//PATHS
        assert_(x.shape == (len(s), K, 1, PATHS))
        x = sp.process(t=x.t, x=x[:, :, 0, :])
        assert_(fargs[param].shape == (K, 1, 1))
        fargs[param] = fargs[param].reshape(K, 1)

    # check mean and std
    x1 = x['t', -1]
    if 'heston' in Xid:
        x1 = log(x1)
    if Xmean is not None:
        check(pvalues,
              ('mean', x1.pmean()[0],
               Xmean(t1 - t0, **fargs)),
              ('std', x1.pstd()[0],
               Xstd(t1 - t0, **fargs)),
              fig_id='stats', mode='rel', xlabel=param)

    # check cdf and chf
    tests = []
    y = x[0, ..., 0].mean()  # y = initial condition of the process
    if Xcdf is not None:
        tests += (('cdf', x.cdf(t1, y),
                   Xcdf(t1-t0, y, **fargs)[..., 0]),)
    # NOTE: x.cdf(t1, y).shape is (K,)
    # xcdf(t1-t0, y, **fargs).shape is (K, 1) since params in
    # fargs have a path axis
    if Xchf is not None:
        if 'log_chf' in Xchf.__name__:
            u = 1  # 1/log(y)
            A = (log(x)).chf(t1, u)
        else:
            u = 1  # 1/y
            A = x.chf(t1, u)
        B = Xchf(t1 - t0, u, **fargs)[..., 0]
        tests += (
            ('chf_re', np.real(A), np.real(B)),
            ('chf_im', np.imag(A), np.imag(B)))
    if tests:
        check(pvalues, *tests, fig_id='cdf_chf',
              mode='abs', xlabel=param)


def test_bs():
    """a bare-bone test on Black-Scholes call and put valuation"""
    np.random.seed(SEED)

    Kc, Kp = 1.2, 0.6
    T = 2.
    r = 0.05
    q = 0.10
    sigma = 0.30
    args = dict(r=r, q=q, sigma=sigma)

    X = sp.lognorm_process(mu=r - q, sigma=sigma,
                           paths=100*1000)((0, T))

    # test put and call values
    p1 = sp.bscall(Kc, T, **args)
    p2 = sp.montecarlo(np.maximum(X[-1] - Kc, 0)*exp(-r*T))
    assert_(abs(p1-p2.m)/p2.e < 3)

    p1 = sp.bsput(Kp, T, **args)
    p2 = sp.montecarlo(np.maximum(Kp - X[-1], 0)*exp(-r*T))
    assert_(abs(p1-p2.m)/p2.e < 3)

    eps = 1e-4

    # test bscall_delta and bsput_delta formulae
    argsplus = {**args, **dict(x0=1 + eps)}
    argsminus = {**args, **dict(x0=1 - eps)}
    cd1 = sp.bscall_delta(Kc, T, **args),
    cd2 = (sp.bscall(Kc, T, **argsplus) -
           sp.bscall(Kc, T, **argsminus))/(2*eps)
    assert_((cd2 - cd1)/cd1 < 3*eps*eps)

    pd1 = sp.bsput_delta(Kp, T, **args),
    pd2 = (sp.bsput(Kp, T, **argsplus) -
           sp.bsput(Kp, T, **argsminus))/(2*eps)
    assert_((pd2 - pd1)/pd1 < 3*eps*eps)
