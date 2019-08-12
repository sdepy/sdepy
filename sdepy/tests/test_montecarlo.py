"""
===================================================================
FORMAL (FAST) AND QUANTITATIVE (SLOW) TESTS ON THE MONTECARLO CLASS
===================================================================
"""
from .shared import *

import scipy
import scipy.stats
import scipy.interpolate

montecarlo = sp.montecarlo


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


# -------------
# test workflow
# -------------

# main test
def test_montecarlo_workflow():
    np.random.seed(SEED)

    # do cases
    shape = [(), (2,), (3, 2)]
    paths = [1, 10]
    dtype = [None, float, np.float64, np.float32,
             int, np.int64, np.int32]
    do(montecarlo_workflow, shape, paths, dtype)


# cases
def montecarlo_workflow(shape, paths, dtype):
    # print(shape, paths, dtype)
    PATHS = paths
    size = np.empty(shape).size
    indexes = [i for i in np.ndindex(shape)]

    # cumulate 10 simulations
    a = montecarlo(bins='auto')
    assert_raises(ValueError, a.histogram)
    for i in range(10):
        sample = np.random.normal(size=shape + (PATHS,)).astype(dtype)
        sample *= (1+np.arange(size)).reshape(shape + (1,))
        a.update(sample)
    assert_(a.paths == PATHS*10)

    # access results
    str(a)
    x = np.linspace(-3*size, 3*size, 100)
    for k in range(5):
        i = choice(indexes)
        a[i].pdf(x)  # default
        a[i].pdf(x, bandwidth=.7)  # gaussian kde with explicit bandwidth
        a[i].pdf(x, kind='linear', method='interp')
        a[i].pdf(x, kind='quadratic', method='interp')
        bins, counts = a[i].histogram()
        bins, counts = a[i].density_histogram()
        a[i].h, a[i].dh  # access using shortcuts
        a[i].outpaths, a[i].outerr()

    # access attributes
    assert_(a.paths == 10*PATHS)
    assert_(a.vshape == shape)
    assert_(a.shape == shape + (10*PATHS,))
    assert_array_equal(a.m, a.mean())
    assert_array_equal(a.m, a.stats['mean'])
    assert_array_equal(a.e, a.stderr())
    assert_array_equal(a.e, a.stats['stderr'])
    assert_allclose(a.e, a.stats['std']/np.sqrt(a.paths-1),
                    rtol=eps(a.m.dtype))

    # check dtype
    if len(shape) >= 1:
        type_ = np.dtype(dtype).type
        if type_(0.1) != type_(0):
            assert_(a.m.dtype == dtype)
            assert_(a.e.dtype == dtype)
        else:
            assert_(a.m.dtype == float)
            assert_(a.e.dtype == float)
    assert_(a._counts[i].dtype == a.ctype)

    # no histograms
    b = montecarlo(sample, bins=None)  # fist run
    b.stats['skew'], b.stats['kurtosis']
    assert_(b._bins is None)
    b.update(sample)  # cumulation run

    # explicit bins specification in native format
    c = montecarlo(bins=a._bins)
    c.update(sample)

    # explicit bins specification as array
    bins = np.empty(shape=shape + (10,))
    for k, i in enumerate(np.ndindex(shape), 1):
        bins[i] = np.linspace(-3*k, 3*k, 10)
    d = montecarlo(bins=bins)
    d.update(sample)

    # use paths along different axes
    e = None
    for k in range(len(shape)):
        axis_sample = np.random.normal(
            size=shape[:k] + (PATHS,) + shape[k:]).astype(dtype)
        # first run
        e = montecarlo(axis_sample, axis=k, bins=20, range=(-3, 3))
        # cumulation run
        e.update(axis_sample, axis=k)

    # non-array samples
    f = montecarlo(5)
    f.update(10)
    assert_(f.m == 7.5)
    f = montecarlo(((1, 2, 3), (2, 4, 6)), axis=1)
    f.update(((4, 5, 6), (8, 10, 12)), axis=1)

    assert_allclose(f.mean(), np.array((3.5, 7.)),
                    rtol=eps(f.mean().dtype))

    # antithetic sampling
    sample = np.random.normal(size=(2*PATHS,) + shape)
    even = montecarlo(sample, axis=0, use='even')
    odd = montecarlo(sample, axis=0, use='odd')
    for a in (even, odd):
        assert_(a.std().shape == shape)
        assert_(a.paths == PATHS)
    assert_allclose(even.mean(), montecarlo(sample, 0).mean())
    assert_allclose(odd.mean(), (sample[:PATHS]-sample[PATHS:]).mean(axis=0)/2)
    sample2 = np.random.normal(size=(4*PATHS,) + shape)
    even.update(sample2, axis=0)
    odd.update(sample2, axis=0)
    for a in (even, odd):
        assert_(a.mean().shape == shape)
        assert_(a.std().shape == shape)
        assert_(a.paths == 3*PATHS)
    even = montecarlo(use='even')
    odd = montecarlo(use='odd')
    even.update(sample, axis=0)
    odd.update(sample, axis=0)
    for a in (even, odd):
        assert_(a.mean().shape == shape)
        assert_(a.std().shape == shape)
        assert_(a.paths == PATHS)

    assert_raises(ValueError, montecarlo, sample, use=None)
    assert_raises(ValueError, montecarlo, np.zeros((3, 2)), axis=0,
                  use='even')


# -------------
# test cumulate
# -------------

# main test
def test_montecarlo_cumulate():
    np.random.seed(SEED)

    shape = [(), (2,), (3, 2)]
    paths = [1, 10]
    dtype = [None, float, np.float64, np.float32, np.float16,
             int, np.int64, np.int32, np.int16]
    do(montecarlo_cumulate, shape, paths, dtype)


# cases
def montecarlo_cumulate(shape, paths, dtype):
    PATHS = paths
    sample = np.random.normal(size=shape + (4*PATHS,)).astype(dtype)

    # full sample as first run
    a = montecarlo(sample)

    # same sample loaded in 3 runs
    b = montecarlo(sample[..., :PATHS])
    b.update(sample[..., PATHS:2*PATHS])
    b.update(sample[..., 2*PATHS:])

    rtol = max(eps(sample.dtype), eps(a.mean().dtype))
    assert_allclose(a.mean(), b.mean(), rtol=2*rtol)
    assert_allclose(a.stderr(), b.stderr(), rtol=rtol)
    assert_allclose(a.mean(), sample.mean(axis=-1),
                    rtol=rtol)
    assert_allclose(a.stderr(), sample.std(axis=-1)/np.sqrt(4*PATHS-1),
                    rtol=rtol)

    # same sample loaded in 3 runs, using same bins
    c = montecarlo()
    c._bins = a._bins
    c.update(sample[..., :PATHS])
    c.update(sample[..., PATHS:2*PATHS])
    c.update(sample[..., 2*PATHS:])
    for i in np.ndindex(c.vshape):
        assert_array_equal(a._counts[i], c._counts[i])
        assert_array_equal(a._bins[i], c._bins[i])
    assert_array_equal(a._paths_outside, c._paths_outside)


# -----------------------------
# test histogram (quantitative)
# context = 'histogram'
# -----------------------------

# main test

@slow
@quant
def test_montecarlo_histogram():
    np.random.seed(SEED)

    context = 'histogram'
    err_expected = load_errors(context)
    err_realized = {}

    BINS = 50
    PATHS = 1000*1000
    RATIO = 20  # ratio of last to first bin in nonlinear bins
    x = np.linspace(0, 1, BINS + 1)
    y = np.full(BINS + 1, RATIO**(1/BINS)).cumprod()
    y = (y - y.min())/(y.max() - y.min())
    assert_allclose((y.min(), y.max()), (0, 1))
    linear_bins = x
    nonlinear_bins = y

    #  dist_info: a dict of specs for testing montecarlo histograms,
    #  with the following structure:
    #    {test_id: (distribution instance, support, bins, paths)}
    dist_info = {
        'unif_lin': (scipy.stats.uniform(0, 1), (0, 1),
                     1.2*linear_bins - 0.1, PATHS),
        'unif_nonlin': (scipy.stats.uniform(0, 1), (0, 1),
                        1.2*nonlinear_bins - 0.1, PATHS),
        'norm_lin': (scipy.stats.norm(0, 1), (-3, 3),
                     8*linear_bins - 4, PATHS),
        'norm_nonlin': (scipy.stats.norm(0, 1), (-3, 3),
                        8*nonlinear_bins - 4, PATHS),
        'trapz_lin': (scipy.stats.trapz(0.2, 0.6), (0, 1),
                      1.2*linear_bins - 0.1, PATHS),
        'trapz_nonlin': (scipy.stats.trapz(0.2, 0.6), (0, 1),
                         1.2 * nonlinear_bins - 0.1, PATHS)
    }
    dist_keys = (
        'unif_lin', 'unif_nonlin', 'norm_lin', 'norm_nonlin',
        'trapz_lin', 'trapz_nonlin')  # make order predictable

    # fast test with several shapes and few paths, no error checking
    # and no plotting
    do(montecarlo_histogram, (context,), dist_keys,
       (noerrors_expected,), (noerrors_realized,),
       ((), (3,), (3, 5)),
       (dist_info,), (7,), (False,))

    # slow test with error checking and plotting
    do(montecarlo_histogram, (context,), dist_keys,
       (err_expected,), (err_realized,), ((3, 2),),
       (dist_info,), (PATHS,), (PLOT,))
    save_errors(context, err_realized)


# cases
def montecarlo_histogram(context, test_id, err_expected, err_realized,
                         shape, dist_info, paths, PLOT):
    dist, support, bins, _ = dist_info[test_id]
    test_key = context + '_' + test_id

    sample = dist.rvs(size=shape + (paths,))
    sample_bins = np.empty(shape + bins.shape)
    for i in np.ndindex(shape):
        sample_bins[i] = bins

    a = montecarlo(bins=sample_bins)
    p = paths//5
    a.update(sample[..., :p])
    a.update(sample[..., p:2*p])
    a.update(sample[..., 2*p:])

    mean_err = np.zeros(a.vshape)
    max_err = np.zeros(a.vshape)
    expected_counts = a.paths*np.diff(dist.cdf(bins))
    mean_counts = expected_counts[expected_counts > 0].mean()
    for i in np.ndindex(shape):
        counts, bins = a[i].histogram()
        delta = np.abs(counts - expected_counts)/mean_counts
        mean_err[i] = delta.mean()
        max_err[i] = delta.max()
        assert_quant(mean_err[i] < err_expected[test_key][0])
        assert_quant(max_err[i] < err_expected[test_key][1])

        ncounts, bins = a[i].density_histogram()
        assert_allclose((np.diff(bins)*ncounts).sum(), 1.,
                        rtol=eps(a.mean().dtype))

    err_realized[test_key] = (mean_err.max(), max_err.max())

    # plot worst case
    if PLOT:
        print('plotting...')
        i = (max_err == max_err.max())
        i = tuple(np.where(i)[k][0] for k in range(i.ndim))
        counts, bins = a[i].histogram()
        fig = plt.figure()
        plt.title(test_key + ':\nmean err = {:.5g}, max err = {:.5g}'
                  .format(float(mean_err[i]), float(max_err[i])) +
                  '\n error is conunts diff/mean counts')
        plot_histogram((expected_counts, bins), color='g',
                       histtype='step', label='expected', lw=2)
        plot_histogram((counts, bins), color='b',
                       histtype='step', label='realized')
        plt.plot(bins, bins*0 + mean_counts, 'c:', label='mean counts')
        plt.legend()
        plt.savefig(os.path.join(DIR, test_key + '.png'), dpi=300)
        plt.close(fig)


# -------------------------------
# test pdf and cdf (quantitative)
# context = 'distr'
# -------------------------------

# main test
@slow
@quant
def test_montecarlo_dist():
    np.random.seed(SEED)

    context = 'distr'
    err_expected = load_errors(context)
    err_realized = {}

    BINS = 50
    PATHS = 1000*1000
    RATIO = 10  # ratio of last to first bin in nonlinear bins
    x = np.linspace(0, 1, BINS + 1)
    y = np.full(BINS + 1, RATIO**(1/BINS)).cumprod()
    y = (y - y.min())/(y.max() - y.min())
    assert_allclose((y.min(), y.max()), (0, 1))
    linear_bins = x
    nonlinear_bins = y

    #  dist_info: a dict of specs for testing montecarlo histograms,
    #  with the following structure:
    #    {test_id: (distribution instance, support, bins, paths,
    #     integral_bounds)}
    dist_info = {
        'unif_lin': (scipy.stats.uniform(0, 1), (0, 1),
                     1.2*linear_bins - 0.1, PATHS, (-1, 2)),
        'unif_nonlin': (scipy.stats.uniform(0, 1), (0, 1),
                        1.2*nonlinear_bins - 0.1, PATHS, (-1, 2)),
        'norm_lin': (scipy.stats.norm(0, 1), (-3, 3),
                     8*linear_bins - 4, PATHS, (-6, 6)),
        'norm_nonlin': (scipy.stats.norm(0, 1), (-3, 3),
                        8*nonlinear_bins - 4, PATHS, (-6, 6)),
        'trapz_lin': (scipy.stats.trapz(0.2, 0.6), (0, 1),
                      1.2*linear_bins - 0.1, PATHS, (-1, 2)),
        'trapz_nonlin': (scipy.stats.trapz(0.2, 0.6), (0, 1),
                         1.2 * nonlinear_bins - 0.1, PATHS, (-1, 2))
    }
    dist_keys = (
        'unif_lin', 'unif_nonlin', 'norm_lin', 'norm_nonlin',
        'trapz_lin', 'trapz_nonlin')  # make order predictable

    # fast test with several shapes and few paths, no error checking
    # and no plotting
    do(montecarlo_dist, (context,), dist_keys,
       (noerrors_expected,), (noerrors_realized,),
       ((), (3,), (3, 5)),
       (dist_info,), (7,), (False,))

    # slow test with error checking and plotting
    do(montecarlo_dist, (context,), dist_keys,
       (err_expected,), (err_realized,), ((3, 2),),
       (dist_info,), (PATHS,), (PLOT,))
    save_errors(context, err_realized)


# cases
def montecarlo_dist(context, test_id, err_expected, err_realized,
                    shape, dist_info, paths, PLOT):
    dist, support, bins, _, integral_bounds = dist_info[test_id]
    test_key = context + '_' + test_id

    # prepare sample
    sample = dist.rvs(size=shape + (paths,))
    sample_bins = np.empty(shape + bins.shape)
    for i in np.ndindex(shape):
        sample_bins[i] = bins

    a = montecarlo(bins=sample_bins)
    p = paths//5
    a.update(sample[..., :p])
    a.update(sample[..., p:2*p])
    a.update(sample[..., 2*p:])

    # test pdf and cdf values
    (mean_err_pdf1, max_err_pdf1, mean_err_cdf1, max_err_cdf1,
     mean_err_pdf2, max_err_pdf2, mean_err_cdf2, max_err_cdf2) = \
        [np.zeros(a.vshape) for i in range(8)]
    x = np.linspace(bins[0], bins[-1], 10*bins.size)
    true_pdf = dist.pdf(x)
    true_cdf = dist.cdf(x)
    mean_pdf = 0.95/(4*dist.std())
    mean_cdf = 0.5
    for i in np.ndindex(shape):

        pdf1 = a[i].pdf(x, method='gaussian_kde')
        pdf2 = a[i].pdf(x, method='interp')
        delta1 = np.abs(pdf1 - true_pdf)/mean_pdf
        delta2 = np.abs(pdf2 - true_pdf)/mean_pdf
        mean_err_pdf1[i] = delta1.mean()
        mean_err_pdf2[i] = delta2.mean()
        max_err_pdf1[i] = delta1.max()
        max_err_pdf2[i] = delta2.max()
        assert_quant(mean_err_pdf1[i] <
                     err_expected[test_key + '_pdf_kde'][0])
        assert_quant(max_err_pdf1[i] <
                     err_expected[test_key + '_pdf_kde'][1])
        assert_quant(mean_err_pdf2[i] <
                     err_expected[test_key + '_pdf_interp'][0])
        assert_quant(max_err_pdf2[i] <
                     err_expected[test_key + '_pdf_interp'][1])

        cdf1 = a[i].cdf(x, method='gaussian_kde')
        cdf2 = a[i].cdf(x, method='interp')
        delta1 = np.abs(cdf1 - true_cdf)/mean_cdf
        delta2 = np.abs(cdf2 - true_cdf)/mean_cdf
        mean_err_cdf1[i] = delta1.mean()
        mean_err_cdf2[i] = delta2.mean()
        max_err_cdf1[i] = delta1.max()
        max_err_cdf2[i] = delta2.max()
        assert_quant(mean_err_cdf1[i] <
                     err_expected[test_key + '_cdf_kde'][0])
        assert_quant(max_err_cdf1[i] <
                     err_expected[test_key + '_cdf_kde'][1])
        assert_quant(mean_err_cdf2[i] <
                     err_expected[test_key + '_cdf_interp'][0])
        assert_quant(max_err_cdf2[i] <
                     err_expected[test_key + '_cdf_interp'][1])

    # store realized errors
    err_realized[test_key + '_pdf_kde'] = (mean_err_pdf1.max(),
                                           max_err_pdf1.max())
    err_realized[test_key + '_pdf_interp'] = (mean_err_pdf2.max(),
                                              max_err_pdf2.max())
    err_realized[test_key + '_cdf_kde'] = (mean_err_cdf1.max(),
                                           max_err_cdf1.max())
    err_realized[test_key + '_cdf_interp'] = (mean_err_cdf2.max(),
                                              max_err_cdf2.max())

    # test pdf normalization
    for i in np.ndindex(shape):
        y = np.linspace(integral_bounds[0], integral_bounds[1], 100*bins.size)
        counts, bins = a[i].histogram()
        integral1 = scipy.integrate.trapz(
            a[i].pdf(y, method='gaussian_kde'), y)
        integral2 = scipy.integrate.trapz(
            a[i].pdf(y, method='interp'), y)
        err1 = np.abs(integral1 - 1)
        err2 = np.abs(integral2 - 1)
        assert_quant(err1 < err_expected[test_key + '_integral_kde'][0])
        assert_quant(err2 < err_expected[test_key + '_integral_lin'][0])
        err_realized[test_key + '_integral_kde'] = (err1, err1)
        err_realized[test_key + '_integral_lin'] = (err2, err2)
        break  # test only first for speed

    # plot worst case
    if PLOT:
        print('plotting...')

        # plot pdf
        i = (max_err_pdf1 == max_err_pdf1.max())
        i = tuple(np.where(i)[k][0] for k in range(i.ndim))
        pdf1 = a[i].pdf(x, method='gaussian_kde')
        pdf2 = a[i].pdf(x, method='interp')
        counts, bins = a[i].density_histogram()
        fig = plt.figure()
        plt.title(test_key + '_pdf:\nmean err = {:.5g}, max err = {:.5g}'
                  .format(float(mean_err_pdf1[i]), float(max_err_pdf1[i])) +
                  '\n error is conunts diff/mean counts')
        plot_histogram((counts, bins), color='y',
                       histtype='step', label='realized histogram')
        plt.plot(x, true_pdf, color='g', label='expected pdf', lw=2)
        plt.plot(x, pdf1, color='b', label='realized pdf - gaussian kde')
        plt.plot(x, pdf2, color='r', label='realized pdf - interpolate')
        plt.plot(x, x*0 + mean_pdf, 'c:', label='mean pdf')
        plt.legend()
        plt.savefig(os.path.join(DIR, test_key + '_pdf.png'), dpi=300)
        plt.close(fig)

        # plot cdf
        i = (max_err_cdf1 == max_err_cdf1.max())
        i = tuple(np.where(i)[k][0] for k in range(i.ndim))
        cdf1 = a[i].cdf(x, method='gaussian_kde')
        cdf2 = a[i].cdf(x, method='interp')
        counts, bins = a[i].density_histogram()
        fig = plt.figure()
        plt.title(test_key + '_cdf:\nmean err = {:.5g}, max err = {:.5g}'
                  .format(float(mean_err_cdf1[i]), float(max_err_cdf1[i])) +
                  '\n error is conunts diff/mean counts')
        plot_histogram(
            hist=(
                (counts*np.diff(bins)).cumsum(),
                np.concatenate(((bins[:-1] + bins[1:])/2, bins[-1:]))
                ),
            color='y', histtype='step', label='realized histogram')
        plt.plot(x, true_cdf, color='g', label='expected cdf', lw=2)
        plt.plot(x, cdf1, color='b', label='realized cdf - gaussian kde')
        plt.plot(x, cdf2, color='r', label='realized cdf - interpolate')
        plt.legend()
        plt.savefig(os.path.join(DIR, test_key + '_cdf.png'), dpi=300)
        plt.close(fig)
