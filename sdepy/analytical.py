"""
=========================================
ANALYTICAL RESULTS FOR REALIZED PROCESSES
=========================================
"""
import numpy as np
import scipy
from numpy import sqrt, exp, log


###################################
# Statistics for realized processes
###################################

# when used for multi-dimensional processes,
# these analytical results are valid only if there is
# no correlation among the process components.

# --------------------
# wiener_process stats
# --------------------

# @kfunc(nvar=1)
def wiener_mean(t, *, x0=0., mu=0., sigma=1.):
    """
    wiener_mean(t, *, x0=0., mu=0., sigma=1.)

    Mean of values at time t of a Wiener process
    (as per the wiener_process class) with
    time-independent parameters.

    See Also
    --------
    wiener_process
    """
    t, x0, mu, sigma = np.broadcast_arrays(t, x0, mu, sigma)
    return x0 + mu*t


# @kfunc(nvar=1)
def wiener_var(t, *, x0=0., mu=0., sigma=1.):
    """
    wiener_var(t, *, x0=0., mu=0., sigma=1.)

    Variance of values at time t of a Wiener process
    (as per the wiener_process class) with
    time-independent parameters.

    See Also
    --------
    wiener_process
    """
    t, x0, mu, sigma = np.broadcast_arrays(t, x0, mu, sigma)
    return sigma*sigma*t


# @kfunc(nvar=1)
def wiener_std(t, *, x0=0., mu=0., sigma=1.):
    """
    wiener_std(t, *, x0=0., mu=0., sigma=1.)

    Standard deviation of values at time t of a Wiener process
    (as per the wiener_process class) with
    time-independent parameters.

    See Also
    --------
    wiener_process
    """
    return sqrt(wiener_var(t, x0=x0, mu=mu, sigma=sigma))


# @kfunc(nvar=2)
def wiener_pdf(t, x, *, x0=0., mu=0., sigma=1.):
    """
    wiener_pdf(t, x, *, x0=0., mu=0., sigma=1.)

    Probability distribution function of values
    at time t of a Wiener process (as per the wiener_process class)
    with time-independent parameters, evaluated at x.

    See Also
    --------
    wiener_process
    """
    t, x, x0, mu, sigma = np.broadcast_arrays(t, x, x0, mu, sigma)
    mean = x0 + mu*t
    std = sigma*sqrt(t)
    return scipy.stats.norm.pdf(x, loc=mean, scale=std)


# @kfunc(nvar=2)
def wiener_cdf(t, x, *, x0=0., mu=0., sigma=1.):
    """
    wiener_cdf(t, x, *, x0=0., mu=0., sigma=1.)

    Cumulative probability distribution function of values
    at time t of a Wiener process (as per the wiener_process class)
    with time-independent parameters, evaluated at x.

    See Also
    --------
    wiener_process
    """
    t, x, x0, mu, sigma = np.broadcast_arrays(t, x, x0, mu, sigma)
    mean = x0 + mu*t
    std = sigma*sqrt(t)
    return scipy.stats.norm.cdf(x, loc=mean, scale=std)


# @kfunc(nvar=2)
def wiener_chf(t, u, *, x0=0., mu=0., sigma=1.):
    """
    wiener_chf(t, u, *, x0=0., mu=0., sigma=1.)

    Characteristic function of the probability distribution of values
    at time t of a Wiener process (as per the wiener_process class)
    with time-independent parameters, evaluated at u.

    See Also
    --------
    wiener_process
    """
    t, u, x0, mu, sigma = np.broadcast_arrays(t, u, x0, mu, sigma)
    return exp(1j*u*x0 + t*(1j*mu*u - sigma*sigma*u*u/2))


# ---------------------
# lognorm_process stats
# ---------------------

# @kfunc(nvar=1)
def lognorm_mean(t, *, x0=1., mu=0., sigma=1.):
    """
    lognorm_mean(t, *, x0=1., mu=0., sigma=1.)

    Mean of values at time t of a lognormal process
    (as per the lognorm_process class) with time-independent parameters.

    See Also
    --------
    lognorm_process
    """
    t, x0, mu, sigma = np.broadcast_arrays(t, x0, mu, sigma)
    return x0 * exp(mu*t)


# @kfunc(nvar=1)
def lognorm_var(t, *, x0=1., mu=0., sigma=1.):
    """
    lognorm_var(t, *, x0=1., mu=0., sigma=1.)

    Variance of values at time t of a lognormal process
    (as per the lognorm_process class) with time-independent parameters.

    See Also
    --------
    lognorm_process
    """
    t, x0, mu, sigma = np.broadcast_arrays(t, x0, mu, sigma)
    return x0*x0 * (exp(sigma*sigma*t) - 1) * exp(2*mu*t)


# @kfunc(nvar=1)
def lognorm_std(t, *, x0=1., mu=0., sigma=1.):
    """
    lognorm_std(t, *, x0=1., mu=0., sigma=1.)

    Standard deviation of values at time t of a lognormal process
    (as per the lognorm_process class) with time-independent parameters.

    See Also
    --------
    lognorm_process
    """
    return sqrt(lognorm_var(t, x0=x0, mu=mu, sigma=sigma))


# @kfunc(nvar=2)
def lognorm_pdf(t, x, *, x0=1., mu=0., sigma=1.):
    """
    lognorm_pdf(t, x, *, x0=1., mu=0., sigma=1.)

    Probability distribution function of values at time t of a
    lognormal process (as per the lognorm_process class)
    with time-independent parameters, evaluated at x.

    See Also
    --------
    lognorm_process
    """
    t, x, x0, mu, sigma = np.broadcast_arrays(t, x, x0, mu, sigma)
    mt, st = mu*t, sigma*sqrt(t)
    return scipy.stats.lognorm.pdf(x, s=st,
                                   scale=x0*exp(mt - st*st/2))


# @kfunc(nvar=2)
def lognorm_cdf(t, x, *, x0=1., mu=0., sigma=1.):
    """
    lognorm_cdf(t, x, *, x0=1., mu=0., sigma=1.)

    Cumulative probability distribution function of values
    at time t of a lognormal process (as per the lognorm_process class)
    with time-independent parameters, evaluated at x.

    See Also
    --------
    lognorm_process
    """
    t, x, x0, mu, sigma = np.broadcast_arrays(t, x, x0, mu, sigma)
    mt, st = mu*t, sigma*sqrt(t)
    return scipy.stats.lognorm.cdf(x, s=st,
                                   scale=x0*exp(mt - st*st/2))


# @kfunc(nvar=2)
def lognorm_log_chf(t, u, *, x0=1., mu=0., sigma=1.):
    """
    lognorm_log_chf(t, u, *, x0=1., mu=0., sigma=1.)

    Characteristic function of the probability distribution of values
    at time t of the logarithm of a lognormal process (as per the
    lognorm_process class) with time-independent parameters, evaluated at u.

    See Also
    --------
    lognorm_process
    """
    t, u, x0, mu, sigma = np.broadcast_arrays(t, u, x0, mu, sigma)
    mu0 = mu - sigma*sigma/2
    return exp(1j*u*log(x0) + t*(1j*mu0*u - sigma*sigma*u*u/2))


# --------------------------------
# ornstein_uhlenbeck_process stats
# --------------------------------

# @kfunc(nvar=1)
def oruh_mean(t, *, x0=0., theta=0., k=1., sigma=1.):
    """
    oruh_mean(t, *, x0=0., theta=0., k=1., sigma=1.)

    Mean of values at time t of an Ornstein-Uhlenbeck process
    (as per the ornstein_uhlenbeck_process class) with
    time-independent parameters.

    See Also
    --------
    ornstein_uhlenbeck_process
    """
    t, x0, theta, k, sigma = np.broadcast_arrays(t, x0, theta, k, sigma)
    return theta - exp(-k*t) * (theta - x0)


# @kfunc(nvar=1)
def oruh_var(t, *, x0=0., theta=0., k=1., sigma=1.):
    """
    oruh_var(t, *, x0=0., theta=0., k=1., sigma=1.)

    Variance of values at time t of an Ornstein-Uhlenbeck process
    (as per the ornstein_uhlenbeck_process class) with
    time-independent parameters.

    See Also
    --------
    ornstein_uhlenbeck_process
    """
    t, x0, theta, k, sigma = np.broadcast_arrays(t, x0, theta, k, sigma)
    return (sigma*sigma)/(2*k)*(1 - exp(-2*k*t))


# @kfunc(nvar=1)
def oruh_std(t, *, x0=0., theta=0., k=1., sigma=1.):
    """
    oruh_std(t, *, x0=0., theta=0., k=1., sigma=1.)

    Standard deviation of values at time t of an Ornstein-Uhlenbeck
    process (as per the ornstein_uhlenbeck_process class)
    with time-independent parameters.

    See Also
    --------
    ornstein_uhlenbeck_process
    """
    return sqrt(oruh_var(t, x0=x0, theta=theta, k=k, sigma=sigma))


# @kfunc(nvar=2)
def oruh_pdf(t, x, *, x0=0., theta=0., k=1., sigma=1.):
    """
    oruh_pdf(t, x, *, x0=0., theta=0., k=1., sigma=1.)

    Probability distribution function of values at time t of an
    Ornstein-Uhlenbeck process (as per the ornstein_uhlenbeck_process class)
    with time-independent parameters, evaluated at x.

    See Also
    --------
    ornstein_uhlenbeck_process
    """
    mean = oruh_mean(t, x0=x0, theta=theta, k=k, sigma=sigma)
    var = oruh_var(t, x0=x0, theta=theta, k=k, sigma=sigma)
    return scipy.stats.norm.pdf(x, loc=mean, scale=sqrt(var))


# @kfunc(nvar=2)
def oruh_cdf(t, x, *, x0=0., theta=0., k=1., sigma=1.):
    """
    oruh_cdf(t, x, *, x0=0., theta=0., k=1., sigma=1.)

    Cumulative probability distribution function of values at time t of an
    Ornstein-Uhlenbeck process (as per the ornstein_uhlenbeck_process class)
    with time-independent parameters, evaluated at x.

    See Also
    --------
    ornstein_uhlenbeck_process
    """
    mean = oruh_mean(t, x0=x0, theta=theta, k=k, sigma=sigma)
    var = oruh_var(t, x0=x0, theta=theta, k=k, sigma=sigma)
    return scipy.stats.norm.cdf(x, loc=mean, scale=sqrt(var))


# -----------------------------------
# hull_white_process stats, factors=1
# -----------------------------------

hw1f_mean = oruh_mean
hw1f_var = oruh_var
hw1f_std = oruh_std
hw1f_pdf = oruh_pdf
hw1f_cdf = oruh_cdf


# -----------------------------------
# hull_white_process stats, factors=2
# -----------------------------------

def _hw2f_args_setup(x0, theta, k, sigma, *var):

    x0, theta, k, sigma = np.broadcast_arrays(x0, theta, k, sigma)
    shape = x0.shape
    if len(shape) == 0:
        check = True
        x1, x2 = x0, x0
        theta1, theta2 = theta, theta
        k1, k2 = k, k
        s1, s2 = sigma, sigma
    elif len(shape) == 1:
        check = (shape[-1] in (1, 2))
        x1, x2 = x0[0], x0[-1]
        theta1, theta2 = theta[0], theta[-1]
        k1, k2 = k[0], k[-1]
        s1, s2 = sigma[0], sigma[-1]
    else:
        check = (shape[-2] in (1, 2))
        x1, x2 = x0[..., 0, :], x0[..., -1, :]
        theta1, theta2 = theta[..., 0, :], theta[..., -1, :]
        k1, k2 = k[..., 0, :], k[..., -1, :]
        s1, s2 = sigma[..., 0, :], sigma[..., -1, :]

    if not check:
        raise ValueError(
            'for a 2 factor Hull-White process, '
            'each of x0, theta, k, sigma should be scalar or '
            'a 2-component vector (representing factors) or '
            'have axis -2 of size 1 or 2 (representing factors) '
            'and axis -1 of size 1 (matching the paths axis of '
            'hull_white_process output)'
        )

    return np.broadcast_arrays(x1, x2, theta1, theta2, k1, k2, s1, s2, *var)


# @kfunc(nvar=1)
def hw2f_mean(t, *, x0=(0., 0.), theta=(0., 0.), k=(1., 1.),
              sigma=(1., 1.), rho=0.):
    """
    hw2f_mean(t, *, x0=(0., 0.), theta=(0., 0.), k=(1., 1.),
              sigma=(1., 1.), rho=0.)

    Mean of values at time t of a Hull-White 2-factors process
    (as per the hull_white_process class) with time-independent parameters.

    See Also
    --------
    hull_white_process
    """
    x1, x2, theta1, theta2, k1, k2, s1, s2, rho, t = \
        _hw2f_args_setup(x0, theta, k, sigma, rho, t)

    return (theta1 - exp(-k1*t) * (theta1 - x1) +
            theta2 - exp(-k2*t) * (theta2 - x2))


# @kfunc(nvar=1)
def hw2f_var(t, *, x0=(0., 0.), theta=(0., 0.), k=(1., 1.),
             sigma=(1., 1.), rho=0.):
    """
    hw2f_var(t, *, x0=(0., 0.), theta=(0., 0.), k=(1., 1.),
             sigma=(1., 1.), rho=0.)

    Variance of values at time t of a Hull-White 2-factors process
    (as per the hull_white_process class) with time-independent parameters.

    See Also
    --------
    hull_white_process
    """
    x1, x2, theta1, theta2, k1, k2, s1, s2, rho, t = \
        _hw2f_args_setup(x0, theta, k, sigma, rho, t)

    return (s1*s1)/(2*k1)*(1 - exp(-2*k1*t)) + \
           (s2*s2)/(2*k2)*(1 - exp(-2*k2*t)) + \
           (2*s1*s2*rho)/(k1 + k2)*(1 - exp(-(k1 + k2)*t))


# @kfunc(nvar=1)
def hw2f_std(t, *, x0=(0., 0.), theta=(0., 0.), k=(1., 1.),
             sigma=(1., 1.), rho=0.):
    """
    hw2f_std(t, *, x0=(0., 0.), theta=(0., 0.), k=(1., 1.),
             sigma=(1., 1.), rho=0.)

    Standard deviation of values at time t of a Hull-White 2-factors
    process (as per the hull_white_process class)
    with time-independent parameters.

    See Also
    --------
    hull_white_process
    """
    return sqrt(hw2f_var(t, x0=x0, theta=theta, k=k,
                         sigma=sigma, rho=rho))


# @kfunc(nvar=2)
def hw2f_pdf(t, x, *, x0=(0., 0.), theta=(0., 0.), k=(1., 1.),
             sigma=(1., 1.), rho=0.):
    """
    hw2f_pdf(t, x, *, x0=(0., 0.), theta=(0., 0.), k=(1., 1.),
             sigma=(1., 1.), rho=0.)

    Probability distribution function of values at time t of a
    Hull-White 2-factors process (as per the hull_white_process class)
    with time-independent parameters, evaluated at x.

    See Also
    --------
    hull_white_process
    """
    x1, x2, theta1, theta2, k1, k2, s1, s2, rho, t, x = \
        _hw2f_args_setup(x0, theta, k, sigma, rho, t, x)
    mean = hw2f_mean(t, x0=x0, theta=theta, k=k, sigma=sigma, rho=rho)
    var = hw2f_var(t, x0=x0, theta=theta, k=k, sigma=sigma, rho=rho)
    return scipy.stats.norm.pdf(x, loc=mean, scale=sqrt(var))


# @kfunc(nvar=2)
def hw2f_cdf(t, x, *, x0=(0., 0.), theta=(0., 0.), k=(1., 1.),
             sigma=(1., 1.), rho=0.):
    """Cumulative probability distribution function of values at time t
    of a Hull-White 2-factors process (as per the hull_white_process class)
    with time-independent parameters, evaluated at x.

    See Also
    --------
    hull_white_process
    """
    x1, x2, theta1, theta2, k1, k2, s1, s2, rho, t, x = \
        _hw2f_args_setup(x0, theta, k, sigma, rho, t, x)
    mean = hw2f_mean(t, x0=x0, theta=theta, k=k, sigma=sigma, rho=rho)
    var = hw2f_var(t, x0=x0, theta=theta, k=k, sigma=sigma, rho=rho)
    return scipy.stats.norm.cdf(x, loc=mean, scale=sqrt(var))


# --------------------------------
# cox_ingersoll_ross_process stats
# --------------------------------

# @kfunc(nvar=1)
def cir_mean(t, *, x0=1., theta=1., k=1., xi=1.):
    """
    cir_mean(t, *, x0=1., theta=1., k=1., xi=1.)

    Mean of values at time t of a Cox-Ingersoll-Ross process (as per the
    cox_ingersoll_ross_process class) with time-independent parameters.

    See Also
    --------
    cox_ingersoll_ross_process
    """
    t, x0, theta, k, xi = np.broadcast_arrays(t, x0, theta, k, xi)
    alpha = exp(-k*t)
    return theta + (x0 - theta) * alpha


# @kfunc(nvar=1)
def cir_var(t, *, x0=1., theta=1., k=1., xi=1.):
    """
    cir_var(t, *, x0=1., theta=1., k=1., xi=1.)

    Variance of values at time t of a Cox-Ingersoll-Ross process (as per the
    cox_ingersoll_ross_process class) with time-independent parameters.

    See Also
    --------
    cox_ingersoll_ross_process
    """
    t, x0, theta, k, xi = np.broadcast_arrays(t, x0, theta, k, xi)
    alpha = exp(-k*t)
    return (xi*xi / k) * (1 - alpha) * \
        (theta/2 + (x0 - theta/2) * alpha)


# @kfunc(nvar=1)
def cir_std(t, *, x0=1., theta=1., k=1., xi=1.):
    """
    cir_std(t, *, x0=1., theta=1., k=1., xi=1.)

    Standard deviation of values at time t of a Cox-Ingersoll-Ross process
    (as per the cox_ingersoll_ross_process class)
    with time-independent parameters.

    See Also
    --------
    cox_ingersoll_ross_process
    """
    return sqrt(cir_var(t, x0=x0, theta=theta, k=k, xi=xi))


# @kfunc(nvar=2)
def cir_pdf(t, x, *, x0=1., theta=1., k=1., xi=1.):
    """
    cir_pdf(t, x, *, x0=1., theta=1., k=1., xi=1.)

    Probability distribution function of values at time t of a
    Cox-Ingersoll-Ross process (as per the cox_ingersoll_ross_process class)
    with time-independent parameters, evaluated at x.

    See Also
    --------
    cox_ingersoll_ross_process
    """
    t, x0, theta, k, xi = np.broadcast_arrays(t, x0, theta, k, xi)
    alpha = exp(-k*t)
    c = 2*k / (1 - alpha) / (xi*xi)
    q = 2*k * theta / (xi*xi) - 1
    return c*exp(-c * (x0*alpha + x)) * \
        (x / (x0*alpha))**(q/2) * \
        scipy.special.iv(q, 2*c * sqrt(x*x0*alpha))


# --------------------
# heston_process stats
# --------------------

# @kfunc(nvar=1)
def heston_log_mean(t, *, x0=1., mu=0., sigma=1.,
                    y0=1., theta=1., k=1.,
                    xi=1., rho=0.):
    """
    heston_log_mean(t, *, x0=1., mu=0., sigma=1., y0=1., theta=1., k=1.,
                    xi=1., rho=0.)

    Mean of the logarithm of values at time t of a Heston process
    (as per the full_heston_process class) with time-independent parameters.

    See Also
    --------
    full_heston_process
    """
    t, x0, mu, sigma, y0, theta, k, xi, rho = \
        np.broadcast_arrays(t, x0, mu, sigma, y0, theta, k, xi, rho)
    alpha = exp(-k*t)
    return log(x0) + \
        sigma*sigma * (theta - y0) * (1-alpha) / (2*k) + \
        (mu - theta*sigma*sigma/2) * t


# @kfunc(nvar=1)
def heston_log_var(t, *, x0=1., mu=0., sigma=1.,
                   y0=1., theta=1., k=1.,
                   xi=1., rho=0.):
    """Variance of the logarithm of values at time t of a Heston process
    (as per the full_heston_process class) with time-independent parameters.

    See Also
    --------
    full_heston_process
    """
    t, x0, mu, sigma, y0, theta, k, xi, rho = \
        np.broadcast_arrays(t, x0, mu, sigma, y0, theta, k, xi, rho)
    alpha = exp(-k*t)
    omega1 = (alpha*sigma*xi)**2 + \
        4*alpha * ((1 + k*t)*(sigma*xi)**2 -
                   2*rho*k*sigma*xi*(2 + k*t) +
                   2*k*k) + \
        (2*k*t - 5) * (sigma*xi)**2 - \
        8*rho*k*sigma*xi * (k*t - 2) + \
        8*k*k * (k*t - 1)
    omega2 = -(alpha*sigma*xi)**2 + \
        2*alpha * (-(sigma*xi)**2 * k*t +
                   2*rho*sigma*xi*k * (k*t + 1) -
                   2*k*k) + \
        (sigma*xi)**2 - \
        4*rho*k*sigma*xi + \
        4*k*k

    return sigma*sigma*(theta*omega1/2 + y0*omega2) / (4*k**3)


# @kfunc(nvar=1)
def heston_log_std(t, *, x0=1., mu=0., sigma=1.,
                   y0=1., theta=1., k=1.,
                   xi=1., rho=0.):
    """
    heston_log_std(t, *, x0=1., mu=0., sigma=1., y0=1., theta=1., k=1.,
                   xi=1., rho=0.)

    Standard deviation of the logarithm of values at time t of a Heston
    process (as per the full_heston_process class)
    with time-independent parameters.

    See Also
    --------
    full_heston_process
    """
    return sqrt(heston_log_var(t, x0=x0, mu=mu, sigma=sigma,
                               y0=y0, theta=theta, k=k,
                               xi=xi, rho=rho))


# @kfunc(nvar=2)
def heston_log_chf(t, u, *, x0=1., mu=0., sigma=1.,
                   y0=1., theta=1., k=1.,
                   xi=1., rho=0.):
    """
    heston_log_chf(t, u, *, x0=1., mu=0., sigma=1., y0=1., theta=1., k=1.,
                   xi=1., rho=0.)

    Characteristic function of the probability distribution of values at
    time t of the logarithm of a Heston process (as per the full_heston_process
    class) , with time-independent parameters, evaluated at u.

    See Also
    --------
    full_heston_process
    """
    t, u, x0, mu, sigma, y0, theta, k, xi, rho = \
        np.broadcast_arrays(t, u, x0, mu, sigma, y0, theta, k, xi, rho)
    d = sqrt((rho*sigma*xi*u*1j - k)**2 + (sigma*xi)**2 * u*(1j + u))
    ap, am = (k - rho*sigma*xi*u*1.j - d), (k - rho*sigma*xi*u*1j + d)
    g = ap/am
    alpha = exp(-d*t)

    A = 1j*u*(log(x0) + mu*t)
    A += theta*k/(xi*xi) * (ap*t - 2*log((1 - g*alpha)/(1-g)))
    A += y0/(xi*xi) * ap * (1 - alpha)/(1 - g*alpha)
    return exp(A)


def _chf_to_pdf(t, x, chf, **chf_args):
    """
    Estimate by numerical integration, using ``scipy.integrate.quad``,
    of the probability distribution described by the given characteristic
    function. Integration errors are not reported/checked.
    Either ``t`` or ``x`` must be a scalar.
    """
    t = np.asarray(t)
    x = np.asarray(x)

    def f(u, t, x):
        return np.real(
            exp(-1j*u*x) / (2*np.pi) * chf(t, u, **chf_args))
    if t.shape != ():
        pdf = np.empty(t.shape)
        for i in np.ndindex(t.shape):
            pdf[i] = scipy.integrate.quad(
                lambda u: f(u, t[i], x), -np.inf, np.inf)[0]
    else:
        pdf = np.empty(x.shape)
        for i in np.ndindex(x.shape):
            pdf[i] = scipy.integrate.quad(
                lambda u: f(u, t, x[i]), -np.inf, np.inf)[0]
    return pdf


# @kfunc(nvar=2)
def heston_log_pdf(t, logx, *, x0=1., mu=0., sigma=1.,
                   y0=1., theta=1., k=1.,
                   xi=1., rho=0.):
    """
    heston_log_pdf(t, logx, *, x0=1., mu=0., sigma=1., y0=1., theta=1., k=1.,
                   xi=1., rho=0.)

    Probability distribution function of values at time t of the
    logarithm of a Heston process, (as per the full_heston_process class)
    with time-independent parameters, evaluated at logx.

    Notes
    -----
    Estimate by numerical integration, using ``scipy.integrate.quad``,
    of the closed-form characteristic function ``heston_log_chf``.
    Integration errors are not reported/checked. Either ``t`` or ``logx``
    must be a scalar.

    See Also
    --------
    full_heston_process
    """
    return _chf_to_pdf(t, logx, heston_log_chf,
                       x0=x0, mu=mu, sigma=sigma,
                       y0=y0, theta=theta,
                       k=k, xi=xi, rho=rho)


# -----------------------------------
# merton_jumpdiff_process stats
# -----------------------------------

# @kfunc(nvar=1)
def mjd_mean(t, *, x0=1., mu=0., sigma=1., lam=1., a=0.0, b=1.):
    """
    mjd_mean(t, *, x0=1., mu=0., sigma=1., lam=1., a=0.0, b=1.)

    Mean of values at time t of a Merton jump-diffusion process, (as per the
    merton_jumpdiff_process class) with time-independent parameters.

    See Also
    --------
    jumpdiff_process
    merton_jumpdiff_process
    """
    t, x0, mu, sigma, lam, a, b = \
        np.broadcast_arrays(t, x0, mu, sigma, lam, a, b)

    # martingale correction *NOT* applied
    exp_mean = exp(a + b*b/2)
    nu = lam*(exp_mean - 1)
    mu1 = mu + nu

    return x0*exp(mu1*t)


# @kfunc(nvar=2)
def mjd_log_chf(t, u, *, x0=1., mu=0., sigma=1., lam=1., a=0.0, b=1.):
    """
    mjd_log_chf(t, u, *, x0=1., mu=0., sigma=1., lam=1., a=0.0, b=1.)

    Characteristic function of the probability distribution of values
    at time t of the logarithm of a Merton jump-diffusion process
    (as per the merton_jumpdiff_process class), with
    time-independent parameters, evaluated at u.

    See Also
    --------
    jumpdiff_process
    merton_jumpdiff_process
    """

    t, u, x0, mu, sigma, lam, a, b = \
        np.broadcast_arrays(t, u, x0, mu, sigma, lam, a, b)

    # martingale correction *NOT* applied
    #   exp_mean = exp(a + b*b/2)
    #   nu = lam*(exp_mean - 1)
    #   mu0 = mu - sigma*sigma/2 - nu

    mu0 = mu - sigma*sigma/2
    A = 1j*u*log(x0) + \
        t*(+ 1j*mu0*u -
           sigma*sigma*u*u/2 +
           lam * (exp(1j*a*u - b*b*u*u/2) - 1))

    return exp(A)


# @kfunc(nvar=2)
def mjd_log_pdf(t, logx, *, x0=1., mu=0., sigma=1., lam=1., a=0.0, b=1.):
    """
    mjd_log_pdf(t, logx, *, x0=1., mu=0., sigma=1., lam=1., a=0.0, b=1.)

    Probability distribution function of values at time t of the logarithm
    of a Merton jump-diffusion process (as per the
    merton_jumpdiff_process class), with time-independent parameters,
    evaluated at logx.

    Notes
    -----
    Estimate by numerical integration, using ``scipy.integrate.quad``,
    of the closed-form characteristic function ``mjd_log_chf``.
    Integration errors are not reported/checked. Either ``t`` or ``logx``
    must be a scalar.

    See Also
    --------
    jumpdiff_process
    merton_jumpdiff_process
    mjd_log_chf
    """
    return _chf_to_pdf(t, logx, mjd_log_chf,
                       x0=x0, mu=mu, sigma=sigma,
                       lam=lam, a=a, b=b)


# --------------------------------
# kou_jumpdiff_process stats
# --------------------------------

# @kfunc(nvar=1)
def kou_mean(t, *, x0=1., mu=0., sigma=1., lam=1., a=0.5, b=0.5, pa=0.5):
    """
    kou_mean(t, *, x0=1., mu=0., sigma=1., lam=1., a=0.5, b=0.5, pa=0.5)

    Mean of values at time t of a double exponential (Kou)
    jump-diffusion process (as per the kou_jumpdiff_process class)
    with time-independent parameters.

    See Also
    --------
    jumpdiff_process
    kou_jumpdiff_process
    """
    t, x0, mu, sigma, lam, a, b, pa = \
        np.broadcast_arrays(t, x0, mu, sigma, lam, a, b, pa)
    pb = 1 - pa

    # martingale correction *NOT* applied
    exp_mean = pa/(1 - a) + pb/(1 + b)
    nu = lam*(exp_mean - 1)
    mu1 = mu + nu

    return x0*exp(mu1*t)


# @kfunc(nvar=2)
def kou_log_chf(t, u, *, x0=1., mu=0., sigma=1., lam=1., a=0.5, b=0.5, pa=0.5):
    """
    kou_log_chf(t, u, *, x0=1., mu=0., sigma=1., lam=1., a=0.5, b=0.5, pa=0.5)

    Characteristic function of the probability distribution of values
    at time t of the logarithm of a Kou jump-diffusion process,
    (as per the kou_jumpdiff_process class) with
    time-independent parameters, evaluated at u.

    See Also
    --------
    jumpdiff_process
    kou_jumpdiff_process
    """
    t, u, x0, mu, sigma, lam, a, b, pa = \
        np.broadcast_arrays(t, u, x0, mu, sigma, lam, a, b, pa)
    pb = 1 - pa

    # martingale correction *NOT* applied
    #   exp_mean = pa/(1 - a) + pb/(1 + b)
    #   nu = lam*(exp_mean - 1)
    #   mu0 = mu - sigma*sigma/2 - nu

    mu0 = mu - sigma*sigma/2
    A = 1j*u*log(x0) + \
        t*(+ 1j*mu0*u -
           sigma*sigma*u*u/2 +
           1j*u*lam*(pa/(1/a - 1j*u) - pb/(1/b + 1j*u)))

    return exp(A)


# @kfunc(nvar=2)
def kou_log_pdf(t, logx, *, x0=1., mu=0., sigma=1., lam=1.,
                pa=0.5, a=0.5, b=0.5):
    """
    kou_log_pdf(t, logx, *, x0=1., mu=0., sigma=1., lam=1.,
                pa=0.5, a=0.5, b=0.5)

    Probability distribution function of values at time t of the logarithm
    of a double-exponential (Kou) jump-diffusion process (as per the
    kou_jumpdiff_process class), with time-independent parameters,
    evaluated at logx.

    Notes
    -----
    Estimate by numerical integration, using ``scipy.integrate.quad``,
    of the closed-form characteristic function ``kou_log_chf``.
    Integration errors are not reported/checked. Either ``t`` or ``logx``
    must be a scalar.

    See Also
    --------
    jumpdiff_process
    kou_jumpdiff_process
    kou_log_chf
    """
    return _chf_to_pdf(t, logx, kou_log_chf,
                       x0=x0, mu=mu, sigma=sigma,
                       lam=lam, a=a, b=b, pa=pa)


#################################################
# Black-Scholes formulae for call and put options
#################################################

# @kfunc(nvar=2)
def bsd1d2(k, t, *, x0=1., r=0., q=0., sigma=1.):
    """
    bsd1d2(k, t, *, x0=1., r=0., q=0., sigma=1.)

    Black-Scholes d1 and d2 coefficients.

    See Also
    --------
    bscall
    """
    # convert inputs to arrays
    k, t, x0, r, q, sigma = np.broadcast_arrays(
        k, t, x0, r, q, sigma)

    # compute d1 and d2, handling limiting cases t==0 and/or x0==k
    # and/or sigma==0
    A = log(x0/k) + (r - q + 0.5 * sigma**2) * t
    B = sigma * sqrt(t)
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    d1 = np.where(B == 0, np.where(A == 0, 0., np.inf * A), A / B)
    np.seterr(**old_settings)
    assert not np.isnan(d1).any()
    d2 = d1 - B

    return d1, d2


# @kfunc(nvar=2)
def bscall(k, t, *, x0=1., r=0., q=0., sigma=1.):
    """
    bscall(k, t, *, x0=1., r=0., q=0., sigma=1.)

    Black-Scholes call option value.

    Parameters
    ----------
    k : array-like
        Strike.
    t : array-like
        Time to maturity.
    x0 : array-like
        Initial value of underlying security.
    r : array-like
        Risk-free rate.
    q : array-like
        Dividend yield of underlying security.
    sigma : array-like
        Volatility of underlying security.

    Returns
    -------
    array
        Risk neutral valuation at time s=0 of an European call option
        paying ``max(x(t) - k, 0)`` at maturity, where the price ``x(s)``
        of the underlying security follows a lognormal process
        with ``x(0) = x0`` and volatility ``sigma``.

    See Also
    --------
    bsd1d2
    bscall_delta
    bsput
    bsput_delta

    Notes
    -----
    ``bscall(k, t, x0, r, q, sigma)`` returns::

        bscall_value = x0*exp(-q*t)*norm.cdf(d1) + k*exp(-r*t)*norm.cdf(d2)

    where ``cdf`` is ``scipy.stats.norm.cdf`` and
    ``d1, d2 = bsd1d2(k, t, x0, r, q, sigma)`` are given as::

        d1 = (log(x0/k) + (r - q + sigma**2/2)*t)/(sigma*sqrt(t))

        d2 = d1 - sigma*sqrt(t)
    """
    r, q = np.asarray(r), np.asarray(q)
    d1, d2 = bsd1d2(k, t, x0=x0, r=r, q=q, sigma=sigma)
    return x0 * exp(-q*t) * scipy.stats.norm.cdf(d1) - \
        k * exp(-r*t) * scipy.stats.norm.cdf(d2)


# @kfunc(nvar=2)
def bscall_delta(k, t, *, x0=1., r=0., q=0., sigma=1.):
    """
    bscall_delta(k, t, *, x0=1., r=0., q=0., sigma=1.)

    Black-Scholes call option delta.

    See Also
    --------
    bscall
    """
    r, q = np.asarray(r), np.asarray(q)
    d1, d2 = bsd1d2(k, t, x0=x0, r=r, q=q, sigma=sigma)
    return exp(-q*t) * scipy.stats.norm.cdf(d1)


# @kfunc(nvar=2)
def bsput(k, t, *, x0=1., r=0., q=0., sigma=1.):
    """
    bsput(k, t, *, x0=1., r=0., q=0., sigma=1.)

    Black-Scholes put option value.

    See Also
    --------
    bscall
    """
    r, q = np.asarray(r), np.asarray(q)
    d1, d2 = bsd1d2(k, t, x0=x0, r=r, q=q, sigma=sigma)
    return k * exp(-r*t) * scipy.stats.norm.cdf(-d2) - \
        x0 * exp(-q*t) * scipy.stats.norm.cdf(-d1)


# @kfunc(nvar=2)
def bsput_delta(k, t, *, x0=1., r=0., q=0., sigma=1.):
    """
    bsput_delta(k, t, *, x0=1., r=0., q=0., sigma=1.)

    Black-Scholes put option delta.

    See Also
    --------
    bscall
    """
    r, q = np.asarray(r), np.asarray(q)
    d1, d2 = bsd1d2(k, t, x0=x0, r=r, q=q, sigma=sigma)
    return -exp(-q*t) * scipy.stats.norm.cdf(-d1)
