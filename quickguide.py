# ------------------------------------------
# This file has been automatically generated
# from .\doc\quickguide.rst
# ------------------------------------------

# ===========
# Quick Guide
# ===========
#
#
# ------------------
# Install and import
# ------------------
#
# Install using ``pip install sdepy``, or copy the package source code
# in a directory in your Python path.
#
# Import as
#
import sdepy
from sdepy import *  # safe and handy for interactive sessions
import numpy as np
import scipy
import matplotlib.pyplot as plt  # optional, if plots are needed


#
#
# -------------------
# How to state an SDE
# -------------------
#
# Here follows a bare-bone definition of a Stochastic Differential
# Equation (SDE), in this case a Ornstein-Uhlenbeck process:
#
@integrate
def my_process(t, x, theta=1., k=1., sigma=1.):
    return {'dt': k*(theta - x), 'dw': sigma}


#
# This represents the SDE ``dX = k*(theta - X)*dt + sigma*dW(t)``,
# where ``theta``, ``k`` and ``sigma`` are parameters and ``dW(t)`` are Wiener
# process increments. A further ``'dn'`` or ``'dj'`` entry in the returned
# dictionary would allow for Poisson or compound Poisson jumps.
#
# A number of preset processes are provided, including lognormal processes,
# Hull-White n-factor processes, Heston processes, and jump-diffusion processes.
#
#
# -----------------------
# How to integrate an SDE
# -----------------------
#
# Now ``my_process`` is a class, a subclass of the cooperating ``SDE``
# and ``integrator`` classes:
#
issubclass(my_process, integrator), issubclass(my_process, SDE)


# (True, True)
#
# It is to be instantiated with a number
# of parameters, including the SDE parameters ``theta``, ``k`` and ``sigma``;
# its instances are callable, given a timeline they will integrate and
# return the process along it. Decorating ``my_process`` with ``kunfc``
# allows for more concise handling of parameters:
#
myp = kfunc(my_process)
iskfunc(myp)


# True
#
# It is best explained by examples:
#
#
# 1. **Scalar process** in 100000 paths, with default parameters, computed
# at 5 time points, using 100 steps in between::
#
coarse_timeline = (0., 0.25, 0.5, 0.75, 1.0)
np.random.seed(1)  # make doctests predictable
x = my_process(x0=1, paths=100*1000,
               steps=100)(coarse_timeline)
x.shape


# (5, 100000)
#
#
# 2. **Vector process** with three components and **correlated Wiener increments**
# (same parameters, paths, timeline and steps as above)::
#
corr = ((1, .2, -.3), (.2, 1, .1), (-.3, .1, 1))
x = my_process(x0=1, vshape=3, corr=corr,
               paths=100*1000, steps=100)(coarse_timeline)
x.shape


# (5, 3, 100000)
#
# 3. Vector process with **time-dependent parameters and correlations**,
# computed on a fine-grained timeline and 10000 paths, using one
# integration step for each point in the timeline (no ``steps`` parameter)::
#
timeline = np.linspace(0., 1., 101)
corr = lambda t: ((1, .2, -.1*t), (.2, 1, .1), (-.1*t, .1, 1))
theta, k, sigma = (lambda t: 2-t, lambda t: 2/(t+1), lambda t: np.sin(t/2))
x = my_process(x0=1, vshape=3, corr=corr,
               theta=theta, k=k, sigma=sigma, paths=10*1000)(timeline)
x.shape


# (101, 3, 10000)
gr = plt.plot(timeline, x[:, 0, :4])  # inspect a few paths
plt.show(gr) # doctest: +SKIP


#
#
# 4. A scalar process with **path-dependent initial conditions and parameters**,
# integrated **backwards** (``i0=-1``)::
#
x0 = np.random.random(10*1000)
sigma = 1 + np.random.random(10*1000)
x = my_process(x0=x0, sigma=sigma, paths=10*1000,
               i0=-1)(timeline)
x.shape


# (101, 10000)
(x[-1, :] == x0).all()


# True
#
#
# 5. A scalar process computed on a **10 x 15 grid of parameters** ``sigma`` and
# ``k`` (note that the shape of the initial conditions and of each
# parameter should be broadcastable to the values of the process across
# paths, i.e. to shape ``vshape + (paths,)``)::
#
sigma = np.linspace(0., 1., 10).reshape(10, 1, 1)
k = np.linspace(1., 2., 15).reshape(1, 15, 1)
x = my_process(x0=1, theta=2, k=k, sigma=sigma, vshape=(10, 15),
               paths=10*1000)(coarse_timeline)
x.shape


# (5, 10, 15, 10000)
gr = plt.plot(coarse_timeline, x[:, 5, ::2, :].mean(axis=-1))
plt.show() # doctest: +SKIP


#
# In the example above, set ``steps=100`` to go from inaccurate and fast,
# to meaningful and slow (the plot illustrates the ``k``-dependence of
# average process values).
#
#
# 6. Processes generated using **integration results as stochasticity sources**
# (mind using consistent ``vshape`` and ``paths``, and synchronizing timelines)::
#
my_dw = integrate(lambda t, x: {'dw': 1})(vshape=1, paths=10000)(timeline)
p = myp(dw=my_dw, vshape=3, paths=10000,
        x0=1, sigma=((1,), (2,), (3,)))  # using myp = kfunc(my_process)
x = p(timeline)
x.shape


# (101, 3, 10000)
#
# Now, ``x1, x2, x3 = = x[:, 0], x[:, 1], x[:, 2]`` have different ``sigma``,
# but share the same ``dw`` increments, as can be seen plotting a path:
#
k = 0  # path to be plotted
gr = plt.plot(timeline, x[:, :, k])
plt.show()  # doctest: +SKIP


#
# If more integrations steps are needed between points in the output timeline,
# use ``steps`` to keep the integration timeline consistent with the one
# of ``my_dw``:
#
x = p(coarse_timeline, steps=timeline)
x.shape


# (5, 3, 10000)
#
#
# 7. Using **stochasticity sources with memory**
# (mind using consistent ``vshape`` and ``paths``)::
#
my_dw = true_wiener_source(paths=10000)
p = myp(x0=1, k=1, sigma=1, dw=my_dw, paths=10000)


#
t1 = np.linspace(0., 1.,  30)
t2 = np.linspace(0., 1., 100)
t3 = t = np.linspace(0., 1., 300)
x1, x2, x3 = p(t1), p(t2), p(t3)
y1, y2, y3 = p(t, theta=1.5), p(t, theta=1.75), p(t, theta=2)


#
# These processes share the same underlying Wiener increments:
# ``x1, x2, x3`` illustrate SDE integration convergence as steps become
# smaller, and ``y1, y2, y3`` illustrate how ``k`` affects paths,
# all else being equal::
#
i = 0 # path to be plotted
gr = plt.plot(t, x1(t)[:, i], t, x2(t)[:, i], t, x3(t)[:, i])
gr = plt.plot(t, y1[:, i], t3, y2[:, i], t3, y3[:, i])
plt.show() # doctest: +SKIP


#
#
# ------------------------------------
# How to handle the integration output
# ------------------------------------
#
# SDE integrators return ``process`` instances, a subclass of ``np.ndarray``
# with a timeline stored in the ``t`` attribute (note the shape of ``x``,
# repeatedly used in the examples below)::
#
coarse_timeline = (0., 0.25, 0.5, 0.75, 1.0)
timeline = np.linspace(0., 1., 101)
x = my_process(x0=1, vshape=3, paths=1000)(timeline)
x.shape


# (101, 3, 1000)
#
# ``x`` is a ``process`` instance, based on the given timeline:
#
type(x)


# <class 'sdepy.infrastructure.process'>
np.isclose(timeline, x.t).all()


# True
#
#
# Whenever possible, a process will store references, not copies, of timeline
# and values. In fact,
#
timeline is x.t


# True
#
#
# The first axis is reserved for the timeline, the last for paths, and axes
# in the middle match the shape of process values:
#
x.shape == x.t.shape + x.vshape + (x.paths,)


# True
#
#
# Calling processes interpolates in time (the result is an array, not a process)::
#
y = x(coarse_timeline)


#
y.shape


# (5, 3, 1000)
#
type(y)


# <class 'numpy.ndarray'>
#
#
# All array methods, including indexing, work as usual (no overriding),
# and return NumPy arrays::
#
type(x[0])


# <class 'numpy.ndarray'>
type(x.mean(axis=0))


# <class 'numpy.ndarray'>
#
#
# You can slice processes along time, values and paths with special indexing::
#
y = x['t', ::2]  # time indexing
y.shape


# (51, 3, 1000)
y = x['v', 0]  # values indexing
y.shape


# (101, 1000)
y = x['p', :10]  # paths indexing
y.shape


# (101, 3, 10)
#
# The output of a special indexing operation is a process:
#
isinstance(y, process)


# True
#
# Smart indexing is allowed. To select paths that cross ``x=0``
# at some point and for some component, use::
#
i_negative = x.min(axis=(0, 1)) < 0
y = x['p', i_negative]
y.shape == (101, 3, i_negative.sum())


# True
#
# You can do algebra with processes that either share the same timeline, or are constant
# (a process with a one-point timeline is assumed to be constant), and either have the
# same number of paths, or are deterministic (with one path)::
#
x_const = x['t', 0]  # a constant process
x_one_path = x['p', 0]  # a process with one path


#
y = np.exp(x) - x_const
z = np.maximum(x, x_one_path)


#
isinstance(y, process), isinstance(z, process)


# (True, True)
#
# When integrating SDEs, the SDE parameters and/or stochasticity sources
# accept processes as valid values (mind using deterministic processes, or
# synchronizing the number of paths, and make sure that the shape of values
# do broadcast together). To use a realization of ``my_process``
# as the volatility of a 3-component lognormal process, do as follows::
#
stochastic_vol = my_process(x0=1, paths=10*1000)(timeline)
stochastic_vol_x = lognorm_process(x0=1, vshape=3, paths=10*1000,
    mu=0, sigma=stochastic_vol)(timeline)


#
#
# Processes have specialized methods, and may be analyzed, and their statistics
# cumulated across multiple runs, using the ``montecarlo`` class. Some examples follow:
#
# 1. Cumulative probability distribution function at t=0.5
# of the process values of ``x`` across paths:
#
cdf = x.cdf(0.5, x=np.linspace(-2, 2, 100))  # an array


#
# 2. Characteristic function at t=0.5 of the same distribution:
#
chf = x.chf(0.5, u=np.linspace(-2, 2, 100))  # an array


#
# 3. Standard deviation across paths:
#
std = x.pstd()  # a one-path process
std.shape


# (101, 3, 1)
#
# 4. Maximum value reached along the timeline:
#
xmax = x.tmax()  # a constant process
xmax.shape


# (1, 3, 1000)
#
#
# 5. A linearly interpolated, or Gaussian kernel estimate (default)
# of the probability distribution function (pdf) and its cumulated
# values (cdf) across paths, at a given time point,
# may be obtained using the ``montecarlo`` class:
#
y = x(1)[0]  # 0-th component of x at time t=1
a = montecarlo(y, bins=30)
ygrid = np.linspace(y.min(), y.max(), 200)
gr = plt.plot(ygrid, a.pdf(ygrid), ygrid, a.cdf(ygrid))
gr = plt.plot(ygrid, a.pdf(ygrid, method='interp', kind='nearest'))
plt.show()  # doctest: +SKIP


#
#
# 6. A ``montecarlo`` instance can be used to cumulate the results
# of multiple simulations, across multiple components of process values::
#
p = my_process(x0=1, vshape=3, paths=10*1000)
a = montecarlo(bins=100)  # empty montecarlo instance
for _ in range(10):
    x = p(timeline)  # run simulation
    a.update(x(1))  # cumulate x values at t=1
a.paths


# 100000
gr = plt.plot(ygrid, a[0].pdf(ygrid), ygrid, a[0].cdf(ygrid))
gr = plt.plot(ygrid, a[0].pdf(ygrid, method='interp', kind='nearest'))
plt.show()  # doctest: +SKIP


#
#
# --------------------------------
# Example - Stochastic Runge-Kutta
# --------------------------------
#
# Minimal implementation of a basic stochastic Runge-Kutta integration,
# scheme, as a subclass of ``integrator`` (the ``A`` and ``dZ`` methods
# below are the standardized way in which equations are exposed
# to integrators)::
#
from numpy import sqrt
class my_integrator(integrator):
    def next(self):
        t, new_t = self.itervars['sw']
        x, new_x = self.itervars['xw']
        dt = new_t - t
        A, dZ = self.A(t, x), self.dZ(t, dt)
        a, b, dw = A['dt'], A['dw'], dZ['dw']
        b1 = self.A(t, x + a*dt + b*sqrt(dt))['dw']
        new_x[...] = x + a*dt + b*dw + (b1 - b)/2 * (dw**2 - dt)/sqrt(dt)


#
# SDE of a lognormal process, as a subclass of ``SDE``,
# and classes that integrate it with the default integration method (``p1``)
# and via ``my_integrator`` (``p2``)::
#
class my_SDE(SDE):
    def sde(self, t, x): return {'dt': 0, 'dw': x}
class p1(my_SDE, integrator): pass
class p2(my_SDE, my_integrator): pass


#
# Comparison of integration errors, as the integration from ``t=0`` to
# ``t=1`` is carried out with an increasing number of steps::
#
np.random.seed(1)
args = dict(dw=true_wiener_source(paths=100), paths=100, x0=10)
timeline = (0, 1)
steps = np.array((2, 3, 5, 10, 20, 30, 50, 100,
                  200, 300, 500, 1000, 2000, 3000))
exact = lognorm_process(mu=0, sigma=1, **args)(timeline)[-1].mean()
errors = np.abs(np.array([
    [p1(**args, steps=s)(timeline)[-1].mean()/exact - 1,
     p2(**args, steps=s)(timeline)[-1].mean()/exact - 1]
    for s in steps]))
ax = plt.axes(label=0); ax.set_xscale('log'); ax.set_yscale('log')
gr = ax.plot(steps, errors)
plt.show()  # doctest: +SKIP
print('euler error: {:.2e}\n   rk error: {:.2e}'.format(errors[-1,0], errors[-1,1]))


# euler error: 1.70e-03
# rk error: 8.80e-06
#
#
# --------------------------------
# Example - Fokker-Planck Equation
# --------------------------------
#
# Monte Carlo integration of partial differential equations, illustrated
# in the simplest example of the heat equation ``diff(u, t) - k*diff(u, x, 2) == 0``,
# for the function ``u(x, t)``, i.e. the Fokker-Planck equation for the SDE
# ``dX(t) = sqrt(2*k)*dW(t)``. Initial conditions at ``t=t0``, two examples::
#
# 1.  ``u(x, t0) = 1`` for ``lb < x < hb`` and ``0`` otherwise,
# 2.  ``u(x, t0) = sin(x)``.
#
# Setup::
#
from numpy import exp, sin
from scipy.special import erf
from scipy.integrate import quad
np.random.seed(1)
k = .5
x0, x1 = 0, 10;
t0, t1 = 0, 1
lb, hb = 4, 6


#
# Exact green function and solutions, to be checked against results::
#
def green_exact(y, s, x, t):
    return exp(-(x - y)**2/(4*k*(t - s)))/sqrt(4*np.pi*k*(t - s))
def u1_exact(x, t):
    return (erf((x - lb)/2/sqrt(k*(t - t0))) - erf((x - hb)/2/sqrt(k*(t - t0))))/2
def u2_exact(x, t):
    return exp(-k*(t - t0))*sin(x)


#
# Realization of the needed stochastic process, by backward integration from
# a grid of final values of ``x`` at ``t=t1``, using the preset
# ``wiener_process`` class (the ``steps`` keyword is added as a reminder
# of the setup needed for less-than-trivial equations, it does not actually
# make a difference here)::
#
xgrid = np.linspace(x0, x1, 51)
tgrid = np.linspace(t0, t1, 5)
xp = wiener_process(paths=10000,
            sigma=sqrt(2*k), steps=100,
            vshape=xgrid.shape, x0=xgrid[..., np.newaxis],
            i0=-1)(timeline=tgrid)


#
# Computation of the green function and of the solution ``u(x, t1)``
# (note the liberal use of ``scipy.integrate.quad`` below, enabled by
# the smoothness of the Gaussian kernel estimate ``a[i, j].pdf``)::
#
a = montecarlo(xp, bins=100)
def green(y, i, j):
    """green function from (y=y, s=tgrid[i]) to (x=xgrid[j], t=t1)"""
    return a[i, j].pdf(y)
u1, u2 = np.empty(51), np.empty(51)
for j in range(51):
    u1[j] = quad(lambda y: green(y, 0, j), lb, hb)[0]
    u2[j] = quad(lambda y: sin(y)*green(y, 0, j), -np.inf, np.inf)[0]


#
# Comparison against exact values::
#
y = np.linspace(x0, x1, 500)
for i, j in ((1, 20), (2, 30), (3, 40)):
    gr = plt.plot(y, green(y, i, j),
                  y, green_exact(y, tgrid[i], xgrid[j], t1), ':')
plt.show()  # doctest: +SKIP
gr = plt.plot(xgrid, u1, y, u1_exact(y, t1), ':')
gr = plt.plot(xgrid, u2, y, u2_exact(y, t1), ':')
plt.show()  # doctest: +SKIP
print('u1 error: {:.2e}\nu2 error: {:.2e}'.format(
    np.abs(u1 - u1_exact(xgrid, t1)).mean(),
    np.abs(u2 - u2_exact(xgrid, t1)).mean()))


# u1 error: 2.49e-03
# u2 error: 5.51e-03
#
#
# --------------------------------
# Example - Basket Lookback Option
# --------------------------------
#
# Take a basket of 4 financial securities, with risk-neutral probabilities following
# lognormal processes in the Black-Sholes framework. Correlations, dividend yields
# and term structure of volatility (will be linearly interpolated) are given below::
#
corr = [
    [1,    0.50, 0.37, 0.35],
    [0.50,    1, 0.47, 0.46],
    [0.37, 0.47,    1, 0.19],
    [0.35, 0.46,  0.19,   1]]


#
dividend_yield = process(c=(0.20, 4.40, 0., 4.80))/100
riskfree = 0  # to keep it simple


#
vol_timepoints = (0.1, 0.2, 0.5, 1, 2, 3)
vol = np.array([
    [0.40, 0.38, 0.30, 0.28, 0.27, 0.27],
    [0.31, 0.29, 0.22, 0.16, 0.18, 0.21],
    [0.24, 0.22, 0.19, 0.19, 0.21, 0.22],
    [0.35, 0.31, 0.21, 0.18, 0.19, 0.19]])
sigma = process(t=vol_timepoints, v=vol.T)
sigma.shape


# (6, 4, 1)
#
# The prices of the securities at the end of each quarter for the next 2 years,
# simulated across 50000 independent paths and their antithetics
# (``odd_wiener_source`` is used), are::
#
maturity = 2
timeline = np.linspace(0, maturity, 4*maturity + 1)
p = lognorm_process(x0=100, corr=corr, dw=odd_wiener_source,
                    mu=(riskfree - dividend_yield),
                    sigma=sigma,
                    vshape=4, paths=100*1000, steps=maturity*250)
np.random.seed(1)
x = p(timeline)
x.shape


# (9, 4, 100000)
#
# A call option knocks in if any of the securities reaches a price below 80
# at any quarter (starting from 100), and pays the lookback maximum attained
# by the basket (equally weighted), minus 105, if positive.
# Its price is::
#
x_worst = x.min(axis=1)
x_mean = x.mean(axis=1)
down_and_in_paths = (x_worst.min(axis=0) < 80)
lookback_x_mean = x_mean.max(axis=0)
payoff = np.maximum(0, lookback_x_mean - 105)
payoff[np.logical_not(down_and_in_paths)] = 0
a = montecarlo(payoff, use='even')
print(a)  # doctest: +SKIP


# 4.997 +/- 0.027
