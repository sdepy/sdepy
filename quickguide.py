# ------------------------------------------
# This file has been automatically generated
# from .\doc\quickguide.rst
# ------------------------------------------

import sdepy
from sdepy import *  # safe and handy for interactive sessions
import numpy as np
import scipy
import matplotlib.pyplot as plt  # optional, if plots are needed
@integrate
def my_process(t, x, theta=1., k=1., sigma=1.):
    return {'dt': k*(theta - x), 'dw': sigma}
issubclass(my_process, integrator), issubclass(my_process, SDE)
myp = kfunc(my_process)
iskfunc(myp)
coarse_timeline = (0., 0.25, 0.5, 0.75, 1.0)
np.random.seed(1)  # make doctests predictable
x = my_process(x0=1, paths=100*1000,
               steps=100)(coarse_timeline)
x.shape
corr = ((1, .2, -.3), (.2, 1, .1), (-.3, .1, 1))
x = my_process(x0=1, vshape=3, corr=corr,
               paths=100*1000, steps=100)(coarse_timeline)
x.shape
timeline = np.linspace(0., 1., 101)
corr = lambda t: ((1, .2, -.1*t), (.2, 1, .1), (-.1*t, .1, 1))
theta, k, sigma = (lambda t: 2-t, lambda t: 2/(t+1), lambda t: np.sin(t/2))
x = my_process(x0=1, vshape=3, corr=corr,
               theta=theta, k=k, sigma=sigma, paths=10*1000)(timeline)
x.shape
gr = plt.plot(timeline, x[:, 0, :4])  # inspect a few paths
plt.show(gr) # doctest: +SKIP
x0 = np.random.random(10*1000)
sigma = 1 + np.random.random(10*1000)
x = my_process(x0=x0, sigma=sigma, paths=10*1000,
               i0=-1)(timeline)
x.shape
(x[-1, :] == x0).all()
sigma = np.linspace(0., 1., 10).reshape(10, 1, 1)
k = np.linspace(1., 2., 15).reshape(1, 15, 1)
x = my_process(x0=1, theta=2, k=k, sigma=sigma, vshape=(10, 15),
               paths=10*1000)(coarse_timeline)
x.shape
gr = plt.plot(coarse_timeline, x[:, 5, ::2, :].mean(axis=-1))
plt.show() # doctest: +SKIP
my_dw = integrate(lambda t, x: {'dw': 1})(vshape=1, paths=10000)(timeline)
p = myp(dw=my_dw, vshape=3, paths=10000,
        x0=1, sigma=((1,), (2,), (3,)))  # using myp = kfunc(my_process)
x = p(timeline)
x.shape
k = 0  # path to be plotted
gr = plt.plot(timeline, x[:, :, k])
plt.show()  # doctest: +SKIP
x = p(coarse_timeline, steps=timeline)
x.shape
my_dw = true_wiener_source(paths=10000)
p = myp(x0=1, k=1, sigma=1, dw=my_dw, paths=10000)
t1 = np.linspace(0., 1.,  30)
t2 = np.linspace(0., 1., 100)
t3 = t = np.linspace(0., 1., 300)
x1, x2, x3 = p(t1), p(t2), p(t3)
y1, y2, y3 = p(t, theta=1.5), p(t, theta=1.75), p(t, theta=2)
i = 0 # path to be plotted
gr = plt.plot(t, x1(t)[:, i], t, x2(t)[:, i], t, x3(t)[:, i])
gr = plt.plot(t, y1[:, i], t3, y2[:, i], t3, y3[:, i])
plt.show() # doctest: +SKIP
coarse_timeline = (0., 0.25, 0.5, 0.75, 1.0)
timeline = np.linspace(0., 1., 101)
x = my_process(x0=1, vshape=3, paths=1000)(timeline)
x.shape
type(x)
np.isclose(timeline, x.t).all()
timeline is x.t
x.shape == x.t.shape + x.vshape + (x.paths,)
y = x(coarse_timeline)
y.shape
type(y)
type(x[0])
type(x.mean(axis=0))
y = x['t', ::2]  # time indexing
y.shape
y = x['v', 0]  # values indexing
y.shape
y = x['p', :10]  # paths indexing
y.shape
isinstance(y, process)
i_negative = x.min(axis=(0, 1)) < 0
y = x['p', i_negative]
y.shape == (101, 3, i_negative.sum())
x_const = x['t', 0]  # a constant process
x_one_path = x['p', 0]  # a process with one path
y = np.exp(x) - x_const
z = np.maximum(x, x_one_path)
isinstance(y, process), isinstance(z, process)
stochastic_vol = my_process(x0=1, paths=10*1000)(timeline)
stochastic_vol_x = lognorm_process(x0=1, vshape=3, paths=10*1000,
    mu=0, sigma=stochastic_vol)(timeline)
cdf = x.cdf(0.5, x=np.linspace(-2, 2, 100))  # an array
chf = x.chf(0.5, u=np.linspace(-2, 2, 100))  # an array
std = x.pstd()  # a one-path process
std.shape
xmax = x.tmax()  # a constant process
xmax.shape
y = x(1)[0]  # 0-th component of x at time t=1
a = montecarlo(y, bins=30)
ygrid = np.linspace(y.min(), y.max(), 200)
gr = plt.plot(ygrid, a.pdf(ygrid), ygrid, a.cdf(ygrid))
gr = plt.plot(ygrid, a.pdf(ygrid, method='interp', kind='nearest'))
plt.show()  # doctest: +SKIP
p = my_process(x0=1, vshape=3, paths=10*1000)
a = montecarlo(bins=100)  # empty montecarlo instance
for _ in range(10):
    x = p(timeline)  # run simulation
    a.update(x(1))  # cumulate x values at t=1
a.paths
gr = plt.plot(ygrid, a[0].pdf(ygrid), ygrid, a[0].cdf(ygrid))
gr = plt.plot(ygrid, a[0].pdf(ygrid, method='interp', kind='nearest'))
plt.show()  # doctest: +SKIP
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
class my_SDE(SDE):
    def sde(self, t, x): return {'dt': 0, 'dw': x}
class p1(my_SDE, integrator): pass
class p2(my_SDE, my_integrator): pass
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
from numpy import exp, sin
from scipy.special import erf
from scipy.integrate import quad
np.random.seed(1)
k = .5
x0, x1 = 0, 10;
t0, t1 = 0, 1
lb, hb = 4, 6
def green_exact(y, s, x, t):
    return exp(-(x - y)**2/(4*k*(t - s)))/sqrt(4*np.pi*k*(t - s))
def u1_exact(x, t):
    return (erf((x - lb)/2/sqrt(k*(t - t0))) - erf((x - hb)/2/sqrt(k*(t - t0))))/2
def u2_exact(x, t):
    return exp(-k*(t - t0))*sin(x)
xgrid = np.linspace(x0, x1, 51)
tgrid = np.linspace(t0, t1, 5)
xp = wiener_process(paths=10000,
            sigma=sqrt(2*k), steps=100,
            vshape=xgrid.shape, x0=xgrid[..., np.newaxis],
            i0=-1)(timeline=tgrid)
a = montecarlo(xp, bins=100)
def green(y, i, j):
    """green function from (y=y, s=tgrid[i]) to (x=xgrid[j], t=t1)"""
    return a[i, j].pdf(y)
u1, u2 = np.empty(51), np.empty(51)
for j in range(51):
    u1[j] = quad(lambda y: green(y, 0, j), lb, hb)[0]
    u2[j] = quad(lambda y: sin(y)*green(y, 0, j), -np.inf, np.inf)[0]
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
corr = [
    [1,    0.50, 0.37, 0.35],
    [0.50,    1, 0.47, 0.46],
    [0.37, 0.47,    1, 0.19],
    [0.35, 0.46,  0.19,   1]]
dividend_yield = process(c=(0.20, 4.40, 0., 4.80))/100
riskfree = 0  # to keep it simple
vol_timepoints = (0.1, 0.2, 0.5, 1, 2, 3)
vol = np.array([
    [0.40, 0.38, 0.30, 0.28, 0.27, 0.27],
    [0.31, 0.29, 0.22, 0.16, 0.18, 0.21],
    [0.24, 0.22, 0.19, 0.19, 0.21, 0.22],
    [0.35, 0.31, 0.21, 0.18, 0.19, 0.19]])
sigma = process(t=vol_timepoints, v=vol.T)
sigma.shape
maturity = 2
timeline = np.linspace(0, maturity, 4*maturity + 1)
p = lognorm_process(x0=100, corr=corr, dw=odd_wiener_source,
                    mu=(riskfree - dividend_yield),
                    sigma=sigma,
                    vshape=4, paths=100*1000, steps=maturity*250)
np.random.seed(1)
x = p(timeline)
x.shape
x_worst = x.min(axis=1)
x_mean = x.mean(axis=1)
down_and_in_paths = (x_worst.min(axis=0) < 80)
lookback_x_mean = x_mean.max(axis=0)
payoff = np.maximum(0, lookback_x_mean - 105)
payoff[np.logical_not(down_and_in_paths)] = 0
a = montecarlo(payoff, use='even')
print(a)  # doctest: +SKIP
