"""
test_histogram.py

Performs repeated coinflip simulations to ensure that the variance
of the estimators agrees with analytics
"""
# Make sure that stack can be found!
import sys
sys.path.append('../../')

import random
import numpy as np
from numpy import sqrt, exp
from scipy.integrate import quad
from scipy.interpolate import interp1d
from stack.montecarlo.integrate import TruncatedNormal, Integrator
from tqdm import tqdm
import statistics
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(exp((x-0.4)*10)+1)

def func_coinflip(prob, function):
    def func(x):
        if random.random() > prob(x):
            return 0
        else:
            return function(x)
    return func

def compute_quad_results(func, pbh, prob, n, params):
    """
    Computes quadrature integrals for all of the desired results
    :param func: Function to integrate (not including pbh)
    :param pbh: Probability function to use in coinflips
    :param prob: Probability density function
    :param n: Number of samples
    :param params: Parameters used in integration
    :returns: I, standard_var, standard_err, standard_vve, coin_var, coin_err, coin_vve
    """
    # Integration options
    opts = {'epsabs': 1e-10, 'epsrel': 1e-7, 'limit': 100}

    # Compute the desired integrand
    I = quad(lambda x: func(x) * pbh(x),
             params['lower'], params['upper'], **opts)[0]

    # Do the standard MC integration results first (no coin flip)
    # Compute moments for the standard integral
    g = lambda x: func(x) * pbh(x) / prob(x)
    moment_2 = quad(lambda x: g(x)**2 * prob(x),
                    params['lower'], params['upper'], **opts)[0]
    moment_3 = quad(lambda x: g(x)**3 * prob(x),
                    params['lower'], params['upper'], **opts)[0]
    moment_4 = quad(lambda x: g(x)**4 * prob(x),
                    params['lower'], params['upper'], **opts)[0]

    standard_var = moment_2 - I**2
    standard_err = sqrt(standard_var / n)
    standard_mu4 = moment_4 - 4*I*moment_3 + 6*I**2*moment_2 - 3*I**4
    standard_vve = standard_mu4/n - (n-3)/n/(n-1)*standard_var**2

    # Do the coin flip MC integration results next
    # Compute moments for the coinflip integral
    g = lambda x: func(x) / prob(x)
    moment_2 = quad(lambda x: g(x)**2 * pbh(x) * prob(x),
                    params['lower'], params['upper'], **opts)[0]
    moment_3 = quad(lambda x: g(x)**3 * pbh(x) * prob(x),
                    params['lower'], params['upper'], **opts)[0]
    moment_4 = quad(lambda x: g(x)**4 * pbh(x) * prob(x),
                    params['lower'], params['upper'], **opts)[0]

    coin_var = moment_2 - I**2
    coin_err = sqrt(coin_var / n)
    coin_mu4 = moment_4 - 4*I*moment_3 + 6*I**2*moment_2 - 3*I**4
    coin_vve = coin_mu4/n - (n-3)/n/(n-1)*coin_var**2

    # Compute the minimum possible variance
    lamda = quad(lambda x: func(x) * sqrt(pbh(x)),
                 params['lower'], params['upper'], **opts)[0] ** 2
    coin_minvar = lamda - I*I

    return (I, standard_var, standard_err, standard_vve,
            coin_var, coin_err, coin_vve, coin_minvar)

# Import the data
with open("data.dat") as f:
    data = f.readlines()
nu = []
peaksnu = []
for line in data[1:]:
    if len(line) == 0:
        continue
    nums = line.split(", ")
    nuval = float(nums[0])
    pnu = float(nums[1])
    nu.append(nuval)
    peaksnu.append(pnu)
nu = np.array(nu)
peaksnu = np.array(peaksnu)

# Parameters describing our probability density
# (these are fairly arbitrary)
params = {'lower': 0.0,
          'upper': 0.9,
          'x0': 0.38,
          'sigma': 0.32}

# Make an interpolator
dndnux = interp1d(nu, peaksnu, kind='linear')

# Construct a sampler
sampler = TruncatedNormal(params)
samples = 10
runs = 40000

# Obtain the quadrature results
(quad_result, _, _, _, quad_var, _,
 quad_vve, _) = compute_quad_results(dndnux, sigmoid, sampler.prob, samples, params)

# quad_result: integration result. Should be at center of histogram.
# coin_err: root of variance in integration result. Width of histogram.
# coin_var: variance. Should be at center of variance histogram.
# coin_vve: quadrature variance of variance estimator. Root of this should be
#           the width of the variance histogram.

# Set up the integrator
coinflip_int = Integrator(sampler, func_coinflip(sigmoid, dndnux))

# Set up data storage
# Note that we only care about the estimated integral and variance
# The other quantities are attempting to estimate the error on these
mc_results = np.zeros(runs)
mc_variances = np.zeros(runs)

# Perform all the runs
for i in tqdm(range(runs)):
    mc_results[i], mc_variances[i], _, _ = coinflip_int.integrate(n=samples)

# Compute some statistics from our runs
mean_results = statistics.mean(mc_results)
variance_results = statistics.variance(mc_results, mean_results)
mean_variance = statistics.mean(mc_variances)
variance_variance = statistics.variance(mc_variances, mean_variance)

print(f"Performed {runs} runs with {samples} samples each")
print(f"Quadrature mean: {quad_result}")
print(f"Mean over all results: {mean_results}")
print(f"Expected variance in mean: {quad_var / samples}")
print(f"Variance of means in results: {variance_results}")
print()
print(f"Quadrature variance: {quad_var}")
print(f"Variance over all results: {mean_variance}")
print(f"Expected variance in variance: {quad_vve}")
print(f"Variance of variance in results: {variance_variance}")

# Show histograms of means and variances
plt.rcParams["font.family"] = "serif"
fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].ticklabel_format(style='scientific', axis='both', scilimits=(1, 4))
axs[1].ticklabel_format(style='scientific', axis='both', scilimits=(1, 4))
axs[0].hist(mc_results, bins=25)
axs[1].hist(mc_variances, bins=25)
axs[0].set_xlabel('Integral value')
axs[0].set_ylabel('Count')
axs[1].set_xlabel('Sample variance')
axs[0].axvline(quad_result, color='r')
axs[1].axvline(quad_var, color='r')
axs[0].hlines(y=runs/30,
              xmin=quad_result-sqrt(quad_var/samples),
              xmax=quad_result+sqrt(quad_var/samples),
              color='r')
axs[1].hlines(y=runs/30,
              xmin=quad_var-sqrt(quad_vve),
              xmax=quad_var+sqrt(quad_vve),
              color='r')
plt.show()
