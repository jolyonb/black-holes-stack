"""
test_single.py

Performs an MC simulation using standard and coin flip methods,
and compares to quadrature
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

def sigmoid(x):
    return 1/(exp((x-0.4)*10)+1)

def func_coinflip(prob, function):
    def func(x):
        if random.random() > prob(x):
            return 0
        else:
            return function(x)
    return func

def compute_quad_results(func, pbh, prob, n, parameters):
    """
    Computes quadrature integrals for all of the desired results
    :param func: Function to integrate (not including pbh)
    :param pbh: Probability function to use in coinflips
    :param prob: Probability density function
    :param n: Number of samples
    :param parameters: Parameters used in integration
    :returns: I, standard_var, standard_err, standard_vve, coin_var, coin_err, coin_vve
    """
    # Integration options
    opts = {'epsabs': 1e-7, 'epsrel': 1e-7}

    # Compute the desired integrand
    integral = quad(lambda x: func(x) * pbh(x),
                    parameters['lower'], parameters['upper'], **opts)[0]

    # Do the standard MC integration results first (no coin flip)
    # Compute moments for the standard integral
    g = lambda x: func(x) * pbh(x) / prob(x)
    moment_2 = quad(lambda x: g(x)**2 * prob(x),
                    parameters['lower'], parameters['upper'], **opts)[0]
    moment_3 = quad(lambda x: g(x)**3 * prob(x),
                    parameters['lower'], parameters['upper'], **opts)[0]
    moment_4 = quad(lambda x: g(x)**4 * prob(x),
                    parameters['lower'], parameters['upper'], **opts)[0]

    standard_var = moment_2 - integral ** 2
    standard_err = sqrt(standard_var / n)
    standard_mu4 = moment_4 - 4 * integral * moment_3 + 6 * integral ** 2 * moment_2 - 3 * integral ** 4
    standard_vve = standard_mu4/n - (n-3)/n/(n-1)*standard_var**2

    print("Standard integration: quadrature results:")
    print(f"Integral: {integral}, Variance: {standard_var}, 4th moment: {standard_mu4}, evve: {standard_vve}")

    # Do the coin flip MC integration results next
    # Compute moments for the coinflip integral
    g = lambda x: func(x) / prob(x)
    moment_2 = quad(lambda x: g(x)**2 * pbh(x) * prob(x),
                    parameters['lower'], parameters['upper'], **opts)[0]
    moment_3 = quad(lambda x: g(x)**3 * pbh(x) * prob(x),
                    parameters['lower'], parameters['upper'], **opts)[0]
    moment_4 = quad(lambda x: g(x)**4 * pbh(x) * prob(x),
                    parameters['lower'], parameters['upper'], **opts)[0]

    coin_var = moment_2 - integral ** 2
    coin_err = sqrt(coin_var / n)
    coin_mu4 = moment_4 - 4 * integral * moment_3 + 6 * integral ** 2 * moment_2 - 3 * integral ** 4
    coin_vve = coin_mu4/n - (n-3)/n/(n-1)*coin_var**2

    print("Coinflip integration: quadrature results:")
    print(f"Integral: {integral}, Variance: {coin_var}, 4th moment: {coin_mu4}, evve: {coin_vve}")

    # Compute the minimum possible variance
    lamda = quad(lambda x: func(x) * sqrt(pbh(x)),
                 parameters['lower'], parameters['upper'], **opts)[0]
    coin_minvar = lamda * lamda - integral * integral

    return (integral, standard_var, standard_err, standard_vve,
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
params = {'lower': 0.0507,
          'upper': 0.981,
          'x0': 0.497,
          'sigma': 0.22}

# Make an interpolator
dndnux = interp1d(nu, peaksnu, kind='linear')

# Construct a sampler
sampler = TruncatedNormal(params)
num_samples = 10000

# Obtain the quadrature results
(quad_result, s_var, s_err, s_vve, c_var, c_err,
 c_vve, c_minvar) = compute_quad_results(dndnux, sigmoid, sampler.prob, num_samples, params)

# Test 1: Standard Monte Carlo integration of integral
standard_int = Integrator(sampler, lambda x: sigmoid(x) * dndnux(x))
mc_result, mc_var, mc_error, mc_vve = standard_int.integrate(n=num_samples)

# Test 2: Monte Carlo integration with "coin flip" method
coinflip_int = Integrator(sampler, func_coinflip(sigmoid, dndnux))
mccf_result, mccf_var, mccf_error, mccf_vve = coinflip_int.integrate(n=num_samples)

# Compare results
print()

print("Test 1 results: Standard MC integration vs quadrature")
print("Integral:")
print(f"MC evaluation: {mc_result}")
print(f"Error estimate: {mc_error}")
print(f"Expected error: {s_err}")
print(f"Actual error: {abs(quad_result - mc_result)}")
print("Variance:")
print(f"MC evaluation: {mc_var}")
print(f"Error estimate: {sqrt(mc_vve)}")
print(f"Expected error: {sqrt(s_vve)}")
print(f"Actual error: {abs(s_var - mc_var)}")

print()

print("Test 2 results: Coinflip MC integration vs quadrature")
print("Integral:")
print(f"MC evaluation: {mccf_result}")
print(f"Error estimate: {mccf_error}")
print(f"Expected error: {c_err}")
print(f"Actual error: {abs(quad_result - mccf_result)}")
print("Variance:")
print(f"MC evaluation: {mccf_var}")
print(f"Error estimate: {sqrt(mccf_vve)}")
print(f"Expected error: {sqrt(c_vve)}")
print(f"Actual error: {abs(c_var - mccf_var)}")

print()

print(f"Minimum variance: {c_minvar}, Ratio: {c_var/c_minvar}")
