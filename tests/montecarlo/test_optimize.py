"""
test_optimize.py

Tests optimization methods to find ranges and Gaussian parameters
"""
# Make sure that stack can be found!
import sys
sys.path.append('../../')

import numpy as np
from numpy import exp
from scipy.interpolate import interp1d
from stack.montecarlo.optimize import find_range, optimize_gaussian, fit_logistic, construct_logistic

def my_sigmoid(x):
    # x0 = -0.4, k = 10
    return 1/(1 + exp((x-0.4)*10))

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

# Make an interpolator
dndnu = interp1d(nu, peaksnu, kind='linear')

# Sample the sigmoid to imitate curve fitting to a sigmoid
xvals = np.linspace(nu[0], nu[-1], num=30)
yvals = my_sigmoid(xvals)
result = fit_logistic(xvals, yvals)
sigmoid = construct_logistic(result)
print(f"Sigmoid curve fitted: x0 = {result[0]}, k = {result[1]}")

# Set the range appropriately
min_x, max_x = find_range(sigmoid, dndnu, (nu[0], nu[-1]))
params = {'lower': min_x, 'upper': max_x}
print("Range:")
print(f"Lower: {params['lower']}")
print(f"Upper: {params['upper']}")

# Find the optimal Gaussian parameters
x0, sigma, var, min_var = optimize_gaussian(sigmoid, dndnu, params)
print("Gaussian fit:")
print(f"x0: {x0}")
print(f"sigma: {sigma}")
print(f"Variance: {var}")
print(f"Minimum possible variance: {min_var}")
print(f"Ratio: {var/min_var}")
