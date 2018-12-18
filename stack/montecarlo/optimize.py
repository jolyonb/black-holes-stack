"""
optimize.py

Used to optimize the choice of Gaussian used for sampling
"""
import numpy as np
from numpy import sqrt, exp
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize, brentq, curve_fit
from stack.montecarlo.integrate import truncated_norm

np.seterr(all='raise')

# Integration options
opts = {'epsabs': 1e-7, 'epsrel': 1e-7, 'limit': 100}

def find_range(pbh, dndnu, bounds, threshold_min=0.005, threshold_max=0.005):
    """
    Given pbh and dndnu, computes an appropriate range to sample
    from in order to determine black hole formation
    Works within given bounds (min, max). Note that functions should
    evaluate within the bounds!
    """
    # Start by integrating pbh*dndnu over the bounds
    # This is the quantity that we want to approximate
    full = quad(lambda x: pbh(x) * dndnu(x), bounds[0], bounds[1], **opts)[0]

    # We want to cut off half the threshold from the tail at each end
    def lower(minx):
        return quad(lambda x: pbh(x) * dndnu(x), bounds[0], minx, **opts)[0] / full - threshold_min/2

    def upper(maxx):
        return quad(lambda x: pbh(x) * dndnu(x), maxx, bounds[1], **opts)[0] / full - threshold_max/2

    min_x = brentq(lower, *bounds)
    max_x = brentq(upper, *bounds)

    return min_x, max_x

def fit_logistic(datax, datay):
    """
    Given datapoints at x and y, construct a logistic function to approximate the data
    The function that is fit is the following:
    1 / (1 + exp(-k * (x - x0)))
    Returns x0, k (numpy array)
    """
    # Here is the sigmoid that we want to fit
    def sigmoid(x, x0, k):
        return 1 / (1 + np.exp(k * (x - x0)))

    # popt is the optimal parameters, pcov is the estimated covariance
    popt, pcov = curve_fit(sigmoid, datax, datay)
    return popt

def construct_logistic(popt):
    """Constructs a logistic function based on popt from fit_logistic"""
    x0, k = popt

    def sigmoid(x):
        return 1 / (1 + np.exp(k * (x - x0)))

    return sigmoid

def optimize_gaussian(pbh, dndnu, parameters):
    """
    Pass in two functions: pbh(nu) and dndnu(nu)
    Computes the Gaussian parameters x0 and sigma to minimize
    the variance, the resulting variance, and the optimal variance.
    """
    # Compute the minimum possible variance
    result = quad(lambda x: dndnu(x) * pbh(x),
                  parameters['lower'], parameters['upper'], **opts)[0]
    lamda = quad(lambda x: dndnu(x) * sqrt(pbh(x)),
                 parameters['lower'], parameters['upper'], **opts)[0]
    min_var = lamda * lamda - result * result

    # Gaussian PDF
    def gaussian(x, x0, sigma):
        norm = truncated_norm(x0, sigma, parameters['lower'], parameters['upper'])
        return exp(-(x-x0)**2/2/sigma**2) / norm

    # Set up the function to optimize
    def cost_function(values):
        x0, sigma = values
        return quad(lambda x: dndnu(x)**2 * pbh(x) / gaussian(x, x0, sigma),
                    parameters['lower'], parameters['upper'], **opts)[0]

    # Find the maximum of pbh(x) * dndnu(x) for an estimate of x0
    optx = minimize_scalar(lambda x: -pbh(x) * dndnu(x),
                           bounds=(parameters['lower'], parameters['upper']),
                           method='bounded')

    # Estimate the width of the Gaussian as half the range
    initial = np.array([optx.x, (parameters['upper'] - parameters['lower']) / 2])

    # Perform the optimization
    opt = minimize(cost_function, initial, tol=1e-6, method='Nelder-Mead')
    new_x0, new_sigma = opt.x

    # Compute the variance with this optimization
    var = cost_function(opt.x) - result * result

    return new_x0, new_sigma, var, min_var
