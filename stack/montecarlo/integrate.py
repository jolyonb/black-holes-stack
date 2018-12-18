"""
integrate.py

Code to perform Monte Carlo integration
"""
import statistics
import random
import numpy as np
from numpy import sqrt, exp, pi
from scipy.special import erf

class Sampler(object):
    """Abstract class describing methods to sample from a distribution"""

    def __init__(self, parameters=None):
        """
        Initialize the sampler
        :param parameters: Parameters used in constructing samples
        """
        self.parameters = parameters

    def gen_sample(self):
        """Generate a new sample. Returns (value, probability density at value)"""
        raise NotImplementedError()

def truncated_norm(x0, sigma, a, b):
    """
    Returns the integral of exp(-(x-x0)^2/2/sigma^2) from a to b
    :param x0: center of Gaussian
    :param sigma: standard deviation of Gaussian
    :param a: lower limit
    :param b: upper limit
    :return: integral result (float)
    """
    return sqrt(pi/2) * sigma * (erf((b-x0)/sqrt(2)/sigma) - erf((a-x0)/sqrt(2)/sigma))

class TruncatedNormal(Sampler):
    """Generates samples from a truncated normal distribution"""
    def __init__(self, parameters):
        """Initialize the parameters"""
        super().__init__(parameters)
        self.x0 = parameters["x0"]
        self.sigma = parameters["sigma"]
        self.sigma2 = self.sigma * self.sigma
        self.upper = parameters["upper"]
        self.lower = parameters["lower"]
        self.norm = truncated_norm(self.x0, self.sigma, self.lower, self.upper)

    def gen_sample(self):
        x = self.lower - 1
        while x < self.lower or x > self.upper:
            x = random.normalvariate(self.x0, self.sigma)
        return x, self.prob(x)

    def prob(self, x):
        return exp(-(x-self.x0)**2/2/self.sigma2) / self.norm

class Integrator(object):
    """MC integration class"""

    def __init__(self, sampler, func):
        """Store the sampler object and function to integrate"""
        self.sampler = sampler
        self.func = func

    def integrate(self, n=10000):
        """Perform the integration"""
        samples = np.zeros(n)

        for i in range(n):
            x, prob = self.sampler.gen_sample()
            f = self.func(x)
            samples[i] = f / prob

        # Compute statistics from the run
        # We use the statistics package for mean and sample variance
        # calculations, as it is more accurate than doing it ourselves
        est_mean = statistics.mean(samples)
        est_variance = statistics.variance(samples, est_mean)
        # Given the variance, this is the estimate of the error on the mean
        est_error = sqrt(est_variance/n)
        # We also want to estimate the error on the variance
        # We have the estimated mean and variance
        # To get the estimated variance of the estimated variance, we
        # need the fourth order moment
        centered = samples - est_mean
        fourth_moment = statistics.mean(centered**4)
        est_vve = n/(n-2)/(n-3)*fourth_moment - (n**2-3)/n/(n-2)/(n-3) * est_variance**2

        return est_mean, est_variance, est_error, est_vve
