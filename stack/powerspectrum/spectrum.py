#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Power spectrum of waterfall field perturbations in hybrid inflation
v1.0 by Jolyon Bloomfield, April 2017
See arXiv:xxxx.xxxxx for details
"""
# Note that this file can be used as a library or stand-alone

from __future__ import division, print_function

import numpy as np
from math import exp, log, pi, sqrt, log10
from scipy.integrate import ode, quad
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt

class PowerSpectrum(object) :
    """
    Computes the power spectrum for hybrid inflation theories
    """

    def __init__(self) :
        # Initialize spectrum
        self.spectrum = np.array([])
        self.kvals = np.array([])
        self.has_spectrum = False
        self.has_params = False

    def set_params(self, m_0, m_psi, Nstart=6.0, Nend=15.0) :
        """
        Sets parameters for the theory.

        m_0 is the dimensionless waterfall field mass (divided by H)
        m_psi is the dimensionless timer field mass (divided by H)

        psi_c, the number of fields, and H aren't needed

        Nstart is the number of efolds before the waterfall transition
        to start evolving fields from (this is modified by k internally)

        Nend is the number of efolds after the waterfall transition
        to stop evolving at.
        """
        # Compute mu_phi^2 and mu_psi^2
        self.muphi2 = m_0*m_0
        self.mupsi2 = 3 - sqrt(9 - 4 * m_psi*m_psi)

        # Store parameters
        self.Nend = Nend
        self.Nstart = Nstart

        self.has_params = True

    def compute_mode(self, k) :
        """Compute the mode function for a given k"""
        if self.has_params == False :
            print("Error: Parameters not set.")
            return

        def derivs(N, deltaval, k) :
            # Extract values
            delta, deltadot = deltaval
            # Compute the second derivative of delta
            deltaddot = - (deltadot*deltadot + deltadot - 2 + k*k*exp(-2*N)*(1-exp(-4*delta)) + self.muphi2*(exp(-self.mupsi2*N)-1))
            # Return results
            return [deltadot, deltaddot]

        # Compute the start time, adjusted for k
        startN = -self.Nstart + log(k * k / self.muphi2) / (2.0 - self.mupsi2)

        # Compute quantities for initial conditions
        expn = exp(startN)
        expnon2k2 = expn / (2 * k * k)
        correction = expnon2k2 * (1 + self.muphi2 / 2 * (1 - exp(-self.mupsi2*startN)))

        # Set up the initial conditions
        R0 = 1.0 / expn + correction
        Rp0 = - 1.0 / expn + correction + self.muphi2 * self.mupsi2 / 2 * expnon2k2 * exp(-self.mupsi2*startN)
        delta0 = np.array([log(R0) + startN, Rp0/R0 + 1])

        # Set up the integrator
        r = ode(derivs).set_integrator('vode', nsteps=1000000, method="bdf", rtol=1e-13, atol=1e-10)
        # Store the initial value and the parameter k
        r.set_initial_value(delta0, startN).set_f_params(k)

        # Storage arrays for results
        deltavals = np.array([delta0[0]])
        times = np.array([startN])

        # Integrate!
        while r.successful() and r.t < self.Nend :
            newtime = r.t + 0.05
            if newtime > self.Nend :
                newtime = self.Nend
            result = r.integrate(newtime)
            # Store results (we don't care about the derivative)
            deltavals = np.append(deltavals, result[0])
            times = np.append(times, newtime)

        # We've computed delta(N). Now compute r(N), R(N) and the power spectrum
        rvals = deltavals - times
        Rvals = np.exp(rvals)
        Pk = Rvals[-1] * Rvals[-1] / (2*pi)**3 / 2 / k

        return times, deltavals, rvals, Rvals, Pk

    def compute_spectrum(self, startk, endk, samples, logarithmic=True, verbose=True) :
        """
        Compute the power spectrum
        startk and endk give the start and end values of k to sample
        samples is the number of k values to sample
        logarithmic=True uses log spacing, False uses linear spacing
        """
        if self.has_params == False :
            print("Error: Parameters not set.")
            return

        if logarithmic :
            self.kvals = np.logspace(log10(startk), log10(endk), num=samples)
        else :
            self.kvals = np.linspace(startk, endk, samples)

        self.spectrum = np.zeros(samples)

        for n, k in enumerate(self.kvals) :
            if verbose :
                print("Computing spectrum:", n + 1, "of", samples)
            _, _, _, _, Pk = self.compute_mode(k)
            self.spectrum[n] = Pk

        # Include k=0
        if self.kvals[0] != 0 :
            self.kvals = np.append(np.array([0.0]), self.kvals)
            self.spectrum = np.append(np.array([self.spectrum[0]]), self.spectrum)

        # Flag that we have a spectrum
        self.has_spectrum = True

    def plot_spectrum(self, logk=True, logP=False, k2=False) :
        """Plot the power spectrum as a function of k"""
        if self.has_spectrum == False :
            print("Have not yet computed/loaded spectrum")
            return

        if logk :
            xaxis = np.log10(self.kvals)
            xlabel = "log10(k)"
        else :
            xaxis = self.kvals
            xlabel = "k"

        if k2 :
            yaxis = self.spectrum * self.kvals * self.kvals
            ylabel = "k^2 P(k)"
        else :
            yaxis = self.spectrum
            ylabel = "P(k)"

        if logP :
            yaxis = np.log10(yaxis)
            ylabel = "log10(" + ylabel + ")"

        plt.plot(xaxis, yaxis, 'b')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()
        plt.close()

    @staticmethod
    def plot_mode(nvals, modevals, logarithmic=False) :
        """Plot a mode as a function of n"""
        if logarithmic :
            plt.plot(nvals, np.log10(modevals), 'b')
            ylabel = "log10(mode(N))"
        else :
            plt.plot(nvals, modevals, 'b')
            ylabel = "mode(N)"
        plt.xlabel('N')
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()
        plt.close()

    def scale_spectrum(self, scale) :
        """Scales spectrum by dividing by scale"""
        if self.has_spectrum == False :
            print("Have not yet computed/loaded spectrum")
            return

        self.spectrum /= scale

    def save_spectrum(self, filename) :
        """Save the spectrum to a file"""
        if self.has_spectrum == False :
            print("Spectrum hasn't been computed yet!")
            return
        data = np.array([self.kvals, self.spectrum])
        np.savetxt(filename, data.transpose(), delimiter=",")

    def load_spectrum(self, filename) :
        """Load a spectrum from a file"""
        self.kvals, self.spectrum = np.loadtxt(filename, delimiter=",").transpose()
        self.has_spectrum = True

    def compute_power(self) :
        """Compute the total power in the power spectrum"""
        if self.has_spectrum == False :
            print("Spectrum hasn't been computed yet!")
            return

        # Get a spline of the data
        spl, maxk = self.get_spline(2)

        # Compute the power by integrating the spline
        power = spl.integral(0.0, maxk)

        # Return the result
        return power

    def get_spline(self, kpower=0) :
        """
        Compute a spline of the power spectrum multiplied by k^kpower
        Returns spline, maxk
        (mink = 0)
        """
        if self.has_spectrum == False :
            print("Have not yet computed/loaded spectrum")
            return

        kvals = self.kvals
        spectrum = self.spectrum

        # Make sure that we include k=0
        if kvals[0] != 0 :
            kvals = np.append(np.array([0.0]), kvals)
            spectrum = np.append(np.array([spectrum[0]]), spectrum)

        if kpower > 0 :
            data = spectrum * (kvals ** kpower)
        else :
            data = spectrum

        # Construct a spline of the data
        spl = InterpolatedUnivariateSpline(kvals, data)

        return spl, kvals[-1]

# Only use if running this file directly (not as a library)
if __name__ == "__main__":
    #
    # Deal with command line arguments
    #
    import argparse
    parser = argparse.ArgumentParser(description="Compute the power spectrum of waterfall fields in hybrid inflation")

    # Helper functions to demand valid ranges on variables from command line arguments
    def is_float(str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    def is_pos_int(str):
        if str.isdigit() :
            return True
        return False

    def check_samples(value):
        if not is_float(value) :
            raise argparse.ArgumentTypeError("must be a number. You supplied: %s" % value)
        ivalue = float(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError("must be at least 1. You supplied: %s" % value)
        return int(ivalue)

    def pos_float(value):
        if not is_float(value) :
            raise argparse.ArgumentTypeError("must be a number. You supplied: %s" % value)
        ivalue = float(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError("must be > 0. You supplied: %s" % value)
        return ivalue

    def pos_int(value):
        if not is_pos_int(value) :
            raise argparse.ArgumentTypeError("must be a positive integer. You supplied: %s" % value)
        return int(value)

    # Required arguments
    # Output file
    parser.add_argument(help="Output file (e.g. 'spectrum.dat')", dest="filename")

    # Optional arguments
    # Number of samples
    parser.add_argument("-s", help="Number of samples (default 100)",
        default=100, type=check_samples, dest="samples")

    # Values for scanning over k
    parser.add_argument("-k", help="Range for k (default 1e-3 1e3)",
        default=[1e-3, 1e3], type=pos_float, dest="k_range", nargs=2, metavar=('MIN', 'MAX'))

    # Values for m_0 and m_psi
    parser.add_argument("-m0", help="Value for m_0 (default 10.0)",
        default=10.0, type=pos_float, dest="m_0")
    parser.add_argument("-mpsi", help="Value for m_psi (default 0.1)",
        default=0.1, type=pos_float, dest="m_psi")

    # Log spacing or Linear spacing?
    spacing_group = parser.add_mutually_exclusive_group()
    spacing_group.add_argument("-log", help="Logarithmic spacing in k (default)", action="store_true", dest="log")
    spacing_group.add_argument("-lin", help="Linear spacing in k", action="store_false", dest="log")
    parser.set_defaults(log=True)

    # Quiet mode? Verbose mode?
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument("-q", help="Quiet mode", action="store_false", dest="verbose")
    parser.set_defaults(verbose=True)

    # Parse the command line
    args = parser.parse_args()
    # Print the header
    if args.verbose : print(__doc__)


    #
    # Computations start here
    # Compute a power spectrum and save to file
    #

    # Initialize our object
    ps = PowerSpectrum()
    ps.set_params(args.m_0, args.m_psi)

    # Compute the power spectrum
    ps.compute_spectrum(args.k_range[0], args.k_range[1], args.samples, args.log, args.verbose)

    # Get the total power
    power = ps.compute_power()

    # Normalize the power spectrum by the total power
    ps.scale_spectrum(4*pi*power)

    # Output to file
    ps.save_spectrum(args.filename)
    print("Power spectrum written to", args.filename)
