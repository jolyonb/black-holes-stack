#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots a power spectrum, using spectrum.py
"""

from spectrum import *

ps = PowerSpectrum()
ps.load_spectrum("output.dat")
# Plot the spectrum (arguments refer to log scales in X and Y axes, last is whether or not we multiply by k^2)
ps.plot_spectrum(True, False, True)
