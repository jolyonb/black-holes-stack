"""
This is set up to test linear evolution
"""

import numpy as np
from stack.model import Model
from stack.linear.linearevolve import Linear
import matplotlib.pyplot as plt

# Set up an evenly spaced grid with gridpoints at 0, h/2, 3h/2, 5h/2, etc
rvals = np.linspace(0, 100, 100)
rvals += rvals[1]/2
rvals = np.concatenate((np.array([0]), rvals))

def test_phi(r):
    width = 10
    nu = 0.2
    return 1 - (1 - nu) * np.exp(-r**2/2/width**2)

# Compute a test function for phi
phi = test_phi(rvals)

model = Model(n_ef=15, n_fields=5, mpsi=0.5, m0=2)
lin = Linear(model=model, phi=phi, rvals=rvals)

lin.construct_grid()
# # We now have a grid, and we can now fix delta rho and delta rho dot on the grid for testing purposes
# n = 2
# lin.deltarho = np.sinc(n * lin.Agrid / 10)
lin.construct_spectral()

def plotrho(xi):
    tildeR, tildeU, tildem, tilderho = lin.full_evolution(xi)
    plt.plot(lin.Agrid, tilderho)
    plt.show()

def plotU(xi):
    tildeR, tildeU, tildem, tilderho = lin.full_evolution(xi)
    plt.plot(lin.Agrid, tildeU)
    plt.show()
