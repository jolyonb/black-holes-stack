import numpy as np
from scipy.fftpack import dst

maxA = 1
gridpoints = 12
grid = (np.array(list(range(gridpoints))) + 0.5) / gridpoints
Agrid = maxA * grid

fact = 1.0 * np.array(list(range(1, gridpoints + 1)))
fact[-1] /= 2

coeff = np.pi / gridpoints / maxA

n = 1
spatial = 2.7*np.sinc(n * Agrid / maxA) + 3.5 * np.sinc(3 * n * Agrid / maxA)
spectral = dst(spatial * Agrid, type=2) * coeff * fact

spatial2 = dst(spectral / fact, type=3) / Agrid / coeff / 2 / gridpoints

# construct the spatial representation manually
kngrid = 1.0 * np.array(list(range(1, gridpoints + 1))) / maxA
spatial3 = np.zeros_like(spatial)
for i in range(gridpoints):
    spatial3[i] = np.sum(spectral * np.sinc(kngrid * Agrid[i]))

spatial4 = np.zeros_like(spatial)
for i in range(gridpoints):
    spatial4[i] = np.sinc(kngrid[n-1] * Agrid[i])

print(spectral)
