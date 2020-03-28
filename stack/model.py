"""
model.py

Contains the Model class, which stores the definitions of the given model
"""
from math import sqrt

class Model(object):

    def __init__(self, n_ef, n_fields, mpsi, m0):
        self.n_ef = n_ef
        self.n_fields = n_fields
        self.m0 = m0
        self.mpsi = mpsi
        # Compute derivative quantities
        self.mupsi2 = 3 - sqrt(9 - 4*mpsi**2)
        self.muphi2 = m0**2
        self.lamda = -3/2 + sqrt(9/4 + m0**2)
        self.beta = 1/(2*self.lamda)
