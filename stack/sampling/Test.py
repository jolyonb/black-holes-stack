#!/usr/bin/python
from Covariance import Covariance
from Sampler import Sampler
import numpy as np
import pickle
from matplotlib import pyplot as plt

""" 
A testing script which provides a sample implementation
of the Covariance and Sampler classes.
"""

#===============Covariance Options==============#
#Recompute covariance? (set this to True the first time around!)
compute_cov = True 
#Spectrum filename
spectrum_filename = "dummy"
#Upper and lower bounds in frequency-space.
k_low = 0.001
k_high = 5.0
#Cutoff harmonic mode.
l_max = 10
#Maximum radius and number of radial samples (uniform grid).
r_max = 10.0
r_step = r_max/10

#=================Sampler Options===============#
nubar = 1.0
nfields = 8
nsamples = 10 

#===============Compute Covariance==============#
cov = Covariance()
if compute_cov==True:
	cov.ComputeCovariances(spectrum_filename,k_low,k_high,l_max,r_max,r_step)
	cov.Save("covariances")

#===============Sample 00 Phi Mode==============#
cov.Load("covariances.npz")
samp = Sampler(cov)
print("===============Sampling Phi00==============")
samps = samp.GetSamples(nubar,nfields,nsamples)

#==============Compute mean w/droop=============#
print("===================Computing Mean==================")
#Recompute the rho splines.
#Ideally we would store the splines, but we can't
#store arbitrary objects in the .npz file, and 
#pickling splines is a headache.
cov.ComputeRhoCSpline()
cov.ComputeRhoDSpline()
npoints = len(cov.rgrid)
droopy_mean = np.zeros(npoints)
for l in range(0,l_max+1):
	cl_vec = np.asarray([cov.ComputeRhoCLElement(l,r,r) for r in cov.rgrid])
	cl_vec*=(2*l+1)
	droopy_mean+=cl_vec
droopy_mean*=nfields
rhoc_vec = np.square(np.asarray([cov.rho_c_spline(r) for r in cov.rgrid]))
rhoc_vec*=(nubar**2 - nfields)
droopy_mean+=rhoc_vec
rhod_vec = np.square(np.asarray([cov.rho_d_spline(r) for r in cov.rgrid]))
droopy_mean-=rhod_vec
droopy_mean*=np.sqrt(4.0*np.pi)*cov.sigma0**2

#==============Compute variance w/droop=============#
print("===================Computing Mean==================")
npoints = len(cov.rgrid)
droopy_variance = np.zeros(npoints)
rhoc_vec = np.asarray([cov.rho_c_spline(r) for r in cov.rgrid])
rhod_vec = np.asarray([cov.rho_d_spline(r) for r in cov.rgrid])
for l in range(0,l_max+1):
	cl_vec = np.square(np.asarray([cov.ComputeRhoCLElement(l,r,r) for r in cov.rgrid]))
	cl_vec*=(2*l+1)
	droopy_variance+=cl_vec
droopy_variance*=2.0*nfields

T2 = 4.0*(nubar**2-nfields)*np.square(rhoc_vec)
T2 *= np.asarray([cov.ComputeRhoCLElement(0,r,r) for r in cov.rgrid])
droopy_variance+=T2

T3 = -4.0*np.square(rhod_vec)
T3*= np.asarray([cov.ComputeRhoCLElement(1,r,r) for r in cov.rgrid])
droopy_variance+=T3

T4 = 2.0*(nfields-2.0*nubar**2)*np.power(rhoc_vec,4)
droopy_variance+=T4

T5 = (2.0/3.0)*np.power(rhod_vec,4)
droopy_variance+=T5

droopy_variance*=4.0*np.pi*cov.sigma0**4

#==================Save results!====================#
np.savez("samples.npz",samples = samps, grid=cov.rgrid, droopmean = droopy_mean, droopvar = droopy_variance, nsamples=nsamples)
