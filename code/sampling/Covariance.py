#! /usr/bin/python

import numpy as np
from numpy import linalg as la
from scipy.interpolate import interp1d
from scipy import optimize
from scipy import integrate
from scipy import special
from scipy import sqrt
import Levin
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Todo/notes:
 In the paper, we define the sigma integrals as occurring over the full power spectrum,
	but apply k cutoffs to the C and D integrals. If this is true, then should we just 
	compute the sigma integrals over the full range of k given to us by the data we load
	up? 
"""

class Covariance(object):

	def __init__(self):
		"""Instantiate a levin integrator and set some default parameters."""
		#instantiate levin integrator
		self.levintegrator = Levin.SphericalBesselIntegrator()
		#FIXME: don't hardcode spline resolution. How to choose a good value?
		self.spectrum_spline_resolution = 100000
		#Set hi-resolution grid-spacing overcounting factor for computing the C
		#and D splines.
		#FIXME: what's sufficient?
		self.hires_factor = 10
		#Initialize rho_c_hires and rho_d_hires to None, to prevent recomputation
		#from the Sampler object.
		self.rho_c_hires = None
		self.rho_d_hires = None

	def ComputeCovariances(self,fname_spectrum, k_low, k_high, l_max, r_max, r_step):
		"""
		Takes the name of a power spectrum
		data file [fname_spectrum], the lower frequency bound [k_low], upper 
		frequency bound [k_high], maximum radius [r_max], and maximum harmonic mode 
		[l_max]. Prepares a radial grid with resolution determined by either [r_step]
		or, if no r_step value is given, the nyquist frequency associated to k_high, 
		computes splines for RhoC and RhoD, and then computes covariance matrices for 
		the biased fields at relevant field index and harmonic mode. Next, the square 
		roots of these matrices are computed, after zeroing out small negative eigenvalues 
		from numerical errors. These matrices are accessed through the 
		GetSqrtCovariance(l,alpha) function. We achieve persistence of the covariance 
		matrices by constructing the object before sampling, storing to disk using Save(), 
		and then loading it using Load() and calling GetSqrtCovariance() from the sampling 
		code. 
		"""
		self.k_low = k_low
		#Option to compute frequency cutoff with fraction of power below.
		#self.k_high = self.ComputeFrequencyCutoff(0.95)
		self.k_high = k_high

		self.fname_spectrum = fname_spectrum
		#self.LoadPowerSpectrum(fname_spectrum)
		#FIXME
		print "Loading Power Spectrum"
		self.LoadDummyPowerSpectrum()

		self.sigma0 = self.ComputeSigmaN(0)
		self.sigma1 = self.ComputeSigmaN(1)
		self.r_max = r_max
		#Should have option to compute value according to paper.	
		self.l_max = l_max
		self.wavelength = 2.0*np.pi/self.k_high
		if r_step==None:
			#Setting r_step to nyquist frequency.
			self.r_step = self.wavelength/2.0
		else:
			self.r_step = r_step 
		#Define uniform radial grid (+1 to include both endpoints!).
		self.gridpoints = int(np.ceil(self.r_max/self.r_step))+1
		print "Gridpoints: ",self.gridpoints
		self.rgrid = np.linspace(0.0,self.r_max,num=self.gridpoints,endpoint=True)
		
		#Compute RhoC and RhoD splines.
		print "Computing RhoC Spline"
		self.ComputeRhoCSpline()
		#Compute RhoC over the rgrid for later usage in Sampler (to compute means).
		self.rho_c_vec = np.asarray([self.rho_c_spline(r) for r in self.rgrid])
		print "Computing RhoD Spline"
		self.ComputeRhoDSpline()
		#Compute covariance matrices.
		print "Computing Covariance Matrices"
		self.ComputeCovarianceMatrices()
		#Compute square roots of covariance matrices.
		print "Computing Sqrt Covariance Matrices"
		self.ComputeSqrtCovariances()

	def LoadDummyPowerSpectrum(self):
		"""
		Load the power spectrum from (x,y) value pair file.
		"""
		#FIXME: will this be k^2*P(k) or just P(k)?
		#Dummy power spectrum, replace with proper loading!
		kvec = np.linspace(self.k_low,self.k_high,num=self.spectrum_spline_resolution,endpoint=True)
		yvec = np.exp(-kvec)
		self.spectrum = interp1d(kvec,yvec,kind='cubic')

	#def LoadPowerSpectrum():
		#FIXME: implement actual spectrum loading!
	
	def ComputeFrequencyCutoff(self,power_fraction):
		"""
		Computes a cutoff frequency bounding [power_fraction] of the total power
		from above. Takes a float [power_fraction] which lies in [0,1].
		"""
		def func(z,mypower_fraction):
			x_low = self.spectrum.x[0]
			x_high = self.spectrum.x[-1]
			(total_power,perr) = integrate.quad(self.spectrum,x_low,x_high)
			(frac_power,err) = integrate.quad(self.spectrum,x_low,z)
			result = frac_power - total_power * mypower_fraction
			return result
		x_low = self.spectrum.x[0]
		x_high = self.spectrum.x[-1]
		root = optimize.brentq(func,x_low,x_high,args=(power_fraction))
		return root
	
	def ComputeSigmaN(self,n):
		""" Returns the n^th moment of the power spectrum. """
		def func(k,m):
			return k**(2+2*m) * self.spectrum(k)
		#FIXME: using the full available spectrum k range???
		(integral,err) = integrate.quad(func,self.k_low,self.k_high,args=(n))
		return sqrt(4*np.pi*integral)

	def ComputeRhoC(self,r):
		""" Returns RhoC evaluated at radius [r] using the levin integrator."""
		def func(x):
			return x**2 * self.spectrum(x)
			#For high-precision checks against mathematica, don't use the spline!
			#instead, use the function itself:
			#return x**2 * np.exp(-x) 
		#Using the frequency cutoffs specified in the constructor.
		integral = self.levintegrator.ICalc(self.k_low,self.k_high,r,0,func)
		return 4*np.pi*integral/(self.sigma0**2)

	def ComputeRhoD(self,r):
		""" Returns RhoD evaluated at radius [r] using the levin integrator."""
		def func(x):
			return x**3 * self.spectrum(x)
			#For high-precision checks against mathematica, don't use the spline!
			#instead, use the function itself:
			#return x**3 * np.exp(-x) 
		#Using the frequency cutoffs specified in the constructor.
		integral = self.levintegrator.ICalc(self.k_low,self.k_high,r,1,func)
		return sqrt(3.0)*4*np.pi*integral/(self.sigma0*self.sigma1)

	def ComputeRhoCSpline(self):
		""" 
		Computes a high-resolution discretization of RhoC over [0,2*r_max],
		if the variable self.rho_c_hires is NoneType (not yet computed). 
		Fits a spline to this discretization.
		"""
		#Interpolate out to 2*r_max anticipatiting the C_l integral calculation.
		npoints = (2*self.r_max)/(self.r_step/self.hires_factor)+1
		self.rho_c_grid = np.linspace(0.0,2.0*self.r_max,num=npoints,endpoint=True)
		#Only compute high resolution discretization if not already computed.
		if self.rho_c_hires==None:
			ylist = []
			for r in tqdm(self.rho_c_grid):
				ylist.append(self.ComputeRhoC(r))
			self.rho_c_hires = np.asarray(ylist)
		self.rho_c_spline = interp1d(self.rho_c_grid,self.rho_c_hires,kind='cubic')

	def ComputeRhoDSpline(self):
		""" 
		Computes a high-resolution discretization of RhoD over [0,2*r_max],
		if the variable self.rho_d_hires is NoneType (not yet computed). 
		Fits a spline to this discretization.
		"""
		npoints = self.r_max/(self.r_step/self.hires_factor)+1
		self.rho_d_grid = np.linspace(0.0,self.r_max,num=npoints,endpoint=True)
		#Only compute high resolution discretization if not already computed.
		if self.rho_d_hires==None:
			ylist = []
			for r in tqdm(self.rho_d_grid):
				ylist.append(self.ComputeRhoD(r))
			self.rho_d_hires = np.asarray(ylist)
		self.rho_d_spline = interp1d(self.rho_d_grid,self.rho_d_hires,kind='cubic')

	def ComputeRhoCLElement(self,l,r1,r2):
		""" Returns RhoC at [r1],[r2] pair and mode [l] computed using the RhoC spline."""
		def func(x,l,r1,r2):
			return 0.5*special.eval_legendre(l,x)*self.rho_c_spline(np.sqrt(r1**2+r2**2-2.0*r1*r2*x))
		(integral,err) = integrate.quad(func,-1.0,1.0,args=(l,r1,r2))
		return integral
	
	def ComputeCovarianceMatrices(self):
		""" Computes biased covariance matrices over the radial grid. """
		self.Covariances_lneq1 = np.empty([self.l_max,self.gridpoints,self.gridpoints])
		self.Covariances_leq1 = np.empty([2,self.gridpoints,self.gridpoints]) 
		#Indexed by (l,alpha), where alpha=0 for generic alpha,
		#and alpha=1 indicates that the field index is actually one.
		#Finally, alpha=2 indicates that the field index is greater than one.

		#l=0,alpha generic
		cov = np.empty([self.gridpoints,self.gridpoints])
		l=0
		for n1,r1 in enumerate(self.rgrid):
			for n2,r2 in enumerate(self.rgrid):
				self.Covariances_lneq1[0,n1,n2] = (4.0*np.pi)*(self.ComputeRhoCLElement(l,r1,r2) - self.rho_c_spline(r1)*self.rho_c_spline(r2))
		#l=1,alpha=1
		l=1
		for n1,r1 in enumerate(self.rgrid):
			for n2,r2 in enumerate(self.rgrid):
				self.Covariances_leq1[0,n1,n2] = (4.0*np.pi)*(self.ComputeRhoCLElement(l,r1,r2) - (1.0/3.0)*self.rho_d_spline(r1)*self.rho_d_spline(r2))
		#l=1,alpha>1
		l=1
		for n1,r1 in enumerate(self.rgrid):
			for n2,r2 in enumerate(self.rgrid):
				self.Covariances_leq1[1,n1,n2] = (4.0*np.pi)*(self.ComputeRhoCLElement(l,r1,r2))
		#l>1,alpha generic
		for l in tqdm(range(2,self.l_max+1)):
			for n1,r1 in enumerate(self.rgrid):
				for n2,r2 in enumerate(self.rgrid):
					self.Covariances_lneq1[l-1,n1,n2] = (4.0*np.pi)*(self.ComputeRhoCLElement(l,r1,r2))

	def ScrubNegativeEigenvalues(self,eigvals):
		""" Scrubs small negative numbers out of the [eigvals] array. Returns scrubbed array. """
		#FIXME: Should we have some sort of test to make sure we aren't 
		#generating large negative eigenvalues (some sort of error), and
		#just throwing them out, effectively silencing the error?
		#Any magnitude cutoff seems kind of arbitrary, but maybe since our 
		#precision target is roughly one in a thousand, I'll choose 1e-6, and
		#say that three orders of magnitude below our precision limit is irrelevant
		#and probably numerical.
		clean_eigvals = np.empty(len(eigvals))
		for n,eigval in enumerate(eigvals):
			if eigval<0.0:
				assert np.abs(eigval)<1e-6
				clean_eigvals[n] = 0.0
			else:
				clean_eigvals[n] = eigval
		return clean_eigvals

	def ComputeSqrtMatrix(self,mat):
		""" Compute (and return) the square root A of a positive definite matrix [mat]. """
		L,M = la.eig(mat)
		cleanL = self.ScrubNegativeEigenvalues(L)
		sqrtL = np.diag(np.sqrt(cleanL))
		A = np.dot(M,sqrtL)
		return A

	def ComputeSqrtCovariances(self):
		""" Computes the square roots of each biased covariance matrix. """
		self.sqrtcov_lneq1 = np.empty([self.l_max,self.gridpoints,self.gridpoints])
		self.sqrtcov_leq1 = np.empty([2,self.gridpoints,self.gridpoints])
		self.sqrtcov_lneq1[0,:,:] = self.ComputeSqrtMatrix(self.Covariances_lneq1[0,:,:])
		self.sqrtcov_leq1[0,:,:] = self.ComputeSqrtMatrix(self.Covariances_leq1[0,:,:])
		self.sqrtcov_leq1[1,:,:] = self.ComputeSqrtMatrix(self.Covariances_leq1[1,:,:])
		for l in tqdm(range(2,self.l_max+1)):
			self.sqrtcov_lneq1[l-1,:,:] = self.ComputeSqrtMatrix(self.Covariances_lneq1[l-1,:,:])
	

	def GetSqrtCovariance(self,l,alpha):
		"""
		A function to retrieve sqrt-covariance matrices at spherical harmonic mode [l]
		and field index [alpha]. Returns a 2D nparray with dimensions gridpoints^2.
		"""
		if l==0:
			return self.sqrtcov_lneq1[0,:,:]
		if l==1:
			if alpha==1:
				return self.sqrtcov_leq1[0,:,:]
			else:
				return self.sqrtcov_leq1[1,:,:]
		else:
			return self.sqrtcov_lneq1[l-1,:,:]


	def Save(self,fname):
		"""
		Save all object data useful to Sampler as numpy arrays to a .npz file.
		Note: some of the arrays will not be used in sampling, but will be useful
		for sanity checks and debugging. In any case, we expect the bulk of the
		file size to be consumed by the sqrt-covariance matrices, as these scale
		with the square of the number of radial grid samples, whereas auxiliary 
		arrays scale at most linearly.
		"""
		np.savez(fname, sqrtcov_lneq1 = self.sqrtcov_lneq1, 
			sqrtcov_leq1 = self.sqrtcov_leq1, rho_c_vec = self.rho_c_vec,
			sigma0 = self.sigma0, sigma1 = self.sigma1, k_low = self.k_low,
			k_high = self.k_high, r_max = self.r_max, l_max = self.l_max,
			wavelength = self.wavelength, r_step = self.r_step,
			gridpoints = self.gridpoints, rgrid = self.rgrid, 
			rho_c_grid = self.rho_c_grid, rho_c_hires = self.rho_c_hires,
			rho_d_grid = self.rho_d_grid, rho_d_hires = self.rho_d_hires)

	def Load(self,fname):
		"""
		Load all object data udeful to Sampler as numpy arrays from a .npz file.
		"""
		data = np.load(fname)
		self.sqrtcov_lneq1 = data['sqrtcov_lneq1']		
		self.sqrtcov_leq1 = data['sqrtcov_leq1']		
		self.rho_c_vec = data['rho_c_vec']
		self.sigma0 = data['sigma0']
		self.sigma1 = data['sigma1']
		self.k_low = data['k_low']
		self.k_high = data['k_high']
		self.r_max = data['r_max']
		self.l_max = data['l_max']
		self.wavelength = data['wavelength']
		self.r_step = data['r_step']
		self.gridpoints = data['gridpoints']
		self.rgrid = data['rgrid']
		self.rho_c_grid = data['rho_c_grid']
		self.rho_c_hires = data['rho_c_hires']
		self.rho_d_grid = data['rho_d_grid']
		self.rho_d_hires = data['rho_d_hires']
