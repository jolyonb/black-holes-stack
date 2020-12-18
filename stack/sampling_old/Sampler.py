#! /usr/bin/python
import numpy as np
import pickle
from tqdm import tqdm
from numpy.random import multivariate_normal

class Sampler(object):
	def __init__(self,covariance):
		"""
		Takes a Covariance object [covariance], and grabs some
		useful data from it.
		"""
		self.covariance = covariance
		self.rho_c_vec = self.covariance.rho_c_vec
		self.gridpoints = self.covariance.gridpoints
		self.l_max = self.covariance.l_max
		self.sigma0 = self.covariance.sigma0

	def GetSamples(self,nubar,nfields,nsamples):
		"""
		Takes a dimensionless field height [nubar], number of 
		waterfall fields [nfields], and number of samples [nsamples],
		and returns a 2D array of samples of Phi_{00} over the radial
		grid specified in the Covariance object. The first index points 
		to the sample number, and the second is the radial index within a 
		given sample.
		"""
		zerovec = np.zeros(self.gridpoints)
		identity = np.identity(self.gridpoints)
		PhiArray = np.zeros([nsamples,self.gridpoints])
		for n in tqdm(range(nsamples)):
			for alpha in range(1,nfields+1):
				for l in range(0,self.l_max+1):
					if (l==0 and alpha==1):
						mean = np.sqrt(4.0*np.pi)*nubar*self.rho_c_vec
					else:
						mean = zerovec
					for m in range(-l,l+1):
						sqrt_cov_mat = self.covariance.GetSqrtCovariance(l,alpha)
						z = multivariate_normal(zerovec,identity)
						phi_lm = np.dot(sqrt_cov_mat,z) + mean
						PhiArray[n,:]+=np.square(phi_lm)
		#Up to this point, the whole calculation has been done with the
		#spectrum moment normalizations removed (hence the rho_c and rho_d
		#in the covariance calculations). We now restore the sigma_0^2 prefactor
		#to get the magnitude of the field right!
		PhiArray = PhiArray*self.sigma0**2
		PhiArray = PhiArray*(1/np.sqrt(4.0*np.pi))
		return PhiArray		
