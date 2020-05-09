#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul	6 14:46:18 2017

@author: jolyon
"""

import numpy as np
import scipy
from scipy import special
from scipy import integrate as sciint
from math import pi

class LevinIntegrals(object) :
	"""
	Implements levin integration by method of collocation. An "I type" integral
	(function*j_l(alpha*k)) is computed by the member function compute_I. This
	class is wrapped by the SphericalBesselIntegrator class.
	"""
	def __init__(self, numpoints) :
			self.numpoints = numpoints
			theta = pi * np.linspace(0.0, 1.0, numpoints)
			# Collocation points in x space
			self.colloc = -np.cos(theta)

			# These are (n, collocation) indexed
			self.chebyTs = np.array([self.chebyT(theta, n) for n in range(numpoints)])
			self.dchebyTs = np.array([n * self.chebyU(theta, n-1) for n in range(0, numpoints)])

	def kvals(self, a, b) :
			"""Return k values for a given set of collocation points, given a and b"""
			return (b-a)/2*self.colloc + (b+a)/2

	@staticmethod
	def chebyT(theta, n) :
			"""Returns Chebyshev polynomials of the first kind of order n at value -theta"""
			return (-1) ** (n % 2) * np.cos(n * theta)

	@staticmethod
	def chebyU(theta, n) :
			"""Returns Chebyshev polynomials of the second kind of order n at value -theta"""
			if n == -1 :
					return np.zeros_like(theta)
			subtheta = theta[1:-1]
			n1 = n + 1
			return (-1) ** (n % 2) * np.concatenate(([n1], np.sin(n1 * subtheta) / np.sin(subtheta), [(-1) ** (n % 2) * n1]))
	
	def compute_K(self, a, b, alpha, beta, l, func):
		"""Computes int(func(k) j_l(alpha k) j_l(beta k), {k, a, b})"""
		
		Abase = np.zeros((4, 4))
		Abase[0, 1] = -alpha
		Abase[0, 2] = -beta
		Abase[1, 0] = alpha
		Abase[2, 0] = beta
		Abase[1, 3] = -beta
		Abase[3, 1] = beta
		Abase[2, 3] = -alpha
		Abase[3, 2] = alpha
		
		diagbase = np.diag([2 * l, -2, -2, -2 * (2 + l)])
		
		ks = self.kvals(a, b)
		derivs = 2 / (b - a) * self.dchebyTs
		
		sphlaa = scipy.special.spherical_jn(l, alpha * a)
		sphlba = scipy.special.spherical_jn(l, beta * a)
		sphl1aa = scipy.special.spherical_jn(l + 1, alpha * a)
		sphl1ba = scipy.special.spherical_jn(l + 1, beta * a)
		sphlab = scipy.special.spherical_jn(l, alpha * b)
		sphlbb = scipy.special.spherical_jn(l, beta * b)
		sphl1ab = scipy.special.spherical_jn(l + 1, alpha * b)
		sphl1bb = scipy.special.spherical_jn(l + 1, beta * b)
		
		weightsa = np.array([
			sphlaa * sphlba,
			sphl1aa * sphlba,
			sphlaa * sphl1ba,
			sphl1aa * sphl1ba
		])
		weightsb = np.array([
			sphlab * sphlbb,
			sphl1ab * sphlbb,
			sphlab * sphl1bb,
			sphl1ab * sphl1bb
		])
		
		# Indexed by weight, then collocation point
		flist = np.transpose(np.array([[func(k), 0.0, 0.0, 0.0] for k in ks]))
		
		# Construct the matrix equations a_{ijkl} c_{ik} = f_{jl}
		# a has indices of weight, weight, Chebyshev order, collocation index
		# F_i(x) = sum_k c_{ik} T_k(x)
		a = np.zeros([4, 4, self.numpoints, self.numpoints])
		# Loop over collocation points
		for l, kval in enumerate(ks):
			Amat = Abase + diagbase / kval
			a[:, :, :, l] = np.tensordot(Amat, self.chebyTs[:, l], 0) + np.tensordot(np.eye(4), derivs[:, l], 0)
		
		# Reorganize the tensor so that the contracted indices are rightmost
		atensor = np.transpose(a, (1, 3, 0, 2))
		
		# Solve the system
		cij = np.linalg.tensorsolve(atensor, flist)
		
		# Construct the integral
		resulta = np.dot(np.dot(cij, self.chebyTs[:, 0]), weightsa)
		resultb = np.dot(np.dot(cij, self.chebyTs[:, -1]), weightsb)
		
		# Return the result
		return resultb - resulta
	
	def compute_H(self, a, b, alpha, l, func):
		"""Computes int(func(k) j_l(alpha k)^2, {k, a, b})"""
		
		Abase = np.zeros((3, 3))
		Abase[0, 1] = -2.0 * alpha
		Abase[1, 0] = alpha
		Abase[1, 2] = -alpha
		Abase[2, 1] = 2.0 * alpha
		
		diagbase = np.diag([2 * l, -2, -2 * (2 + l)])
		
		ks = self.kvals(a, b)
		derivs = 2 / (b - a) * self.dchebyTs
		
		sphla = scipy.special.spherical_jn(l, alpha * a)
		sphl1a = scipy.special.spherical_jn(l + 1, alpha * a)
		sphlb = scipy.special.spherical_jn(l, alpha * b)
		sphl1b = scipy.special.spherical_jn(l + 1, alpha * b)
		
		weightsa = np.array([
			sphla ** 2,
			sphla * sphl1a,
			sphl1a ** 2
		])
		weightsb = np.array([
			sphlb ** 2,
			sphlb * sphl1b,
			sphl1b ** 2
		])
		
		# Indexed by weight, then collocation point
		flist = np.transpose(np.array([[func(k), 0.0, 0.0] for k in ks]))
		
		# Construct the matrix equations a_{ijkl} c_{ik} = f_{jl}
		# a has indices of weight, weight, Chebyshev order, collocation index
		# F_i(x) = sum_k c_{ik} T_k(x)
		a = np.zeros([3, 3, self.numpoints, self.numpoints])
		# Loop over collocation points
		for l, kval in enumerate(ks):
			Amat = Abase + diagbase / kval
			a[:, :, :, l] = np.tensordot(Amat, self.chebyTs[:, l], 0) + np.tensordot(np.eye(3), derivs[:, l], 0)
		
		# Reorganize the tensor so that the contracted indices are rightmost
		atensor = np.transpose(a, (1, 3, 0, 2))
		
		# Solve the system
		cij = np.linalg.tensorsolve(atensor, flist)
		
		# Construct the integral
		resulta = np.dot(np.dot(cij, self.chebyTs[:, 0]), weightsa)
		resultb = np.dot(np.dot(cij, self.chebyTs[:, -1]), weightsb)
		
		# Return the result
		return resultb - resulta
	
	def compute_I(self, a, b, alpha, l, func) :
			"""Computes int(func(k) j_l(alpha k), {k, a, b})"""

			Abase = np.zeros((2,2))
			Abase[0,1] = -alpha
			Abase[1,0] = alpha

			diagbase = np.diag([l, -(2+l)])

			ks = self.kvals(a, b)
			derivs = 2 / (b - a) * self.dchebyTs

			sphla = scipy.special.spherical_jn(l, alpha * a)
			sphl1a = scipy.special.spherical_jn(l+1, alpha * a)
			sphlb = scipy.special.spherical_jn(l, alpha * b)
			sphl1b = scipy.special.spherical_jn(l+1, alpha * b)

			weightsa = np.array([
									sphla,
									sphl1a
								 ])
			weightsb = np.array([
									sphlb,
									sphl1b
								 ])

			# Indexed by weight, then collocation point
			flist = np.transpose(np.array([[func(k), 0.0] for k in ks]))

			# Construct the matrix equations a_{ijkl} c_{ik} = f_{jl}
			# a has indices of weight, weight, Chebyshev order, collocation index
			# F_i(x) = sum_k c_{ik} T_k(x)
			a = np.zeros([2, 2, self.numpoints, self.numpoints])
			# Loop over collocation points
			for l, kval in enumerate(ks) :
					Amat = Abase + diagbase / kval
					a[:,:,:,l] = np.tensordot(Amat, self.chebyTs[:,l], 0) + np.tensordot(np.eye(2), derivs[:,l], 0)

			# Reorganize the tensor so that the contracted indices are rightmost
			atensor = np.transpose(a, (1, 3, 0, 2))

			# Solve the system
			cij = np.linalg.tensorsolve(atensor, flist)

			# Construct the integral
			resulta = np.dot(np.dot(cij, self.chebyTs[:, 0]), weightsa)
			resultb = np.dot(np.dot(cij, self.chebyTs[:, -1]), weightsb)

			# Return the result
			return resultb - resulta

def calc_relerr(err,result):
	if err!=0.0:
		rel_err = abs(err/result)
	else:
		rel_err = 0
	return rel_err

class SphericalBesselIntegrator(object) :
	"""
	A wrapper class for LevinIntegrals. Instantiates two LevinIntegrals
	objects at 21 and 10 collocation points to estimate error (persistence
	of these objects requires a wrapper class, instead of a function).
	Also implements case-checking and -handling for alpha==0. ICalc() computes
	an "I type" integral of a function func() times j_l(alpha*k).
	"""

	def __init__(self) :
		#Allow three stages of recursive subdivision.
		self.integrators=[]
		base_divisions=10
		self.recursion_power=0
		#self.max_recursion=6
		self.max_recursion=8
		for p in range(self.max_recursion+1):
			low_div = base_divisions*2**p
			high_div = 2*low_div-1
			self.integrators.append([LevinIntegrals(low_div),LevinIntegrals(high_div)])
		#FIXME: shouldn't hard-code relative tolerance!
		self.rel_tol = 1e-6

	def k_integrand(self, k, l, alpha, beta, func):
		res = func(k) * special.spherical_jn(l, k*alpha) * special.spherical_jn(l, k*beta)
		return res

	def h_integrand(self, k, l, alpha, func):
		res = func(k) * special.spherical_jn(l, k*alpha)**2
		return res
	def i_integrand(self, k, l, alpha, func):
		res = func(k) * special.spherical_jn(l, k*alpha)
		return res
	
	def KCalc(self, a, b, alpha_tup, beta_tup, l, func):
		"""
		Assuming the first entry in the radial coordinate array
		is zero. Check to see if both alpha and beta have index
		zero and l==0. If so, the collocation method fails because the
		differential matrix is singular. But in this case the
		covariance is simply the integral of the power spectrum,
		which is easily done with scipy's built-in quadrature routine.
		"""
		
		if (alpha_tup[1] == 0 and beta_tup[1] == 0 and l == 0):
			# Absolute error tolerance
			intepsrel = 1e-8
			# Compute integral using quadrature.
			(result, err) = sciint.quad(func, a, b, epsrel=intepsrel)
			# Estimate and check relative error.
			rel_err = abs(err / result)
			if rel_err > self.rel_tol:
				# FIXME: handle this error properly!
				print(str(self.rel_tol) + " Relative Error Bound Exceeded (Quadrature)")
				print(rel_err)
		else:
			result_hiprec = self.integrators[self.recursion_power][1].compute_K(a, b, alpha_tup[0], beta_tup[0], l,
																				func)
			result_loprec = self.integrators[self.recursion_power][0].compute_K(a, b, alpha_tup[0], beta_tup[0], l,
																				func)
			result = result_hiprec
			# Estimate and check relative error.
			err = abs(result_hiprec - result_loprec)
			if err != 0.0:
				rel_err = abs(err / result)
			else:
				rel_err = 0
			if (rel_err > self.rel_tol):
				if self.recursion_power < self.max_recursion:
					self.recursion_power += 1
					print(self.recursion_power)
					print(rel_err)
					result = self.KCalc(a, b, alpha_tup, beta_tup, l, func)
				else:
					print("KCalc Max Recursions Exceeded (LevinCollocation)")
					print("Switching to Quadrature")
					intepsrel = self.rel_tol
					(result, err) = sciint.quad(self.k_integrand, a, b, args=(l, alpha_tup[0], beta_tup[0], func),
												epsrel=intepsrel, limit=10000)
					if abs(err / result) > self.rel_tol:
						print("KCalc Error Tolerance Exceeded")
			# FIXME: handle this error properly!
		
		self.recursion_power = 0
		return result
	
	def HCalc(self, a, b, alpha_tup, l, func):
		"""
		Assuming the first entry in the radial coordinate array
		is zero. Check to see if both alpha has index
		zero and l==0. See KCalc().
		"""
		
		if (alpha_tup[1] == 0 and l == 0):
			# Absolute error tolerance
			intepsrel = 1e-8
			# Compute integral using quadrature.
			(result, err) = sciint.quad(func, a, b, epsrel=intepsrel)
			# Estimate and check relative error.
			rel_err = abs(err / result)
			if rel_err > self.rel_tol:
				# FIXME: handle this error properly!
				print(str(self.rel_tol) + " Relative Error Bound Exceeded (Quadrature)")
		else:
			result_hiprec = self.integrators[self.recursion_power][1].compute_H(a, b, alpha_tup[0], l, func)
			result_loprec = self.integrators[self.recursion_power][0].compute_H(a, b, alpha_tup[0], l, func)
			result = result_hiprec
			# Estimate and check relative error.
			err = abs(result_hiprec - result_loprec)
			if err != 0.0:
				rel_err = abs(err / result)
			else:
				rel_err = 0
			if (rel_err > self.rel_tol):
				if self.recursion_power < self.max_recursion:
					self.recursion_power += 1
					result = self.HCalc(a, b, alpha_tup, l, func)
				else:
					print("HCalc Max Recursions Exceeded (LevinCollocation)")
					print("Switching to Quadrature")
					intepsrel = self.rel_tol
					(result, err) = sciint.quad(self.h_integrand, a, b, args=(l, alpha_tup[0], func), epsrel=intepsrel,
												limit=10000)
					if abs(err / result) > self.rel_tol:
						print("HCalc Error Tolerance Exceeded")
			# FIXME: handle this error properly!
			# FIXME: handle this error properly!
		self.recursion_power = 0
		return result
	
	def ICalc(self, a, b, alpha, l, func):
		"""
		Takes integration bounds [a,b], radial coefficent [alpha], mode [l],
		and function [func], and computes the integral
		Integral[func(k)*j_l(alpha*k),{a,b,k}] using the compute_I method
		of the LevinIntegrals class. Assuming the first entry in the radial
		coordinate array is zero. Check to see if both alpha==0.0  and l==0.
		The first case, in which alpha==0.0 and l==0, corresponds to a straight
		integral over the power spectrum, which is done with SciPy quad().
		The second case, where we are away from the origin, or when l!=0,
		performs the levin integration with an adaptive collocation grid resolution
		(up to a recursion limit defined in the constructor). If the error estimate
		(computed by comparing resolution levels n and (n+1)) is greater than the
		tolerance self.rel_tol after reaching maxmal recursion depth, we try quadrature.
		"""
		if (alpha==0.0 and l==0):
			#Absolute error tolerance
			intepsrel=self.rel_tol*1e-2
			#Compute integral using quadrature.
			(result,err) = sciint.quad(func,a,b,epsrel=intepsrel)
			#Estimate and check relative error.
			rel_err = calc_relerr(err,result)
			if rel_err > self.rel_tol:
				#FIXME: handle this error properly!
				print(str(self.rel_tol)+" Relative Error Bound Exceeded (Quadrature)")
				assert False
		else:
			result_hiprec = self.integrators[self.recursion_power][1].compute_I(a, b, alpha, l, func)
			result_loprec = self.integrators[self.recursion_power][0].compute_I(a, b, alpha, l, func)
			result = result_hiprec
			#Estimate and check relative error.
			err = abs(result_hiprec-result_loprec)
			rel_err = calc_relerr(err,result)
			if (rel_err > self.rel_tol):
				if self.recursion_power<self.max_recursion:
					self.recursion_power+=1
					#print self.recursion_power
					#print rel_err
					result = self.ICalc(a,b,alpha,l,func)
				else:
					print("ICalc Max Recursions Exceeded (LevinCollocation)")
					print("Switching to Quadrature")
					intepsrel=self.rel_tol*1e-2
					(result,err) = sciint.quad(self.i_integrand,a,b,args=(l,alpha,func),epsrel=intepsrel,limit=10000)
					rel_err = calc_relerr(err,result)
					if rel_err>self.rel_tol:
						print("ICalc Error Tolerance Exceeded")
						assert False
		self.recursion_power=0
		return result

