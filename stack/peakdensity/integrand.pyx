cimport vegas
from libc.math cimport exp
from libc.math cimport sqrt
from libc.math cimport tan
from libc.math cimport cos
from libc.math cimport fabs

import vegas
import numpy
import math

cdef double Q_Z(double gamma, double nu, double[::1] lam, double[::1] m_params):
	cdef double trlam = lam[0] + lam[1] + lam[2]
	cdef double trlamsq = lam[0]**2 + lam[1]**2 + lam[2]**2
	cdef double trM = m_params[0]**2 + m_params[1]**2 + m_params[2]**2 + m_params[3]**2 + m_params[4]**2 + m_params[5]**2
	cdef double ret = trM  + (1/(1-gamma**2))*(gamma*nu + trlam)**2 + 5.0/2.0 * (3.0*trlamsq - trlam**2)

	return ret


cdef double det_Z(double[::1] lam, double[::1] m_params, double nufact):
	cdef double l1=lam[0]
	cdef double l2=lam[1]
	cdef double l3=lam[2]
	cdef double a=m_params[0]
	cdef double b=m_params[1]
	cdef double c=m_params[2]
	cdef double d=m_params[3]
	cdef double e=m_params[4]
	cdef double f=m_params[5]

	cdef double term1 = a*a*b*b*c*c
	cdef double term2 = l1*l2*l3
	cdef double term3 = (c*c + e*e + f*f)*l1*l2 + (b*b + d*d)*l1*l3 + a*a*l2*l3
	cdef double term4 = -2.0*b*d*e*f*l1 + (c*c + e*e)*(d*d*l1 + a*a*l2) + b*b*(c*c*l1 + f*f*l1 + a*a*l3)

	term2 *= nufact**3
	term3 *= nufact**2
	term4 *= nufact

	cdef double detret=term1 + term2 + term3 + term4

	return detret


cdef double delta(double[::1] lam):
	cdef double lam1=lam[0]
	cdef double lam2=lam[1]
	cdef double lam3=lam[2]
	cdef double delret = fabs((lam1-lam2)*(lam1-lam3)*(lam2-lam3))
	return delret


cdef unsigned int sylv_index(double[::1] lam, double[::1] m_params, double nufact, double mydet):

	cdef double l1=lam[0]
	cdef double l2=lam[1]
	cdef double l3=lam[2]
	cdef double a=m_params[0]
	cdef double b=m_params[1]
	cdef double c=m_params[2]
	cdef double d=m_params[3]
	cdef double e=m_params[4]
	cdef double f=m_params[5]


	cdef double syl1= nufact*l1 + a*a
	cdef double syl2 = syl1*(nufact * l2 + b*b + d*d) - a*a * d*d
	cdef double syl3= mydet

	cdef unsigned int index
	
	#Index is the number of negative eigenvalues.

	if ((syl1>0.0)and(syl2>0.0)and(syl3>0.0)):
		index=0
	elif ((syl1<0.0)and(syl2>0.0)and(syl3<0.0)):
		index=3
	else:
		if(syl3<0.0):
			index=1
		else:
			index=2

	return index


cdef class f_cython(vegas.BatchIntegrand):
	cdef double N
	cdef double nu
	cdef double gamma
	cdef double pi
	cdef double nufact
	cdef readonly int dim

	def __init__(self,dim,N,nu,gamma):
		self.dim=dim
		self.N=N
		self.nu=nu
		self.gamma=gamma
		self.nufact = 3.0*nu/gamma
		self.pi= numpy.pi


	def __call__(self, double[:, ::1] x):
		cdef double[:,::1] val = numpy.zeros((x.shape[0],5),float)
		cdef double[::1] tanx = numpy.empty(x.shape[1],float)
		cdef double[::1] lam = numpy.empty(3,float)
		cdef double[::1] m_params = numpy.empty(6,float)

		cdef double det
		cdef unsigned int index
		cdef double delt
		cdef double expQ
		cdef double signedval
		cdef double pointval


		cdef double a
		cdef double b
		cdef double c

		cdef double ijac
		cdef int i,l,k

		for i in range(val.shape[0]):
			ijac=1.0
			for l in range(self.dim):
				ijac*=(1/cos(x[i,l]))**2
				tanx[l]=tan(x[i,l])

			for k in range(0,3):
				lam[k] = tanx[k]
			for k in range(3,9):
				m_params[k-3]=tanx[k]
			

			a=m_params[0]
			b=m_params[1]
			c=m_params[2]
		
			det = det_Z(lam,m_params,self.nufact)
			delt = delta(lam)
			expQ = exp((-1.0/2.0)*Q_Z(self.gamma,self.nu,lam,m_params))
						
			signedval = (1.0/6.0)*(ijac)*a*a*b*((a*b*c)**(self.N-4.0))*delt*det*expQ
			pointval = (1.0/6.0)*(ijac)*a*a*b*((a*b*c)**(self.N-4.0))*delt*fabs(det)*expQ
			#pointval = (1.0/6.0)*(ijac)*(a**3)*(b**2)*c*((a*b*c)**(self.N-5.0))*delt*fabs(det)*expQ

			index=sylv_index(lam,m_params,self.nufact,det)

			val[i,0] = signedval
				
			val[i,index+1] = pointval

		return val
