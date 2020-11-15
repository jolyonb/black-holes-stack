"""
This is set up to test linear evolution
"""

import numpy as np
#from stack.model import Model
from model import Model
#from stack.linear.linearevolveMX import Linear
from linearevolveMXold import Linear
import matplotlib.pyplot as plt

# Set up an evenly spaced grid with gridpoints at 0, h/2, 3h/2, 5h/2, etc
rvals = np.linspace(0, 100, 100)
rvals += rvals[1]/2
rvals = np.concatenate((np.array([0]), rvals))

rvals200 = np.linspace(0, 100, 200)
rvals200 += rvals200[1]/2
rvals200 = np.concatenate((np.array([0]), rvals200))

rvals300 = np.linspace(0, 100, 300)
rvals300 += rvals300[1]/2
rvals300 = np.concatenate((np.array([0]), rvals300))
#print(rvals)

# Variables separated for ease of altering test phi
width = 10
nu = 0.2

def test_phi(r):
    return 1 - (1 - nu) * np.exp(-r**2/2/width**2)

def test_phiprime(r):
    return (1 - nu) * r * np.exp(-r**2/2/width**2) / width**2

def test_delU(r):
    m0=2
    lamda = -3/2 + np.sqrt(9/4 + m0**2)
    beta = 1/(2*lamda)
    n_ef = 15
    factor = beta * np.exp(- 2 * n_ef)
    phi = test_phi(r)
    phiprime = test_phiprime(r)
    phipower = test_phi(r) ** (2 * beta - 2)
    return factor * phipower * (beta / 2 * phiprime**2 - phi / r * phiprime)

def test_A(r):
    n_ef = 15
    m0=2
    lamda = -3/2 + np.sqrt(9/4 + m0**2)
    beta = 1/(2*lamda)
    return r * np.exp(n_ef) * test_phi(r)**(-beta)


# Compute a test function for phi
phi = test_phi(rvals)

model = Model(n_ef=15, n_fields=5, mpsi=0.5, m0=2)
linu = Linear(model=model, phi=phi, rvals=rvals,rhomethod=True)
#linrho = Linear(model=model, phi=phi, rvals=rvals, rhomethod=True)

linu.construct_grid()
#linrho.construct_grid()
# # We now have a grid, and we can now fix delta rho and delta rho dot on the grid for testing purposes
# n = 2
# lin.deltarho = np.sinc(n * lin.Agrid / 10)
#linrho.construct_spectral_from_rho()
linu.construct_spectral()

def plotrho(xi):
    tildeR, tildeU, tildem, tilderho = lin.full_evolution(xi)
    plt.plot(lin.Agrid, tilderho)
    plt.show()

def plotU(xi):
    utildeR, utildeU, utildem, utilderho = linu.full_evolution(xi)
    rhotildeR, rhotildeU, rhotildem, rhotilderho = linrho.full_evolution(xi)
    plt.plot(linu.Agrid, utildeU, label='using delta_U')
    plt.plot(linrho.Agrid, rhotildeU, label='using delta_rho_dot')
    plt.legend()
    plt.xlabel('A')
    plt.ylabel('delta U at xi='+str(xi))
    plt.show()

def plotdeltas(xi):
    utildeR, utildeU, utildem, utilderho = linu.full_evolution(xi)
    rhotildeR, rhotildeU, rhotildem, rhotilderho = linrho.full_evolution(xi)

    '''
    plt.plot(linu.Bn, label='using delta_U')
    plt.plot(linrho.Bn, label='using delta_rho_dot')
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('Bn')
    plt.show()

    plt.plot(linu.Cn, label='using delta_U')
    plt.plot(linrho.Cn, label='using delta_rho_dot')
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('Cn')
    plt.show()
    '''

    plt.plot(linu.Agrid, utildeU, label='using delta_U')
    plt.plot(linrho.Agrid, rhotildeU, label='using delta_rho_dot')
    plt.legend()
    plt.xlabel('A')
    plt.ylabel('delta U at xi='+str(xi))
    plt.show()

    plt.plot(linu.Agrid, utilderho, label='using delta_U')
    plt.plot(linrho.Agrid, rhotilderho, label='using delta_rho_dot')
    plt.legend()
    plt.xlabel('A')
    plt.ylabel('delta rho at xi='+str(xi))
    plt.show()

    plt.plot(linu.Agrid, utildeR, label='using delta_U')
    plt.plot(linrho.Agrid, rhotildeR, label='using delta_rho_dot')
    plt.legend()
    plt.xlabel('A')
    plt.ylabel('delta R at xi='+str(xi))
    plt.show()

    plt.plot(self.Agrid, self.delta_U, label='OG delta_U') 
    plt.plot(self.Agrid, self.delU, label='Interpl delta_U') 
    plt.plot(self.Agrid, deltaU, label='reconstructed delta_U')
    plt.legend()
    plt.show()

'''
# Testing interpolation
plt.plot(linu.rvals, linu.phiprime, label='linu.phiprime')
plt.plot(linu.rvals, test_phiprime(linu.rvals), label='test_phiprime')
plt.legend()
plt.xlabel('r')
plt.ylabel('Phi_prime')
plt.show()


plt.plot(test_A(rvals)[1:], linu.delta_U, label='linu.delta_U')
plt.plot(test_A(rvals)[1:], test_delU(rvals[1:]), label='test_delU')
plt.legend()
plt.xlabel('A')
plt.ylabel('delta_U')
plt.show()

random_r = np.sort(np.random.rand(100)*100)
random_A = test_A(random_r)

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Testing interpolation with delta_U')
ax1.plot(random_A, linu.interp(random_A), label='Interpolated delta_U')
ax1.plot(random_A, test_delU(random_r), label='Analytic delta_U')
ax1.legend()
ax1.set(ylabel='delta_U')
ax2.plot(random_A, np.abs(test_delU(random_r) - linu.interp(random_A)))
ax2.set(ylabel = 'Absolute error')
ax3.plot(random_A, np.abs((test_delU(random_r) - linu.interp(random_A)) / test_delU(random_r)))
ax3.set(ylabel = 'Log(Relative error)')
ax3.set_yscale('log')
ax3.set(xlabel='A')
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Testing interpolation with phi')
ax1.plot(random_A, linu.interp(random_A), label='Interpolated phi') #WAIT I DONT THINK THIS WORKS
ax1.plot(random_A, test_phi(random_r), label='Analytic phi')
ax1.legend()
ax1.set(ylabel='delta_U')
ax2.plot(random_A, np.abs(test_phi(random_r) - linu.interp(random_A)))
ax2.set(ylabel = 'Absolute error')
ax3.plot(random_A, np.abs((test_phi(random_r) - linu.interp(random_A)) / test_phi(random_r)))
ax3.set(ylabel = 'Log(Relative error)')
ax3.set_yscale('log')
ax3.set(xlabel='A')
plt.show()


# Testing delta_U reconstruction
utildeR, utildeU, utildem, utilderho = linu.full_evolution(0.)
#linu.construct_spectral_from_u1()
#utildeR, utildeU, utildem, utilderho = linu.full_evolution(0.)
#rhotildeR, rhotildeU, rhotildem, rhotilderho = linrho.full_evolution(0.)


# Getting a black hole to form
ximax = 2.* np.log(2.7437072699922695*linu.max_A / np.pi * np.sqrt(3))
print("xi_max is " + str(ximax))
xivals = np.linspace(0, ximax, 100)
#print(xivals)
print("rho_at_origin(35) = " + str(linu.rho_at_origin(35.)))
rhovals = []
for i in xivals:
    rhovals.append(linu.rho_at_origin(i))
plt.plot(xivals, np.array(rhovals))
plt.xlabel('xi')
plt.ylabel('delta_rho(xi, bar(A)=0)')
plt.show()


xi = 35.
utildeR, udeltaU, utildem, udeltarho = linu.full_evolution(xi)
print(udeltarho[0])
plt.plot(linu.Agrid, udeltarho)
plt.xlabel('bar(A)')
plt.ylabel('delta_rho(xi=' + str(xi) + ')')
plt.show()

'''
# Testing finite diff, delta U

phi200 = test_phi(rvals200)
phi300 = test_phi(rvals300)

linu200 = Linear(model=model, phi=phi200, rvals=rvals200)
linu300 = Linear(model=model, phi=phi300, rvals=rvals300)

linu200.construct_grid()
linu300.construct_grid()

linu200.construct_spectral()
linu300.construct_spectral()

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Testing finite diff with delta_U')
ax1.plot(linu.rvals, linu.delta_U, label='Finite diff 100 pts')
ax1.plot(linu200.rvals, linu200.delta_U, label='Finite diff 200 pts')
ax1.plot(linu300.rvals, linu300.delta_U, label='Finite diff 300 pts')
ax1.plot(linu.rvals, test_delU(linu.rvals), label='Analytic')
ax1.legend()
ax1.set(ylabel='delta_U')

ax2.plot(linu.rvals, np.abs(test_delU(linu.rvals) - linu.delta_U),label='100 pts')
ax2.plot(linu200.rvals, np.abs(test_delU(linu200.rvals) - linu200.delta_U),label='200 pts')
ax2.plot(linu300.rvals, np.abs(test_delU(linu300.rvals) - linu300.delta_U),label='300 pts')
ax2.set(ylabel = 'Absolute error')
ax2.legend()

ax3.plot(linu.rvals, np.abs((test_delU(linu.rvals) - linu.delta_U) / test_delU(linu.rvals)),label='100 pts')
ax3.plot(linu200.rvals, np.abs((test_delU(linu200.rvals) - linu200.delta_U) / test_delU(linu200.rvals)),label='200 pts')
ax3.plot(linu300.rvals, np.abs((test_delU(linu300.rvals) - linu300.delta_U) / test_delU(linu300.rvals)),label='300 pts')
ax3.set(ylabel = 'Log(Relative error)')
ax3.set_yscale('log')
ax3.set(xlabel='r')
ax3.legend()

plt.show()

#Testing finite diff, dPhi

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Testing finite diff with dPhi/dr')
ax1.plot(linu.rvals, linu.phiprime, label='Finite diff 100 pts')
ax1.plot(linu200.rvals, linu200.phiprime, label='Finite diff 200 pts')
ax1.plot(linu300.rvals, linu300.phiprime, label='Finite diff 300 pts')
ax1.plot(linu.rvals, test_phiprime(linu.rvals), label='Analytic')
ax1.legend()
ax1.set(ylabel='dPhi/dr')

ax2.plot(linu.rvals, np.abs(test_phiprime(linu.rvals) - linu.phiprime),label='100 pts')
ax2.plot(linu200.rvals, np.abs(test_phiprime(linu200.rvals) - linu200.phiprime),label='200 pts')
ax2.plot(linu300.rvals, np.abs(test_phiprime(linu300.rvals) - linu300.phiprime),label='300 pts')
ax2.set(ylabel = 'Absolute error')
ax2.legend()

ax3.plot(linu.rvals, np.abs((test_phiprime(linu.rvals) - linu.phiprime) / test_phiprime(linu.rvals)),label='100 pts')
ax3.plot(linu200.rvals, np.abs((test_phiprime(linu200.rvals) - linu200.phiprime) / test_phiprime(linu200.rvals)),label='200 pts')
ax3.plot(linu300.rvals, np.abs((test_phiprime(linu300.rvals) - linu300.phiprime) / test_phiprime(linu300.rvals)),label='300 pts')
ax3.set(ylabel = 'Log(Relative error)')
ax3.set_yscale('log')
ax3.set(xlabel='r')
ax3.legend()

plt.show()

# Testing coefficients 
# Note: full_evolution() has been modified in linearevolve.py to solely output deltaU for this
delta_UC = linu.full_evolution(0.)

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Testing coefficient reconstruction with delta_U')
ax1.plot(linu.rvals, delta_UC, label='Reconstructed')
ax1.plot(linu.rvals, linu.delta_U, label='Finite diff 100 pts')
ax1.legend()
ax1.set(ylabel='delta_U')

ax2.plot(linu.rvals, np.abs(linu.delta_U - delta_UC))
ax2.set(ylabel = 'Absolute error')

ax3.plot(linu.rvals, np.abs((linu.delta_U - delta_UC) / linu.delta_U))
ax3.set(ylabel = 'Log(Relative error)')
ax3.set_yscale('log')
ax3.set(xlabel='r')

plt.show()


# Testing coefficients + finite diff
# Note: full_evolution() has been modified in linearevolve.py to solely output deltaU for this

delta_UC200 = linu200.full_evolution(0.)
delta_UC300 = linu300.full_evolution(0.)

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Testing finite diff + reconstruction with delta_U')
ax1.plot(linu.rvals, delta_UC, label='Reconstructed 100 pts')
ax1.plot(linu200.rvals, delta_UC200, label='Reconstructed 200 pts')
ax1.plot(linu300.rvals, delta_UC300, label='Reconstructed 300 pts')
ax1.plot(linu.rvals, test_delU(linu.rvals), label='Analytic')
ax1.legend()
ax1.set(ylabel='delta_U')

ax2.plot(linu.rvals, np.abs(test_delU(linu.rvals) - delta_UC),label='100 pts')
ax2.plot(linu200.rvals, np.abs(test_delU(linu200.rvals) - delta_UC200),label='200 pts')
ax2.plot(linu300.rvals, np.abs(test_delU(linu300.rvals) - delta_UC300),label='300 pts')
ax2.set(ylabel = 'Absolute error')
ax2.legend()

ax3.plot(linu.rvals, np.abs((test_delU(linu.rvals) - delta_UC) / test_delU(linu.rvals)),label='100 pts')
ax3.plot(linu200.rvals, np.abs((test_delU(linu200.rvals) - delta_UC200) / test_delU(linu200.rvals)),label='200 pts')
ax3.plot(linu300.rvals, np.abs((test_delU(linu300.rvals) - delta_UC300) / test_delU(linu300.rvals)),label='300 pts')
ax3.set(ylabel = 'Log(Relative error)')
ax3.set_yscale('log')
ax3.set(xlabel='r')
ax3.legend()

plt.show()


'''

# Finding final values

ximax = 2.* np.log(2.7437072699922695*linu.max_A / np.pi * np.sqrt(3))
print("xi_max is " + str(ximax))
#tildeR_max, tildeU_max, tildem_max, tilderho_max = linu.full_evolution(ximax)
xi = 37. #delta_u(A=0)~0.8 here, showing black hole does form
tildeR, tildeU, tildem, tilderho = linu.full_evolution(xi)

np.savetxt("Agrid.csv", linu.Agrid, delimiter=",")
np.savetxt("U_37.csv", tildeU, delimiter=",")
np.savetxt("R_37.csv", tildeR, delimiter=",")
np.savetxt("m_37.csv", tildem, delimiter=",")
np.savetxt("rho_37.csv", tilderho, delimiter=",")

#plt.plot(linu.Agrid, tildeU_max, label='xi='+str(ximax))
plt.title('tilde{U} at xi='+str(xi))
plt.plot(linu.Agrid, tildeU, label='xi='+str(xi))
#plt.legend()
plt.xlabel('A')
plt.ylabel('tilde{U}')
plt.show()

plt.plot(linu.Agrid, tildeR, label='xi='+str(xi))
plt.title('tilde{R} at xi='+str(xi))
plt.xlabel('A')
plt.ylabel('tilde{R}')
plt.show()

plt.plot(linu.Agrid, tildem, label='xi='+str(xi))
plt.title('tilde{m} at xi='+str(xi))
plt.xlabel('A')
plt.ylabel('tilde{m}')
plt.show()

plt.plot(linu.Agrid, tilderho, label='xi='+str(xi))
plt.title('tilde{rho} at xi='+str(xi))
plt.xlabel('A')
plt.ylabel('tilde{rho}')
plt.show()

'''
