#!/usr/bin/python
from Covariance import Covariance
from Sampler import Sampler
import numpy as np
import pickle
from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 8
plt.rc('font', size=16)

"""
A script to plot comparisons between sample and analytic
means and variances, as a sanity check for the sampler.
"""

def SampleMean(samples):
	return np.mean(samples,axis=0)

def SampleSigma(samples):
	mean = np.mean(samples,axis=0)
	accum = np.zeros(samples.shape[1])
	for n in range(0,samples.shape[0]):
		accum += np.square(samples[n]-mean)
	accum/=samples.shape[0]
	return np.sqrt(accum)

def StdErr(samples):
	return SampleSigma(samples)/np.sqrt(float(samples.shape[0]))

data = np.load("samples.npz")
grid = data['grid']
samples = data['samples']
droopmean = data['droopmean']
droopvar = data['droopvar']
nsamples = data['nsamples']
print "NSAMPLES"
print nsamples

axes=[]
fig1, ax1 = plt.subplots()
axes.append(ax1)
#Plot mean comparison
axes[0].plot(grid,droopmean,ls='-',lw=1.5,color='black',label="Analytic")
axes[0].errorbar(grid,SampleMean(samples),yerr=StdErr(samples),marker="o",color='red',linestyle='None',lw=1.0,label="Sample Mean")
#Plot variance comparison
#axes[0].plot(grid,droopvar,marker="o",color='red',linestyle='None',lw=1.0,label="Sample Variance")
#axes[0].plot(grid,np.square(SampleSigma(samples)),ls='-',lw=1.5,color='black',label="Analytic")
axes[0].set_title("1000 Sample Variance Vs. Analytics")
axes[0].set_xlabel('Radius',fontsize=18)
#axes[0].set_ylabel(r'$\mathrm{Var}(\Phi_{00})$',fontsize=18)
axes[0].set_ylabel(r'$\langle\Phi_{00}\rangle$',fontsize=18)
legend = axes[0].legend(loc='lower right',shadow=False,fancybox=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('1.0')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize(16)

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width

for ax in axes:
	ax.tick_params(axis='x', which='major', labelsize=16)
	ax.tick_params(axis='y', which='major', labelsize=16)


# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='white', edgecolor='black')

textlist=[
r'$\bar{\nu}=1.0$'+'\n',
r'$N=8$']
textstr=''.join(textlist)

print textstr

axes[0].text(0.85, 0.95, textstr, transform=axes[0].transAxes, fontsize=18,verticalalignment='top', bbox=props)
#axes[0].text(0.85, 0.55, textstr, transform=axes[0].transAxes, fontsize=18,verticalalignment='top', bbox=props)

plt.show()

