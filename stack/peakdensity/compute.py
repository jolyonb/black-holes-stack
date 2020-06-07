"""Number Density of Stationary Points for Chi-Squared Fields
v1.0 by Jolyon Bloomfield and Zander Moss, December 2016
See arXiv:1612.03890 for details
Computational Module"""

import argparse
import numpy as np
from stack.peakdensity.stationary_vegas import number_density
from stack.peakdensity.common import signed_exact
# Status bar library
# sudo pip install tqdm
from tqdm import tqdm

#
# Deal with command line arguments
#
parser = argparse.ArgumentParser(description="Compute the number density of stationary points in chi^2 fields")

# Helper functions to demand valid ranges on variables from command line arguments
def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

def is_pos_int(str):
    if str.isdigit() :
        return True
    return False

def check_N(value):
    if not is_pos_int(value) :
        raise argparse.ArgumentTypeError("must be a positive integer >= 4. You supplied: %s" % value)
    ivalue = int(value)
    if ivalue < 4:
        raise argparse.ArgumentTypeError("must be at >= 4. You supplied: %s" % value)
    return ivalue

def check_gamma(value):
    if not is_float(value) :
        raise argparse.ArgumentTypeError("must be a real number. You supplied: %s" % value)
    ivalue = float(value)
    if ivalue <= 0 or ivalue >= 1:
        raise argparse.ArgumentTypeError("must lie between 0 and 1 (exclusive). You supplied: %s" % value)
    return ivalue

def check_samples(value):
    if not is_float(value) :
        raise argparse.ArgumentTypeError("must be a number. You supplied: %s" % value)
    ivalue = float(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("must be at least 1. You supplied: %s" % value)
    return int(ivalue)

def pos_float(value):
    if not is_float(value) :
        raise argparse.ArgumentTypeError("must be a number. You supplied: %s" % value)
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be > 0. You supplied: %s" % value)
    return ivalue

def pos_int(value):
    if not is_pos_int(value) :
        raise argparse.ArgumentTypeError("must be a positive integer. You supplied: %s" % value)
    return int(value)

# Required arguments
# Values for N and gamma
parser.add_argument("N", type=check_N, help="Number of fields")
parser.add_argument("gamma", type=check_gamma, help="Value for gamma")

# Optional arguments
# Output file
parser.add_argument("-o", help="Output file (default 'data.dat')",
    type=argparse.FileType('w'), default="data.dat", dest="file")

# Integrator
method_group = parser.add_mutually_exclusive_group()
method_group.add_argument("-v", help="Use VEGAS integration (default)",
    dest="method", action="store_const", const="vegas")
method_group.add_argument("-m", help="Use Metropolis-Hastings Monte Carlo integration",
    dest="method", action="store_const", const="mcmc")
parser.set_defaults(method='vegas')

# Number of samples
parser.add_argument("-s", help="Number of samples (default 1e5)",
    default=int(1e5), type=check_samples, dest="samples")

# sigma_0 and sigma_1
parser.add_argument("-s0", help="Value for sigma_0 (default 1.0)",
    default=1.0, type=pos_float, dest="sigma0")
parser.add_argument("-s1", help="Value for sigma_1 (default 1.0)",
    default=1.0, type=pos_float, dest="sigma1")

# Values for scanning over nu
parser.add_argument("-r", help="Range for nu (default 0.05 6.0)",
    default=[0.05, 6.0], type=pos_float, dest="nu_range", nargs=2, metavar=('MIN', 'MAX'))
parser.add_argument("-n", help="Number of steps to take in nu (default 20)",
    default=20, type=pos_int, dest="steps")

# Include zero?
parser.add_argument("-0", help="Do not include nu=0 (default includes nu=0)", action="store_false",
    dest="includezero", default=True)

# Quiet mode? Verbose mode?
noise_group = parser.add_mutually_exclusive_group()
noise_group.add_argument("-q", help="Quiet mode", action="store_true", dest="quiet")
noise_group.add_argument("-V", help="Verbose mode", action="store_true", dest="verbose")

# Parse the command line
args = parser.parse_args()
# Print the header
if not args.quiet : print(__doc__)

#
# Computations start here
#

# Figure out the nu samples
numin, numax = args.nu_range
nucount = args.steps
if args.includezero :
    nuvals = np.concatenate([np.array([0.0]), np.linspace(numin, numax, nucount)])
    nucount += 1
else :
    nuvals = np.linspace(numin, numax, nucount)

# Print what we're computing
if args.verbose :
    print("Outputting to", args.file.name)
    print("Fields N: ", args.N)
    print("gamma:    ", args.gamma)
    print("sigma_0:  ", args.sigma0)
    print("sigma_1:  ", args.sigma1)
    print("Method:   ", args.method)
    print("Samples:  ", args.samples)
    print("Scanning over nu from", numin, "to", numax, "in", nucount, "steps")
    if args.includezero :
        print("Including nu=0")

# Set up for the calculation
results = [[]] * nucount
parameters = [args.N, args.gamma, 0.0, args.sigma0, args.sigma1]

# Iterate over the range of nu
for i in tqdm(range(nucount)):
    parameters[2] = nuvals[i]
    if parameters[2] == 0.0 :
        # We need to treat nu=0 as a special case
        exact = signed_exact(*parameters)
        # result = integrals, errors, exact, acceptance
        results[i] = [exact, 0.0, 0.0, 0.0, exact], [0.0, 0.0, 0.0, 0.0, 0.0], exact, 0.0

    else :
        results[i] = number_density(parameters, args.samples)

# Write results to file
output = args.file
output.write(", ".join(map(str,[args.N, args.gamma, args.sigma0, args.sigma1, args.samples])) + "\n")

for i, nu in enumerate(nuvals) :
    integrals, errors, exact, acceptance = results[i]
    # Integrals and errors are lists for the different number densities:
    # min, saddle++-, saddle+--, max, signed
    # exact is the analytic signed number density
    data = [nu] + list(integrals) + list(errors) + [exact]
    output.write(", ".join(map(str,data)) + "\n")
output.close()
