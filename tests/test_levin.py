"""
test_levin.py

Runs tests of the Levin integration method.
"""
from math import isclose, exp
from stack.integrals.levin import LevinIntegrals


def test_integrate_I():
    rel_tol = 1e-14
    integrator = LevinIntegrals(rel_tol=1e-14)

    # Set the limits of integration
    integrator.set_limits(a=1, b=2)

    # Set the amplitude function
    integrator.set_amplitude(lambda k: 1)
    result, _ = integrator.integrate_I(ell=0, alpha=100)
    actual = 6.15687245041354e-5
    assert isclose(result, actual, rel_tol=1e-13)

    # Set the amplitude function
    integrator.set_amplitude(lambda k: k)
    result, _ = integrator.integrate_I(ell=2, alpha=200)
    actual = -2.54800394477614e-5
    assert isclose(result, actual, rel_tol=1e-13)

    # Set the amplitude function
    # This is a really challenging integral! While we don't converge within the alloted refinements,
    # we still manage to get the answer correct to incredibly good precision.
    integrator.set_amplitude(lambda k: exp(-k))
    result, _ = integrator.integrate_I(ell=5, alpha=300)
    actual = -4.10570536122891e-6
    assert isclose(result, actual, rel_tol=1e-13)
