"""
doublebessel.py

Code to compute integrals of the form
E_l(r) = 4 pi int_{k_min}^{k_max} dk k^2 P(k) j_l(k r)^2
and
G_l(r, r') = 4 pi int_{k_min}^{k_max} dk k^2 P(k) j_l(k r) j_l(k r')
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from math import pi
from scipy.integrate import quad
from scipy.special import spherical_jn

from stack.common import Suppression
from stack.integrals.common import Integrals
from stack.integrals.levin import LevinIntegrals

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from stack import Model

# Raise errors on issues rather than printing warnings (except for underflow; we don't care)
np.seterr(all='raise')
np.seterr(under='ignore')

# These are the 2nd roots (1 full oscillation) of j_ell(x), ell ranging from 0 to 100.
j_ell_roots = [6.28318530717959, 7.72525183693771, 9.09501133047625, 10.4171185473799, 11.7049071545704,
               12.9665301727743, 14.2073924588425, 15.4312892102684, 16.6410028815122, 17.8386431992053,
               19.0258535361278, 20.2039426328117, 21.3739721811627, 22.5368170711198, 23.6932080374714,
               24.8437625975863, 25.9890079764136, 27.1293984123007, 28.2653284366424, 29.3971432134409,
               30.5251466952463, 31.6496081325095, 32.7707673241958, 33.8888388941352, 35.0040158047227,
               36.1164722674122, 37.2263661715630, 38.3338411253205, 39.4390281814521, 40.5420473054268,
               41.6430086311325, 42.7420135404888, 43.8391555961316, 44.9345213508109, 46.0281910527871,
               47.1202392630483, 48.2107353974125, 49.2997442043539, 50.3873261875962, 51.4735379810525,
               52.5584326824965, 53.6420601513630, 54.7244672752637, 55.8056982091289, 56.8857945903217,
               57.9647957326000, 59.0427388014036, 60.1196589726104, 61.1955895766203, 62.2705622293833,
               63.3446069517857, 64.4177522786267, 65.4900253582705, 66.5614520439272, 67.6320569774018,
               68.7018636660544, 69.7708945536301, 70.8391710855421, 71.9067137691272, 72.9735422293379,
               74.0396752602844, 75.1051308729959, 76.1699263397348, 77.2340782351604, 78.2976024746111,
               79.3605143497471, 80.4228285617704, 81.4845592524212, 82.5457200329279, 83.6063240110731,
               84.6663838165231, 85.7259116245535, 86.7849191782942, 87.8434178096031, 88.9014184586710,
               89.9589316924505, 91.0159677219922, 92.0725364187678, 93.1286473300498, 94.1843096934139,
               95.2395324504242, 96.2943242595563, 97.3486935084097, 98.4026483252561, 99.4561965899670,
               100.509345944361, 101.562103802008, 102.614477357522, 103.666473595382, 104.718099298299,
               105.769361055166, 106.820265268611, 107.870818162181, 108.921025787176, 109.970894029150,
               111.020428614108, 112.069635114407, 113.118518954383, 114.167085415718, 115.215339642558,
               116.263286646404]

# These are the 20th roots (10 full oscillations) of j_ell(x), ell ranging from 0 to 100.
j_ell_roots10 = [62.8318530717959, 64.3871195905574, 65.9279415029587, 67.4552844798028, 68.9700122850280,
                 70.4729011938089, 71.9646518901159, 73.4458993623277, 74.9172211938852, 76.3791445561388,
                 77.8321521433816, 79.2766872393004, 80.7131580653214, 82.1419415314487, 83.5633864869638,
                 84.9778165501387, 86.3855325817225, 87.7868148555074, 89.1819249701020, 90.5711075386380,
                 91.9545916871333, 93.3325923873311, 94.7053116458209, 96.0729395679265, 97.4356553121055,
                 98.7936279483148, 100.147017231888, 101.495974302863, 102.840642319348, 104.181157032370,
                 105.517647308672, 106.850235607102, 108.179038413532, 109.504166638623, 110.825725982250,
                 112.143817267922, 113.458536750178, 114.769976397561, 116.078224153511, 117.383364177231,
                 118.685477066389, 119.984640063283, 121.280927245956, 122.574409705572, 123.865155711251,
                 125.153230863415, 126.438698236617, 127.721618512719, 129.002050105195, 130.280049275286,
                 131.555670240634, 132.828965276989, 134.099984813519, 135.368777522207, 136.635390401775,
                 137.899868856544, 139.162256770588, 140.422596577535, 141.680929326302, 142.937294743074,
                 144.191731289762, 145.444276219200, 146.694965627286, 147.943834502282, 149.190916771453,
                 150.436245345217, 151.679852158971, 152.921768212744, 154.162023608795, 155.400647587311,
                 156.637668560296, 157.873114143779, 159.107011188435, 160.339385808706, 161.570263410532,
                 162.799668717740, 164.027625797207, 165.254158082830, 166.479288398398, 167.703038979412,
                 168.925431493915, 170.146487062391, 171.366226276774, 172.584669218624, 173.801835476511,
                 175.017744162642, 176.232413928787, 177.445862981517, 178.658109096816, 179.869169634072,
                 181.079061549503, 182.287801409022, 183.495405400590, 184.701889346074, 185.907268712623,
                 187.111558623604, 188.314773869110, 189.516928916056, 190.718037917889, 191.918114723928,
                 193.117172888352]

# These are the 100th roots (50 full oscillations) of j_ell(x), ell ranging from 0 to 100.
j_ell_roots50 = [314.159265358979, 315.726894402043, 317.291402981732, 318.852836788400, 320.411240420150,
                 321.966657418974, 323.519130305381, 325.068700611567, 326.615408913218, 328.159294860009,
                 329.700397204867, 331.238753832047, 332.774401784097, 334.307377287754, 335.837715778820,
                 337.365451926077, 338.890619654278, 340.413252166260, 341.933381964220, 343.451040870190,
                 344.966260045750, 346.479070011015, 347.989500662921, 349.497581292852, 351.003340603630,
                 352.506806725893, 354.008007233895, 355.506969160754, 357.003719013158, 358.498282785570,
                 359.990685973936, 361.480953588934, 362.969110168768, 364.455179791533, 365.939186087165,
                 367.421152249004, 368.901101044964, 370.379054828348, 371.855035548309, 373.329064759971,
                 374.801163634236, 376.271352967265, 377.739653189679, 379.206084375454, 380.670666250551,
                 382.133418201275, 383.594359282383, 385.053508224934, 386.510883443918, 387.966503045637,
                 389.420384834884, 390.872546321895, 392.323004729105, 393.771776997703, 395.218879793994,
                 396.664329515586, 398.108142297385, 399.550334017437, 400.990920302588, 402.429916533993,
                 403.867337852473, 405.303199163716, 406.737515143338, 408.170300241806, 409.601568689226,
                 411.031334499997, 412.459611477345, 413.886413217736, 415.311753115162, 416.735644365324,
                 418.158099969703, 419.579132739517, 420.998755299583, 422.416980092075, 423.833819380186,
                 425.249285251700, 426.663389622465, 428.076144239784, 429.487560685722, 430.897650380322,
                 432.306424584755, 433.713894404373, 435.120070791709, 436.524964549380, 437.928586332941,
                 439.330946653654, 440.732055881201, 442.131924246321, 443.530561843398, 444.927978632973,
                 446.324184444211, 447.719188977295, 449.113001805778, 450.505632378874, 451.897090023691,
                 453.287383947427, 454.676523239497, 456.064516873630, 457.451373709906, 458.837102496749,
                 460.221711872883]


class DoubleBessel(Integrals):
    """
    Computes double bessel integrals over the power spectrum.
    """
    filename = 'doublebessel'

    def __init__(self, model: 'Model') -> None:
        """
        Initialize the class.

        :param model: Model class we are computing integrals for.
        """
        super().__init__(model)
    
        # Initialize the Levin integrator
        self.integrator = LevinIntegrals(rel_tol=self.err_rel, abs_tol=self.err_abs, refinements=6)

    def save_data(self) -> None:
        """This class does not save any data, but does output some results for comparison with Mathematica"""
        # Construct a grid in physical space
        rvals = np.logspace(start=-3,
                            stop=2.5,
                            num=21,
                            endpoint=True)
        # Compute E on that grid
        Evals = []
        for ell in range(0, 31):
            if self.model.verbose:
                print(f"Computing E_{ell}(r)...")
            Evals.append(np.array([self.compute_E(ell, r, Suppression.RAW) for r in rvals]))
        # Save them to file
        df = pd.DataFrame([rvals] + Evals).transpose()
        df.columns = ['r'] + [f'E_{ell}(r)' for ell in range(0, 31)]
        df.to_csv(self.file_path(self.filename + '.csv'), index=False)

    def compute_E(self, ell: int, r: float, suppression: Suppression) -> float:
        """
        Computes the integral
        E_ell(r) = 4 pi int_{k_min}^{k_max} dk k^2 P(k) j_ell(k r)^2
        
        :param r: Value of r to use in the integral
        :param ell: Value of ell to use in the integral
        :param suppression: High frequency suppression method to use
        :return: Result of the integral
        """
        moments = self.model.get_moments(suppression)
        if suppression == suppression.PEAKS:
            raise ValueError(f'Bad suppression method: {suppression}')

        # Treat the special case
        if r == 0:
            if ell == 0:
                return moments.sigma0
            return 0
        
        pk = self.model.powerspectrum
        min_k = self.model.min_k
        max_k = self.model.max_k
        if suppression == suppression.SAMPLING:
            # No need to go far out into the tail of the suppressing Gaussian
            # At k = n k_0, the suppression is exp(-n^2/2)
            # At n = 6, this is ~10^-8, which seems like a good place to stop
            max_k = min(max_k, self.model.grid.sampling_cutoff * 6)

        # Construct the list of domains
        osc1 = j_ell_roots[ell] / r       # 1 oscillation of j_0(x)
        osc10 = j_ell_roots10[ell] / r    # 10 oscillations of j_0(x)
        osc50 = j_ell_roots50[ell] / r    # 50 oscillations of j_0(x)
        domains = self.generate_domains(min_k, max_k, moments.k2peak, osc1, osc10)

        # Define integration functions
        def f(k):
            """Straight function to integrate"""
            return k * k * pk(k, suppression) * spherical_jn(ell, k*r)**2

        low_osc = self.gen_low_osc(f, "F", r)

        def hi_osc(min_k: float, max_k: float) -> float:
            """Compute integrals for highly-oscillatory functions"""
            def func(k):
                """Integrand weight function (excluding Levin kernel)"""
                return k * k * pk(k, suppression)

            # Set the limits of integration
            self.integrator.set_limits(a=min_k, b=max_k)

            # Set the amplitude function
            self.integrator.set_amplitude(func)
            
            # Perform the integration
            int_result, _ = self.integrator.integrate_H(ell=ell, alpha=r)
            
            # print(f'mink={min_k}; maxk={max_k}; r={r}; l={ell}; ans={int_result};'.replace("e", "*^"))

            return int_result

        # Define selector function
        def selector(min_k: float, max_k: float) -> Callable:
            """Returns the function to use to perform integration on the given domain"""
            if max_k > osc50:  # This works amazingly well - integrand is positive definite
                return hi_osc
            return low_osc

        # Perform integration
        result = self.perform_integral(domains, selector)

        return result
