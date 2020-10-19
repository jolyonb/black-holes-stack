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

from typing import TYPE_CHECKING, Callable, Tuple

if TYPE_CHECKING:
    from stack import Model

# Raise errors on issues rather than printing warnings (except for underflow; we don't care)
np.seterr(all='raise')
np.seterr(under='ignore')

# These are the 1st roots (half oscillation) of j_ell(x), ell ranging from 0 to 100.
j_ell_roots_half = [3.14159265358979, 4.49340945790906, 5.76345919689455, 6.98793200050051, 8.18256145257149,
                    9.35581211104502, 10.5128354080982, 11.6570321925211, 12.7907817119779, 13.9158226105099,
                    15.0334693037494, 16.1447429423013, 17.2504547841260, 18.3512614959473, 19.4477031080947,
                    20.5402298250482, 21.6292214365904, 22.7150016749722, 23.7978490341283, 24.8780050582072,
                    25.9556807850401, 27.0310618213500, 28.1043123876971, 29.1755785769190, 30.2449910046155,
                    31.3126669843224, 32.3787123272001, 33.4432228422462, 34.5062855955485, 35.5679799740761,
                    36.6283785897134, 37.6875480518021, 38.7455496307433, 39.8024398307933, 40.8582708867284,
                    41.9130911963446, 42.9669456985971, 44.0198762054710, 45.0719216942892, 46.1231185660503,
                    47.1735008744784, 48.2231005297262, 49.2719474800637, 50.3200698743775, 51.3674942078915,
                    52.4142454531694, 53.4603471781674, 54.5058216528616, 55.5506899457670, 56.5949720114892,
                    57.6386867703026, 58.6818521806208, 59.7244853051161, 60.7666023711523, 61.8082188261149,
                    62.8493493881528, 63.8900080927861, 64.9302083357819, 65.9699629126566, 67.0092840551198,
                    68.0481834647453, 69.0866723441195, 70.1247614256945, 71.1624609985483, 72.1997809332331,
                    73.2367307048754, 74.2733194146759, 75.3095558099414, 76.3454483027680, 77.3810049874864,
                    78.4162336569653, 79.4511418178646, 80.4857367049176, 81.5200252943178, 82.5540143162767,
                    83.5877102668149, 84.6211194188415, 85.6542478325752, 86.6871013653521, 87.7196856808646,
                    88.7520062578707, 89.7840683984109, 90.8158772355643, 91.8474377407769, 92.8787547307896,
                    93.9098328741920, 94.9406766976264, 95.9712905916646, 97.0016788163785, 98.0318455066226,
                    99.0617946770479, 100.091530226862, 101.121055944352, 102.150375511184, 103.179492506492,
                    104.208410410772, 105.237132609581, 106.265662397074, 107.294002979366, 108.322157477736,
                    109.350128931692, 110.377920301894]

# These are the 2nd roots (1 full oscillation) of j_ell(x), ell ranging from 0 to 100.
j_ell_roots1 = [6.28318530717959, 7.72525183693771, 9.09501133047625, 10.4171185473799, 11.7049071545704,
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
        """This class does not save any data"""
        pass

    def compute_E(self, ell: int, r: float, suppression: Suppression) -> Tuple[float, float]:
        """
        Computes the integral
        E_ell(r) = 4 pi int_{k_min}^{k_max} dk k^2 P(k) j_ell(k r)^2
        
        :param r: Value of r to use in the integral
        :param ell: Value of ell to use in the integral
        :param suppression: High frequency suppression method to use
        :return: Result of the integral
        """
        moments = self.model.get_moments(suppression)
        suppression_factor = None
        if suppression == Suppression.PEAKS:
            raise ValueError(f'Bad suppression method: {suppression}')
        elif suppression == Suppression.RAW:
            suppression_factor = self.model.grid.sampling_cutoff

        # Treat the special case
        if r == 0:
            if ell == 0:
                return moments.sigma0, 0
            return 0, 0
        
        pk = self.model.powerspectrum
        min_k = self.model.min_k
        max_k = self.model.max_k
        if suppression == suppression.SAMPLING:
            # No need to go far out into the tail of the suppressing Gaussian
            # At k = n k_0, the suppression is exp(-n^2/2)
            # At n = 6, this is ~10^-8, which seems like a good place to stop
            max_k = min(max_k, self.model.grid.sampling_cutoff * 6)

        # Construct the list of domains
        osc1 = j_ell_roots1[ell] / r       # 1 oscillation of j_0(x)
        osc10 = j_ell_roots10[ell] / r    # 10 oscillations of j_0(x)
        osc50 = j_ell_roots50[ell] / r    # 50 oscillations of j_0(x)
        domains = self.generate_domains(min_k, max_k, moments.k2peak, osc1, osc10, suppression_factor)

        # Define integration functions
        def f(k):
            """Straight function to integrate"""
            return k * k * pk(k, suppression) * spherical_jn(ell, k*r)**2

        low_osc = self.gen_low_osc(f, "E", r)

        def hi_osc(min_k: float, max_k: float) -> Tuple[float, float]:
            """Compute integrals for highly-oscillatory functions"""
            def func(k):
                """Integrand weight function (excluding Levin kernel)"""
                return k * k * pk(k, suppression)

            # Set the limits of integration
            self.integrator.set_limits(a=min_k, b=max_k)

            # Set the amplitude function
            self.integrator.set_amplitude(func)
            
            # Perform the integration
            int_result, err_est = self.integrator.integrate_H(ell=ell, alpha=r)
            
            # print(f'mink={min_k}; maxk={max_k}; r={r}; l={ell}; ans={int_result};'.replace("e", "*^"))

            return int_result, err_est

        # Define selector function
        def selector(min_k: float, max_k: float) -> Callable:
            """Returns the function to use to perform integration on the given domain"""
            if max_k < osc50 or (max_k - min_k) * r < 2 * pi:
                # This works amazingly well - integrand is positive definite
                return low_osc
            return hi_osc

        # Perform integration
        result, err = self.perform_integral(domains, selector)

        return result, err

    def compute_G(self, ell: int, r: float, rp: float, suppression: Suppression) -> Tuple[float, float]:
        """
        Computes the integral
        G_ell(r, r') = 4 pi int_{k_min}^{k_max} dk k^2 P(k) j_ell(k r) j_ell(k r').
        If r = r', then compute_E is invoked instead.

        :param r: Value of r to use in the integral
        :param rp: Value of r' to use in the integral
        :param ell: Value of ell to use in the integral
        :param suppression: High frequency suppression method to use
        :return: Result of the integral
        """
        # Check for coincidence limit
        if r == rp:
            return self.compute_E(ell, r, suppression)

        rmin = min(r, rp)
        rmax = max(r, rp)
        
        # Treat the special case
        if rmin == 0:
            if ell == 0:
                return self.model.singlebessel.compute_C(rmax, suppression)
            return 0, 0

        moments = self.model.get_moments(suppression)
        suppression_factor = None
        if suppression == Suppression.PEAKS:
            raise ValueError(f'Bad suppression method: {suppression}')
        elif suppression == Suppression.RAW:
            suppression_factor = self.model.grid.sampling_cutoff

        pk = self.model.powerspectrum
        min_k = self.model.min_k
        max_k = self.model.max_k
        if suppression == suppression.SAMPLING:
            # No need to go far out into the tail of the suppressing Gaussian
            # At k = n k_0, the suppression is exp(-n^2/2)
            # At n = 6, this is ~10^-8, which seems like a good place to stop
            max_k = min(max_k, self.model.grid.sampling_cutoff * 6)
    
        # Construct the list of domains
        oschalfmin = j_ell_roots_half[ell] / rmin  # half oscillation of j_ell(x)
        osc1min = j_ell_roots1[ell] / rmin  # 1 oscillation of j_ell(x)
        osc10min = j_ell_roots10[ell] / rmin  # 10 oscillations of j_ell(x)
        oschalfmax = j_ell_roots_half[ell] / rmax  # half oscillation of j_ell(x)
        osc1max = j_ell_roots1[ell] / rmax  # 1 oscillation of j_ell(x)
        osc10max = j_ell_roots10[ell] / rmax  # 10 oscillations of j_ell(x)
        # Note that the min values will be larger than the max values
        domains = self.generate_domains(min_k, max_k, moments.k2peak, osc1min, osc10min, suppression_factor)

        # Define integration functions
        def f(k):
            """Straight function to integrate"""
            return k * k * pk(k, suppression) * spherical_jn(ell, k * rmin) * spherical_jn(ell, k * rmax)
    
        low_osc = self.gen_low_osc(f, "G", r)
    
        def hi_osc(min_k: float, max_k: float) -> Tuple[float, float]:
            """Compute integrals for highly-oscillatory functions"""
        
            def func(k):
                """Integrand weight function (excluding Levin kernel)"""
                return k * k * pk(k, suppression)
        
            # Set the limits of integration
            self.integrator.set_limits(a=min_k, b=max_k)
        
            # Set the amplitude function
            self.integrator.set_amplitude(func)
        
            # Perform the integration
            int_result, err_est = self.integrator.integrate_K(ell=ell, alpha=rmin, beta=rmax)
        
            # print(f"(* hi_osc *)")
            # print(f'mink={min_k}; maxk={max_k}; r={r}; rp={rp}; l={ell}; ans={int_result};'.replace("e", "*^"))
        
            return int_result, err_est
    
        def ell0_osc(min_k: float, max_k: float) -> Tuple[float, float]:
            """Compute integrals for ell = 0 with r_min small enough to be slowly-oscillating"""
            def f_sin(k):
                """Define function to integrate"""
                return k * pk(k, suppression) * spherical_jn(0, k * rmin)

            # Compute the integral using sine-weighted quadrature
            int_result = quad(f_sin, min_k, max_k, weight='sin', wvar=rmax,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)

            # Check for any warnings
            if len(int_result) == 4 and 'roundoff error is detected' not in int_result[-1]:
                print(f"Warning when integrating (sine) G_ell(r, r') at ell = {ell}, r = {r}, r' = {rp}")
                print(int_result[-1])
                
            result = int_result[0] / rmax
            err_est = int_result[1] / rmax

            # print(f"(* ell0_osc *)")
            # print(f'mink={min_k}; maxk={max_k}; r={r}; rp={rp}; l={ell}; ans={int_result};'.replace("e", "*^"))

            return result, err_est

        def ell1_osc(min_k: float, max_k: float) -> Tuple[float, float]:
            """Compute integrals for ell = 1 with r_min small enough to be slowly-oscillating"""
            def f_sin(k):
                return pk(k, suppression) * spherical_jn(1, k * rmin)

            def f_cos(k):
                return k * pk(k, suppression) * spherical_jn(1, k * rmin)

            # Perform the integrations using sin and cos quadrature
            sin_result = quad(f_sin, min_k, max_k, weight='sin', wvar=rmax,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
            cos_result = quad(f_cos, min_k, max_k, weight='cos', wvar=rmax,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)

            # Check for any warnings
            if len(sin_result) == 4 and 'roundoff error is detected' not in sin_result[-1]:
                print(f"Warning when integrating (sine) G_ell(r, r') at ell = {ell}, r = {r}, r' = {rp}")
                print(sin_result[-1])
            if len(cos_result) == 4 and 'roundoff error is detected' not in cos_result[-1]:
                print(f"Warning when integrating (cosine) G_ell(r, r') at ell = {ell}, r = {r}, r' = {rp}")
                print(cos_result[-1])

            # Construct the result
            result = sin_result[0] / (r*r) - cos_result[0] / r
            err_est = sin_result[1] / (r*r) + cos_result[1] / r

            # print(f"(* ell1_osc *)")
            # print(f'mink={min_k}; maxk={max_k}; r={r}; rp={rp}; l={ell}; ans={int_result};'.replace("e", "*^"))

            return result, err_est

        def levin_osc(min_k: float, max_k: float) -> Tuple[float, float]:
            """Compute integrals where one of two bessels are highly oscillatory"""
    
            def func(k):
                """Integrand weight function (excluding Levin kernel)"""
                return k * k * pk(k, suppression) * spherical_jn(ell, k * rmin)
    
            # Set the limits of integration
            self.integrator.set_limits(a=min_k, b=max_k)
    
            # Set the amplitude function
            self.integrator.set_amplitude(func)
    
            # Perform the integration
            int_result, err_est = self.integrator.integrate_I(ell=ell, alpha=rmax)
    
            # print(f"(* levin_osc *)")
            # print(f'mink={min_k}; maxk={max_k}; r={r}; rp={rp}; l={ell}; ans={int_result};'.replace("e", "*^"))
    
            return int_result, err_est

        # Define selector function
        def selector(min_k: float, max_k: float) -> Callable:
            """Returns the function to use to perform integration on the given domain"""
            # Choose from the following regimes
            if max_k < osc10max or (max_k - min_k) * rmax < 2 * pi:
                # Each spherical bessel function has less than 10 oscillations
                # Use direct integration
                return low_osc
            elif max_k < osc1min or (max_k - min_k) * rmin < 2 * pi:
                # The slower oscillations don't have time to complete a full oscillation
                # Treat this like a single-bessel integral with weight k**2*P(k)*j_ell(k*rmin)
                # For ell = 0 or 1, use sine- and cosine-weighted integration
                # For higher ell, use Levin integration (single-Bessel)
                if ell == 0:
                    return ell0_osc
                elif ell == 1:
                    return ell1_osc
                return levin_osc
            # Otherwise, use Levin integration (double-Bessel)
            return hi_osc
    
        # Perform integration
        result, err = self.perform_integral(domains, selector)
    
        return result, err
