import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import math
import sys
from astropy.io import ascii
from scipy.interpolate import interp1d
import collections
from scipy import stats
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import random
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
import argparse

clight = 2.9979E5

#np.random.seed(12345)
#random.seed(12345)


def v1(nu, Gamma, K1, omega, ecc):
    v1 = Gamma + K1 * (np.cos(omega + nu) + ecc * np.cos(omega))
    return v1

# For converting Mean anomalies to eccentric anomalies (M-->E)
def Kepler(E, M, ecc):
    E2 = (M - ecc * (E * np.cos(E) - np.sin(E))) / (1. - ecc * np.cos(E))
    eps = np.abs(E2 - E)
    if np.all(eps < 1E-10):
        return E2
    else:
        return Kepler(E2, M, ecc)

def v1v2(nu, Gamma, K1, K2, omega, ecc):
    v1 = Gamma + K1 * (np.cos(omega + nu) + ecc * np.cos(omega))
    v2 = Gamma + K2 * (np.cos(np.pi + omega + nu) + ecc * np.cos(np.pi + omega))
    return v1, v2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic SB2 spectra.")
    parser.add_argument("--K1", type=float, required=True, help="Radial velocity semi-amplitude of the primary star.")
    parser.add_argument("--K2", type=float, required=True, help="Radial velocity semi-amplitude of the secondary star.")
    parser.add_argument("--S2N", type=float, required=True, help="Signal-to-noise ratio.")
    parser.add_argument("--Q", type=float, required=True, help="Flux ratio parameter.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for spectra.")

    args = parser.parse_args()

    # Fixed parameters
    T0 = 0
    P = 473.0
    e = 0.0
    omega = 90.0 * np.pi / 180.0
    Gamma = 0.0
    Resolution = 6200.0
    Sampling = 2.0
    lamB = 3900.0
    lamR = 4600.0
    specnum = 25

    K1 = args.K1
    K2 = args.K2
    S2N = args.S2N
    Q = args.Q
    output_dir = args.output_dir

    efac = np.sqrt((1 + e) / (1 - e))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lammid = (lamB + lamR) / 2.0
    DlamRes = lammid / Resolution
    Dlam = DlamRes / Sampling

    wavegrid = np.arange(lamB, lamR, Dlam)

    stdConv = DlamRes / Dlam / np.sqrt(2 * np.log(2)) / 2.0
    kernel = Gaussian1DKernel(stddev=stdConv)

    MaskPath = '/Users/tomsayada/spectral_analysis_project/data/templates/G35000g400v10.vis.recvmac30vsini100.dat'
    MaskPath2 = '/Users/tomsayada/spectral_analysis_project/data/templates/BG22000g400v2.vis.rectvmac30vsini300.dat'

    MaskTemp = np.loadtxt(MaskPath)
    MaskTemp2 = np.loadtxt(MaskPath2)

    Waves1 = MaskTemp[:, 0] + np.random.normal(0, 1E-10, len(MaskTemp))
    Waves2 = MaskTemp2[:, 0] + np.random.normal(0, 1E-10, len(MaskTemp2))

    Mask = interp1d(Waves1, MaskTemp[:, 1], bounds_error=False, fill_value=1.0, kind='cubic')(wavegrid)
    Mask2 = interp1d(Waves2, MaskTemp2[:, 1], bounds_error=False, fill_value=1.0, kind='cubic')(wavegrid)

    Mask = convolve(Mask, kernel, normalize_kernel=True, boundary='extend')
    Mask2 = convolve(Mask2, kernel, normalize_kernel=True, boundary='extend')

    sig = 1 / S2N

    for i in range(specnum):
        HJD = random.randint(0, int(P)) + random.randint(-10, 10) / 100.0
        phase = (HJD - T0) / P - int((HJD - T0) / P)
        M = 2 * np.pi * phase
        E = Kepler(1.0, M, e)
        nu = 2.0 * np.arctan(efac * np.tan(0.5 * E))
        v1, v2 = v1v2(nu, Gamma, K1, K2, omega, e)
        Facshift1 = np.sqrt((1 + v1 / clight) / (1 - v1 / clight))
        Facshift2 = np.sqrt((1 + v2 / clight) / (1 - v2 / clight))
        Maskshift1 = interp1d(wavegrid * Facshift1, Mask, bounds_error=False, fill_value=1.0, kind='cubic')(wavegrid)
        Maskshift2 = interp1d(wavegrid * Facshift2, Mask2, bounds_error=False, fill_value=1.0, kind='cubic')(wavegrid)
        MaskSums = (1 - Q) * Maskshift1 + Q * Maskshift2
        noiseobs = MaskSums + np.random.normal(0, sig, len(wavegrid))
        obsname = os.path.join(output_dir, f'obs_{i}_V1_{v1}_V2_{v2}.txt')
        np.savetxt(obsname, np.c_[wavegrid, noiseobs])
