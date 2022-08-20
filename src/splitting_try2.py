#!/usr/bin/env python
import numpy as np
import scipy as sp
import sympy as sy
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import ROOT
import scipy.signal
import scipy.interpolate
from scipy.interpolate import make_interp_spline, BSpline
import csv
from utilities import * # My functions: pair_dat_err, uncertainties_to_root_graph_errors
from lmfit.models import LorentzianModel, QuadraticModel, LinearModel
from uncertainties import umath
from scipy.optimize import leastsq
DATADIR = "../data"
OUTPUTDIR = "../output"

def openDat(path):
  a = pd.read_table(path, header=18, sep="\t", usecols=[0,1])
  #a = a.transpose()
  a = a.rename(index={0: 'Frequency', 1: 'Count'})
  #a = a.transpose()

  data = {'Data' : a
  }
  return data

# f_ODMR = openDat(f'{DATADIR}/ODMR/20220623-1843-21_ODMR_data_ch0_range0.dat')
# freq = f_ODMR['Data'].to_numpy().transpose()[0]   
# count = f_ODMR['Data'].to_numpy().transpose()[1]  

freq, count = np.loadtxt(f'{DATADIR}/ODMR/20220802-1356-34_ODMR_data_ch0_range0.dat', unpack=True, skiprows=19 )

# test = np.loadtxt('spectra.txt')
# freq = test[0, :]
# count = test[1, :]

def add_peak(prefix, center,sigma=7.6e+6): #, amplitude=-8.4e+6, sigma=7.6e+6):
    peak = LorentzianModel(prefix=prefix)
    pars = peak.make_params()
    amplitude = count[center]- max(count)
    pars[prefix + 'center'].set(center)
    pars[prefix + 'amplitude'].set(amplitude)
    pars[prefix + 'sigma'].set(sigma, min=0)
    return peak, pars

model = LinearModel(prefix='bkg_')
params = model.make_params(a=0, b=0, c=0)

contrast = (max(count) - min(count))/(max(count) + min(count))
height = max(count)*(1-contrast)/(1+contrast)
a = 1/1.5
height_new = max(count)*(1-contrast*a)/(1+contrast*a)
print("height", height)
print("height new", height_new)
print("contrast", contrast)
# find all the peaks that associated with the negative peaks
peaks_negative= list(scipy.signal.find_peaks(count * -1., height=-height_new ))[0]
# peaks_negative = (2.8370e+12,2.8414e+12,2.8487e+12,2.8660e+12,2.8763e+12,2.8920e+12,2.9005e+12,2.9054e+12)
print (len(peaks_negative))
for i, cen in enumerate(peaks_negative):
    peak, pars = add_peak('lz%d_' % (i+1), cen)
    print(pars)
    model = model + peak
    params.update(pars)

init = model.eval(params, x=freq)
result = model.fit(count, params, x=freq)
comps = result.eval_components()

print(result.fit_report(min_correl=0.5))

plt.plot(freq, count, label='data')
plt.plot(freq, result.best_fit, label='best fit')
for name, comp in comps.items():
    plt.plot(freq, comp, '--', label=name)
plt.legend(loc='upper right')
plt.show()
