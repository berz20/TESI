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
from uncertainties import umath
from lmfit.models import LorentzianModel, QuadraticModel, LinearModel
from scipy.optimize import leastsq
DATADIR = "../data"
OUTPUTDIR = "../output"

def lorentzian( x, x0, a, gam ):
    return  a * gam**2 / ( gam**2 + ( x - x0 )**2)

def multi_lorentz( x, params ):
    off = params[0]
    paramsRest = params[1:]
    assert not ( len( paramsRest ) % 3 )
    return off + sum( [ lorentzian( x, *paramsRest[ i : i+3 ] ) for i in range( 0, len( paramsRest ), 3 ) ] )

def res_multi_lorentz( params, xData, yData ):
    diff = [ multi_lorentz( x, params ) - y for x, y in zip( xData, yData ) ]
    return diff

def openDat(path):
  a = pd.read_table(path, header=18, sep="\t", usecols=[0,1])
  #a = a.transpose()
  a = a.rename(index={0: 'Frequency', 1: 'Count'})
  #a = a.transpose()

  data = {'Data' : a
  }
  return data



freq, count = np.loadtxt(f'{DATADIR}/ODMR/20220801-1544-10_ODMR_data_ch0_range0.dat', unpack=True, skiprows=19 )

def add_peak(prefix, center,sigma,amplitude): #, amplitude=-8.4e+6, sigma=7.6e+6):
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
a = 1/4
height_new = max(count)*(1-contrast*a)/(1+contrast*a)
print("height", height)
print("height new", height_new)
print("contrast", contrast)
# find all the peaks that associated with the negative peaks
peaks_negative= list(scipy.signal.find_peaks(count * -1., height=-height_new, distance=11 ))[0]
# peaks_negative = (2.8370e+12,2.8414e+12,2.8487e+12,2.8660e+12,2.8763e+12,2.8920e+12,2.9005e+12,2.9054e+12)
sigma_peak = scipy.signal.peak_widths(freq,peaks_negative)
freq_peak = freq[peaks_negative]
ampli_peak = count[peaks_negative]-max(count)
for i in range(0, len(peaks_negative)):
    peak, pars = add_peak('lz%d_' % (i+1), freq_peak[i],sigma_peak[i], ampli_peak[i])
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
# freq, count_data = np.loadtxt(f'{DATADIR}/lor_try.dat', unpack=True, skiprows=19 )
# freq, count_data = np.loadtxt(f'{DATADIR}/ODMR/20220801-1544-10_ODMR_data_ch0_range0.dat', unpack=True, skiprows=19 )
# count = count_data/ np.max(count_data)
# print(freq)
# print(count)
# def add_peak(prefix, center,sigma=7.6e+6): #, amplitude=-8.4e+6, sigma=7.6e+6):
#     peak = LorentzianModel(prefix=prefix)
#     pars = peak.make_params()
#     amplitude = count[center]- max(count)
#     pars[prefix + 'center'].set(center)
#     pars[prefix + 'amplitude'].set(amplitude)
#     pars[prefix + 'sigma'].set(sigma, min=0)
#     return peak, pars
# peak_widths=(0.006e+9, 0.01e+9)
# peak_indicies = list(scipy.signal.find_peaks_cwt(count, peak_widths))
# peak = freq[peak_indicies]
# print(len(peak))
# contrast = (max(count) - min(count))/(max(count) + min(count))
# height = max(count)*(1-contrast)/(1+contrast)
# a = 1/2.5
# height_new = max(count)*(1-contrast*a)/(1+contrast*a)
# print("height", height)
# print("height new", height_new)
# print("contrast", contrast)
# peaks_negative, properties = scipy.signal.find_peaks(count * -1., height=-height_new, distance=11)
# print(len(peaks_negative))
    
# for i in range(0, len(peaks_negative)):
#     peak, pars = add_peak('lz%d_' % (i+1), freq[peaks_negative[i]])
#     print(pars)
#     model = model + peak
#     params.update(pars)

# init = model.eval(params, x=freq)
# result = model.fit(count, params, x=freq)
# comps = result.eval_components()

# print(result.fit_report(min_correl=0.5))

# plt.plot(freq, count, label='data')
# plt.plot(freq, result.best_fit, label='best fit')
# for name, comp in comps.items():
#     plt.plot(freq, comp, '--', label=name)
# plt.legend(loc='upper right')
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot( 1, 1, 1 )
# ax.plot( freq, count )
# ax.plot( freq, testData )
# plt.show()
