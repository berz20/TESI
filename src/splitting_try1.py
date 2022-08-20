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
from scipy.optimize import leastsq
DATADIR = "../data"
OUTPUTDIR = "../output"

def lorentzian( x, x0, a, gam ):
    return  a * gam**2 / ( gam**2 + ( x - x0 )**2)

def lorentzian2( x, x0, a, gam , x1, b, gam1):
    return  a * gam**2 / ( gam**2 + ( x - x0 )**2) + b * gam1**2 / ( gam1**2 + ( x - x1 )**2)

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


# f_ODMR = openDat(f'{DATADIR}/ODMR/20220623-1843-21_ODMR_data_ch0_range0.dat')
# f_ODMR = openDat(f'{DATADIR}/ODMR/20220715-1503-54_ODMR_data_ch0_range0.dat')
# f_ODMR = openDat(f'{DATADIR}/lor_try.dat')
# freq = f_ODMR['Data'].to_numpy().transpose()[0]   
# count_data = f_ODMR['Data'].to_numpy().transpose()[1] 
# freq, count_data = np.loadtxt(f'{DATADIR}/lor_try.dat', unpack=True, skiprows=19 )
freq, count_data = np.loadtxt(f'{DATADIR}/ODMR/20220802-1356-34_ODMR_data_ch0_range0.dat', unpack=True, skiprows=19 )
count = count_data/ np.max(count_data)
# with np.nditer(count, op_flags=['readwrite']) as it:
#     for x in it:
#         if x < 0.992:
#             x[...] = 0.80 * x
#         elif x < 0.993:
#             x[...] = 0.85 * x
#         elif x < 0.994:
#             x[...] = 0.87 * x
#         elif x < 0.995:
#             x[...] = 0.90 * x
#         elif x < 0.996:
#             x[...] = 0.97 * x
#         elif x < 0.997:
#             x[...] = 0.99 * x
#         elif x < 0.998:
#             x[...] = 1 * x
# df_data = pd.read_csv(f'{DATADIR}/ODMR/20220623-1843-21_ODMR_data_ch0_range0.dat', header=18, sep='\t', usecols=[0,1])
# df_data_err = 0.1/100. * df_data
# df_data.head()
# freq_arr = pair_dat_err(df_data['#frequency (Hz)'], df_data_err['#frequency (Hz)'])
# count_arr = pair_dat_err(df_data['count data (counts/s)'], df_data_err['count data (counts/s)'])
print(freq)
print(count)

generalWidth = 1

yDataLoc = count
startValues = [ np.max( count ) ]
counter = 0
while np.max( yDataLoc ) - np.min( yDataLoc ) > .02:
    print(np.max( yDataLoc ) - np.min( yDataLoc ))
    counter += 1
    if counter > 8: ### max 20 peak...emergency break to avoid infinite loop
        break
    minP = np.argmin( yDataLoc )
    minY = count[ minP ]
    x0 = freq[ minP ]
    startValues += [ x0, minY - np.max( yDataLoc ), generalWidth ]
    popt, ier = leastsq( res_multi_lorentz, startValues, args=( freq, count ) )
    yDataLoc = [ y - multi_lorentz( x, popt ) for x,y in zip( freq, count ) ]
    # print(yDataLoc)

print(popt)
testData = [ multi_lorentz(x, popt ) for x in freq ]

fig = plt.figure()
ax = fig.add_subplot( 1, 1, 1 )
ax.plot( freq, count )
ax.plot( freq, testData )
plt.show()
