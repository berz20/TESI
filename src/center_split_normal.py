#!/usr/bin/env python
import os
import math
import random
import numpy as np
import scipy as sp
import sympy as sy
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import ROOT
import csv
from utilities import * # My functions: pair_dat_err, uncertainties_to_root_graph_errors
from uncertainties import umath
from scipy.optimize import leastsq, curve_fit
from scipy import optimize, signal, interpolate
from lmfit import models
import csv
import scipy.constants as cons
DATADIR = "../data"
ODMR = "../data/ODMR/"
OUTPUTDIR = "../output/"
OUTIMGDIR = "../output/prova/"
OUTTXTDIR = "../output/txt/"
OUTTXTDIRPROVA = "../output/prova/"
index_output = ["x", "y", "z"]  # z direction confocal
u1 = np.sqrt(2/3)*np.array([0, 1, 1/np.sqrt(2)])
u2 = np.sqrt(2/3)*np.array([0, -1, 1/np.sqrt(2)])
u3 = np.sqrt(2/3)*np.array([1, 0, -1/np.sqrt(2)])
u4 = np.sqrt(2/3)*np.array([-1, 0, -1/np.sqrt(2)])

B_arr = np.array([])  
B_arr_err = np.array([])  
peak_ext_up = np.array([]) 
peak_ext_down = np.array([])  
peak_ext_up_err = np.array([])  
peak_ext_down_err = np.array([])  

# ignore used to produce images for blog
def plot_to_output(fig, figure_name):
    filename = os.path.expanduser(f'{OUTIMGDIR}/{figure_name}')
    fig.set_figheight(8)
    fig.set_figwidth(8)

    fig.savefig(filename)
    return filename

def generate_model(spec):
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    y_min = np.min(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel','InvLorentzianModel', 'VoigtModel']: # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*(y_max-y_min))
            model.set_param_hint('amplitude', min=1e-6)
            # default guess is horrible!! do not use guess()
            default_params = {
                prefix+'center': x_min + x_range * random.random(),
                prefix+'height': y_max * random.random(),
                prefix+'sigma': x_range * random.random()
            }
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params

def update_spec_from_peaks(spec,peak_indicies, pw=(0, 150), **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = np.max(x) - np.min(x)
    # np.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), range(0,len(peak_indicies))):
        spec['model'].append({'type': 'InvLorentzianModel'})
        model = spec['model'][model_indicie]
        if model['type'] in ['GaussianModel', 'LorentzianModel','InvLorentzianModel', 'VoigtModel']:
            params = {
                'height': max(y)-y[peak_indicie],
                'sigma': x_range / len(x) * np.min(pw),
                'center': x[peak_indicie]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
    return peak_indicies

def peak_finder(x,counter):
    contrast = -min(x)
    height = (min(x)-max(x))*(1-contrast)/(1+contrast)
    height_new = min(x)*(1-contrast*a[counter])/(1+contrast*a[counter])
    print('Contrast:', contrast, '\nHeight:', height, '\nNew height:', height_new)
    print("")
    peak_indicies, properties = signal.find_peaks(x * -1. , height = -height_new, distance=dist)
    print('Number of peaks found:',len(peak_indicies))
    return peak_indicies

def normalize(x):
    x = x / max(x)
    arr = np.sort(x)[::-1]
    arr = np.resize(arr,60)
    x = x - np.mean(arr)
    return x

def write_file(x, x_err, y, file):
    with open(OUTTXTDIRPROVA+file+'.txt', mode='w') as f:
        f_writer = csv.writer(f, delimiter= '\t', quotechar='#', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(['x', 'x_err', 'y'])

        for i in range(0, len(x)):
            f_writer.writerow([x[i], x_err[i], y[i]])

def print_best_values(spec, output):
    sigma = []
    center = []
    amplitude = []
    model_params = {
        'GaussianModel':   ['amplitude', 'sigma'],
        'LorentzianModel': ['amplitude', 'sigma'],
        'InvLorentzianModel': ['amplitude', 'sigma'],
        'VoigtModel':      ['amplitude', 'sigma', 'gamma']
    }
    best_values = output.best_values
    print('center\tmodel\tamplitude\tsigma')
    for i, model in enumerate(spec['model']):
        prefix = f'm{i}_'
        values = ',\t'.join(f'{best_values[prefix+param]:8.5f}' for param in model_params[model["type"]])
        sigma = np.append(sigma,best_values[prefix+'sigma'])
        center = np.append(center,best_values[prefix+'center'])
        amplitude = np.append(amplitude,best_values[prefix+'amplitude'])
        print(f'[{best_values[prefix+"center"]:3.3f}]\t{model["type"]:16}:\t{values}')
    write_file(center,sigma,amplitude,f)
     
def splitting(file, counter):
    # file = '20220801-1527-34'
    # file = '20220802-1101-15'
    # x, y = np.loadtxt(f'{DATADIR}/ODMR/'+file+'_ODMR_data_ch0_range0.dat', unpack=True, skiprows=19 )
    ft = ODMR+file+'_ODMR_data_ch0_range0.dat'
    x, y = np.loadtxt(ft, unpack=True, skiprows=19 )
    # x, y = np.loadtxt('../data/lor_try.dat', unpack=True, skiprows=19 )
    print('######################################')
    print('File:',file)
    print("")
    y = normalize(y)
    x = x / 1e+9

    # for i in range(0,len(y)):
        # if i > len(y)*98/100:
        #     y[i]=y[i]-0.0001
        # if i > len(y)*92/100:
        #     y[i]=y[i]+0.0006
        # elif i > len(y)*82/100:
        #     y[i]=y[i]+0.0005
        # elif i > len(y)*70/100:
        #     y[i]=y[i]+0.0004
        # elif i > len(y)*60/100:
        #     y[i]=y[i]+0.0003
        # elif i > len(y)*45/100:
        #     y[i]=y[i]+0.0007
        # elif i > len(y)*35/100:
        #     y[i]=y[i]+0.0005
        # elif i > len(y)*25/100:
        #     y[i]=y[i]+0.0001
        
    spec = {
        'x': x,
        'y': y,
        'model': []
    }
    spec.update
    peaks_found = update_spec_from_peaks(spec,peak_finder(y,counter), pw)
    
    model, params = generate_model(spec)
    output = model.fit(spec['y'], params, x=spec['x'])

    components = output.eval_components(x=spec['x'])
    sum = 0
    for i, model in enumerate(spec['model']):
        sum = sum + components[f'm{i}_']
    
    b_str ,p_eu,p_ed= analyze(file,counter) 
    y_eu = spec['y'][peaks_found[-1]]
    y_ed = spec['y'][peaks_found[0]]
    # fig = output.plot()   
    # fig, ax = plt.subplots()
    ax.plot(spec['x'], sum + 0.05*counter, c='orange')
    ax.plot(p_eu,y_eu + 0.05*counter, marker='.',color='black') 
    ax.plot(p_ed,y_ed + 0.05*counter, marker='.',color='black') 
    ax.scatter(spec['x'], spec['y'] + 0.05*counter, s=4, label=b_str)
    ax.axes.yaxis.set_visible(False)
    ax.set_xlabel('[GHz]')
    # plot_to_output(fig, 'splitting_normal.pdf')
    
    print_best_values(spec, output)
    print("")

def print_array(title, word, bool, array, array_err, unit, index):
    print("")
    print(title)
    if bool:
        for i in range(0, len(array)):
            print(word, index[i], ": [", array[i], "+-", array_err[i], "]", unit)
    else:
        for i in range(0, len(array)):
            print(word, i, ": [", array[i], "+-", array_err[i], "]", unit)


def diff_peaks(peaks, freq):
    freq_peaks = freq[peaks]
    if ((len(peaks) % 2) == 0):
        diff = np.array([])
        cen = np.array([])
        print("\nPeaks positions:")
        for i in range(0, int(len(peaks)/2)):
            diff = np.append(diff, freq_peaks[len(peaks)-1-i] - freq_peaks[i])
            cen = np.append(cen, (freq_peaks[len(peaks)-1-i] + freq_peaks[i])/2)
            print("Resonance", i, ":", freq_peaks[i], "GHz ||", freq_peaks[len(peaks)-1-i], "GHz")
        return diff, cen

def analyze(file,counter):
    peaks, peaks_err, peaks_amp = np.loadtxt(OUTTXTDIR+file+'.txt', unpack=True, skiprows=1)

    global B_arr   
    global B_arr_err   
    global peak_ext_up  
    global peak_ext_down   
    global peak_ext_up_err   
    global peak_ext_down_err   
    peak_ext_up = np.append(peak_ext_up,peaks[-1])
    peak_ext_down = np.append(peak_ext_down,peaks[0])
    peak_ext_up_err = np.append(peak_ext_up_err,peaks_err[-1])
    peak_ext_down_err = np.append(peak_ext_down_err,peaks_err[0])
    
    if (((len(peaks) % 2) & (len(peaks) < 9)) == 0):
        wid = np.array([])
        wid_err = np.array([])
        cen = np.array([])
        cen_err = np.array([])
        print("Peaks positions:")
        for i in range(0, int(len(peaks)/2)):
            wid = np.append(wid, peaks[len(peaks)-1-i] - peaks[i])
            wid_err = np.append(wid_err, np.sqrt(peaks_err[len(peaks)-1-i]**2 + peaks_err[i]**2))
            cen = np.append(cen, (peaks[len(peaks)-1-i] + peaks[i])/2)
            cen_err = np.append(cen_err, np.sqrt((peaks_err[len(peaks)-1-i]**2 + peaks_err[i]**2)/4))
            print("Resonance", i, ": [", peaks[i],"+-", peaks_err[i], "||",  peaks[len(peaks)-1-i],"+-", peaks_err[len(peaks)-1-i], "] GHz")

        print_array("Resonance width:", "Resonance", False, wid, wid_err, "GHz", [])

        print_array("Resonance center:", "Resonance", False, cen, cen_err, "GHz", [])

        # Magnetic Field

        mu_b = cons.physical_constants["Bohr magneton"][0]
        h = cons.physical_constants["Planck constant"][0]
        mu_b_t = cons.physical_constants["Bohr magneton"]
        h_t = cons.physical_constants["Planck constant"]
        g = 2.002

        # print(2.*g*mu_b/h)

        gamma = 28  # [GHz/T]

        # B = h*1e+3*wid/(2.*g*mu_b)

        B = wid/gamma  # [T]
        B_err = wid_err/gamma  # [T]

        # print(mu_b_t)

        # print(h_t)

        # print(g)

        print_array("Resonance ODMR:", "Resonance", False, B*1e+3, B_err*1e+3, "mT", [])

        if len(wid) > 2:

            left_side_3 = np.array([u1, -1*u2, -1*u3])

            right_side_3 = [B[0],B[1],B[2]]

            B_zee = np.linalg.inv(left_side_3).dot(right_side_3)

            B_zee_module = 0

            for i in range(0, len(B_zee)):

                B_zee_module = B_zee_module + B_zee[i]**2

            B_zee_module = np.sqrt(B_zee_module)

            if len(wid) > 3:

                left_side_4 = u4

                right_side_4 = B[3]

                if (B_zee @ left_side_4) != right_side_4:

                    B_zee = -1*B_zee

            print_array("Magnetic Field Components:", "B", True, B_zee*1e+3, B_err, "mT", index_output)

            print("")

            print("Magnetic Field Module:")

            print("|B| :", B_zee_module*1000, "mT")
        B_arr = np.append(B_arr,B[0]*1000)
        B_arr_err = np.append(B_arr_err,B_err[0]*1000)
        B_str = "["+str("{:.2f}".format(B[0]*1000))+"$\pm$"+str("{:.2f}".format(B_err[0]*1000))+"]"+" mT" 
        return B_str,peak_ext_up[counter],peak_ext_down[counter]

def center_split():
    peak_ext = np.array([])
    peak_ext_err = np.array([])
    for i in range(0,len(peak_ext_up)):
        peak_ext = np.append(peak_ext, (peak_ext_up[i]+peak_ext_down[i])/2 )
        peak_ext_err = np.append(peak_ext_err, (np.sqrt(peak_ext_up_err[i]**2+peak_ext_down_err[i]**2)/4) )
    # ax.plot(B_arr, peak_ext_up)
    # ax.plot(B_arr, peak_ext_down)
    # ax.plot(B_arr, peak_int_up)
    # ax.plot(B_arr, peak_int_down)
    # ax.plot(B_arr, peak_ext)
    # ax.plot(B_arr, peak_int)
    def func_poly(x,a,b):
        return a*x**2+b*x+2.87
    popt_ext, pcov_ext = curve_fit(func_poly, B_arr, peak_ext, sigma= peak_ext_err)
    fit_ext = func_poly(B_arr, *popt_ext)
    # n=2
    # fit_ext = np.poly1d(np.polyfit(B_arr,peak_ext,n))
    # fit_int = np.poly1d(np.polyfit(B_arr,peak_int,n))
    # int_fit_int = np.poly1d(np.polyfit(B_arr,int_peak_int,n))
    # int_fit_ext = np.poly1d(np.polyfit(B_arr,int_peak_ext,n))

    ax.errorbar(B_arr, peak_ext_up, yerr=peak_ext_up_err,capsize=5, fmt='.', color='black', label="Resonance's peaks positions")
    ax.errorbar(B_arr, peak_ext_down, yerr=peak_ext_down_err,capsize=5, fmt='.', color='black')
    ax.plot(B_arr,fit_ext,color='black', label='poly2 fit center splitting')
    ax.axhline(2.87, c='black', linestyle='dotted', label='$2.87 \ GHz$')
    ax.legend()
    ax.set_xlabel('External magnetic field [mT]')
    ax.set_ylabel('[GHz]')
    plot_to_output(fig, 'deviation_normal.pdf')
    print("")
    print('Amplitude Factor:',a,'\nPeak Width:',pw)
    print('')

        # '20220729-1433-04',
        # '20220729-1418-49',
files = [
        '20220728-1325-15',
        '20220728-1309-54',
        '20220728-1256-39',
        '20220729-0934-41',
        '20220728-1351-20',
        '20220728-1241-44',
        '20220728-1338-08',
        '20220728-1222-32',
        '20220801-1113-48',
        '20220728-0912-42',
        '20220728-0947-12',
        '20220728-1128-20',
        '20220801-1030-48',
        ] 
        # '20220801-1053-22',
fig, ax = plt.subplots(figsize=(8, 8))
a = [23,23,23,23,19,19,16,19,23,20,20,18,53] 
# a = [12,12,23,23,23,23,19,19,16,19,20,23,20,18,53,53] 
#Amplitude
# a = 1/0.0209
#Peaks width
pw = (1.5,)
#Peaks distance
dist =  2.1
#Offset
offset = 1 
for f,i in zip(files, range(0,len(files))):
    splitting(f,i)

# ax.axvline(2.87, c='black', linestyle='dotted', label='$2.87 \ GHz$')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title='B Field', loc='upper left')
#ax2[0].legend(loc='upper left')
plt.savefig(f'{OUTIMGDIR}/splitting_normal.pdf')

fig, ax = plt.subplots(figsize=(8, 8))
center_split()
# File used 
# 20220728-0912-42
# 20220728-0947-12
# 20220728-1128-20
# 20220728-1222-32
# 20220728-1241-44
# 20220728-1256-39
# 20220728-1309-54
# 20220728-1325-15
# 20220728-1338-08
# 20220728-1351-20
# 20220729-0934-41
# 20220728-1418-49
# 20220728-1433-04
# 20220801-1030-48
# 20220801-1053-22
# 20220801-1113-48
# 20220801-1053-22
# 20220801-1053-22
# 20220801-1053-22
# 20220801-1053-22

