#!/usr/bin/env python
import os
import numpy as np
import random
from lmfit import models
from scipy import signal
import csv
from hamiltonians import SpinHamiltonian
from itertools import permutations
import scipy.constants as cons
from variables import *

def rad_to_deg(x):
    return x / (2 * np.pi) * 360

def plot_to_output(fig, figure_name, DIR):
    filename = os.path.expanduser(f'{DIR}/{figure_name}')
    fig.set_figheight(6)
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

def peak_finder(x,counter,a,dist):
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

def write_file(x, x_err, y, file, DIR):
    with open(DIR+file+'.txt', mode='w') as f:
        f_writer = csv.writer(f, delimiter= '\t', quotechar='#', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(['x', 'x_err', 'y'])

        for i in range(0, len(x)):
            f_writer.writerow([x[i], x_err[i], y[i]])

def write_field(name,x, x_err, y, y_err, z, z_err, file, DIR):
    with open(DIR+file+'.txt', mode='a', newline='') as f:
        f_writer = csv.writer(f, delimiter= '\t', quotechar='#', quoting=csv.QUOTE_MINIMAL)
        # f_writer.writerow(['x', 'x_err', 'y', 'y_err', 'z', 'z_err'])
        f_writer.writerow([name,x*1e3, x_err*1e3, y*1e3, y_err*1e3, z*1e3, z_err*1e3])

def print_best_values(spec, output, f, DIR):
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
    write_file(center, sigma, amplitude, f, DIR )

ham = SpinHamiltonian()

def Deviation_plotter(B, ang):
    # B = np.linspace(0,0.033,8)
    sham = np.vectorize(ham.transitionFreqs, otypes=[np.ndarray])
    # freqs = np.array(sham(B,0,0))
    # freqs = np.array(freqs.tolist())
    # freqs_dev_90 = np.array(sham(B,90,0))
    # freqs_dev_90 = np.array(freqs_dev_90.tolist())
    freqs_dev = np.array(sham(B,ang,0))
    freqs_dev = np.array(freqs_dev.tolist())
    return freqs_dev[:,2]

def Deviation_plotter_plus(B, ang):
    # B = np.linspace(0,0.033,8)
    sham = np.vectorize(ham.transitionFreqs, otypes=[np.ndarray])
    # freqs = np.array(sham(B,0,0))
    # freqs = np.array(freqs.tolist())
    # freqs_dev_90 = np.array(sham(B,90,0))
    # freqs_dev_90 = np.array(freqs_dev_90.tolist())
    freqs_dev = np.array(sham(B,ang,0))
    freqs_dev = np.array(freqs_dev.tolist())
    return freqs_dev[:,0]

def Deviation_plotter_minus(B, ang):
    # B = np.linspace(0,0.033,8)
    sham = np.vectorize(ham.transitionFreqs, otypes=[np.ndarray])
    # freqs = np.array(sham(B,0,0))
    # freqs = np.array(freqs.tolist())
    # freqs_dev_90 = np.array(sham(B,90,0))
    # freqs_dev_90 = np.array(freqs_dev_90.tolist())
    freqs_dev = np.array(sham(B,ang,0))
    freqs_dev = np.array(freqs_dev.tolist())
    return freqs_dev[:,1]

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

def complete_perm(arr):
    perm_list = list(permutations(arr))
    perm_sign = list(permutations([-1,-1,1,1]))
    perm = []
    
    for i in range(0,len(perm_list)):
        for j in range(0,len(perm_sign)):
            perm.append(list(np.multiply(perm_list[i],perm_sign[j])))
    return perm

def B_calc(file,B_arr,peaks,peaks_err):
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

    B = wid/(2*gamma)  # [T]
    B_err = wid_err/(2*gamma)  # [T]

    # print(mu_b_t)

    # print(h_t)

    # print(g)

    print_array("Resonance ODMR:", "Resonance", False, B*1e+3, B_err*1e+3, "mT", [])
    if len(wid)<3:
        B_list = B
        left_side_3 = np.array([-1*u1, -1*u2, 1*u3])
        right_side_3 = [B_list[0],B_list[0],B_list[0]]
        B_xyz_t = np.linalg.inv(left_side_3).dot(right_side_3)
        left_side_3_err = np.array([u1, u2, u3])
        right_side_3_err = [B_err[0],B_err[0],B_err[0]]
        B_xyz_err_t = np.linalg.inv(left_side_3_err).dot(right_side_3_err)
        B_xyz_module = 0
        B_xyz_module_err = 0
        for j in range(0, len(B_xyz_t)):
            B_xyz_module = B_xyz_module + B_xyz_t[j]**2
            B_xyz_module_err = B_xyz_module_err + B_xyz_err_t[j]**2
        B_xyz_module_t = np.sqrt(B_xyz_module)
        B_xyz_module_err_t = np.sqrt(B_xyz_module_err)

    if len(wid)>2:
        B_list = complete_perm(B)
        perm = complete_perm([1,2,3,4])
        min = 100 # percentage difference
        for i in range(0,len(B_list)):
            left_side_3 = np.array([u1, u2, u3])
            right_side_3 = [B_list[i][0],B_list[i][1],B_list[i][2]]
            B_xyz = np.linalg.inv(left_side_3).dot(right_side_3)
            left_side_3_err = np.array([u1, u2, u3])
            right_side_3_err = [B_err[0],B_err[1],B_err[2]]
            B_xyz_err = np.linalg.inv(left_side_3_err).dot(right_side_3_err)
            B_xyz_module = 0
            B_xyz_module_err = 0
            for j in range(0, len(B_xyz)):
                B_xyz_module = B_xyz_module + B_xyz[j]**2
                B_xyz_module_err = B_xyz_module_err + B_xyz_err[j]**2
            B_xyz_module = np.sqrt(B_xyz_module)
            B_xyz_module_err = np.sqrt(B_xyz_module_err)
            left_side_4 = u4
            right_side_4 = B_list[i][3]
            # print('permutation ', B_list[i])
            check =np.dot(B_xyz,left_side_4) 
            min_v = 2*np.abs(check - right_side_4)/(check + right_side_4)*100
            # if np.abs(min_v) < np.abs(min):
            #     min = min_v
            #     B_xyz_t = B_xyz
            #     print('B_check',B_xyz_t)
            #     B_xyz_err_t = B_xyz_err
            #     B_xyz_module_t = B_xyz_module
            #     B_xyz_module_err_t = B_xyz_module_err
            #     print('permutation',perm[i])
            if perm[i] == [-3,-2,4,1] :
                min = min_v
                B_xyz_t = B_xyz
                print('B_check',B_xyz_t)
                B_xyz_err_t = B_xyz_err
                B_xyz_module_t = B_xyz_module
                B_xyz_module_err_t = B_xyz_module_err
                print('permutation',perm[i])

    print_array("Magnetic Field Components:", "B", True, B_xyz_t*1e+3, B_xyz_err_t*1e3, "mT", index_output)
    if len(wid)>2:
        write_field(file,B_xyz_t[0],B_xyz_err_t[0],B_xyz_t[1],B_xyz_err_t[1],B_xyz_t[2],B_xyz_err_t[2], 'total_B','../data/')
    else:
        write_field(file,B_xyz_t[0],B_xyz_err_t[0],B_xyz_t[1],B_xyz_err_t[1],B_xyz_t[2],B_xyz_err_t[2], 'normal_B','../data/')
    print("")
    print("Magnetic Field Module:")
    print("|B| :", B_xyz_module_t*1000, "mT")
        # check_B(B_xyz,right_side_3)
    # print('min', min)
    B_arr = np.append(B_arr,B_xyz_module_t*1000)
    B_str = "["+str("{:.2f}".format(B_xyz_module_t*1000))+"$\pm$"+str("{:.2f}".format(B_xyz_module_err_t*1000))+"]"+" mT" 
    return B_xyz_t, B_xyz_err_t, B_str, B_arr  

