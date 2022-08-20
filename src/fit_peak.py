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
from scipy.optimize import leastsq
from scipy import optimize, signal, interpolate
from lmfit import models
DATADIR = "../data"
OUTPUTDIR = "../output"

# ignore used to produce images for blog
image_dir = OUTPUTDIR 
def plot_to_output(fig, figure_name):
    filename = os.path.expanduser(f'{image_dir}/{figure_name}')
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
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']: # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
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

def update_spec_from_peaks(spec, model_indicies, peak_widths=(0, 150), **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = np.max(x) - np.min(x)
    # Mine peak finding

    contrast = (max(y) - min(y))/(max(y) + min(y))
    height = max(y)*(1-contrast)/(1+contrast)
    a = 1/2.7
    height_new = max(y)*(1-contrast*a)/(1+contrast*a)
    print("height", height)
    print("height new", height_new)
    print("contrast", contrast)
    peak_indicies, properties = signal.find_peaks(y , height=height_new)
    # peak_indicies = signal.find_peaks_cwt(y, peak_widths)
    np.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
        model = spec['model'][model_indicie]
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                'height': y[peak_indicie],
                'sigma': x_range / len(x) * np.min(peak_widths),
                'center': x[peak_indicie]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
    return peak_indicies

x, y = np.loadtxt(f'{DATADIR}/ODMR/20220801-1145-51_ODMR_data_ch0_range0.dat', unpack=True, skiprows=19 )
# x, y = np.loadtxt('../data/lor_try.dat', unpack=True, skiprows=19 )
y = 1 - y / max(y)  
sigma_min = 2.2e+6 
sigma_max = 4.15e+6
ampli_min = 0.1e+5
ampli_max = 5.e+5
spec = {
    'x': x,
    'y': y,
    'model': [
    #     {'type': 'LorentzianModel',
    #      'help': 
    #             {'sigma': {'min' : sigma_min,'max' : sigma_max},
    #              'amplitude': {'min' : ampli_min,'max' : ampli_max}}
    #     },
    #     {'type': 'LorentzianModel',
    #      'help': 
    #             {'sigma': {'min' : sigma_min,'max' : sigma_max},
    #              'amplitude': {'min' : ampli_min,'max' : ampli_max}}
    #     },
    #     {'type': 'LorentzianModel',
    #      'help': 
    #             {'sigma': {'min' : sigma_min,'max' : sigma_max},
    #              'amplitude': {'min' : ampli_min,'max' : ampli_max}}
    #     },
    #     {'type': 'LorentzianModel',
    #      'help': 
    #             {'sigma': {'min' : sigma_min,'max' : sigma_max},
    #              'amplitude': {'min' : ampli_min,'max' : ampli_max}}
    #     },
    #     {'type': 'LorentzianModel',
    #      'help': 
    #             {'sigma': {'min' : sigma_min,'max' : sigma_max},
    #              'amplitude': {'min' : ampli_min,'max' : ampli_max}}
    #     }, 
    #     {'type': 'LorentzianModel',
    #      'help': 
    #             {'sigma': {'min' : sigma_min,'max' : sigma_max},
    #              'amplitude': {'min' : ampli_min,'max' : ampli_max}}
    #     },
    #     {'type': 'LorentzianModel',
    #      'help': 
    #             {'sigma': {'min' : sigma_min,'max' : sigma_max},
    #              'amplitude': {'min' : ampli_min,'max' : ampli_max}}
    #     },
    #     {'type': 'LorentzianModel',
    #      'help': 
    #             {'sigma': {'min' : sigma_min,'max' : sigma_max},
    #              'amplitude': {'min' : ampli_min,'max' : ampli_max}}
    #     },
        {'type': 'LorentzianModel'},
        {'type': 'LorentzianModel'},
        {'type': 'LorentzianModel'},
        {'type': 'LorentzianModel'},
        {'type': 'LorentzianModel'},
        {'type': 'LorentzianModel'},
        # {'type': 'LorentzianModel'},
        # {'type': 'LorentzianModel'},
    ]
}

peaks_found = update_spec_from_peaks(spec, [0, 1, 2, 3, 4, 5], peak_widths=(5.5,))
fig, ax = plt.subplots()
ax.scatter(spec['x'], spec['y'], s=4)
for i in peaks_found:
    ax.axvline(x=spec['x'][i], c='black', linestyle='dotted')

plot_to_output(fig, 'xrd-fitting-xrd-peaks.png')

model, params = generate_model(spec)
output = model.fit(spec['y'], params, x=spec['x'])
fig = output.plot()
# fig, gridspec = output.plot(data_kws={'markersize':  1})
plot_to_output(fig, 'xrd-fitting-xrd-total.png')

fig, ax = plt.subplots()
ax.scatter(spec['x'], spec['y'], s=4)
components = output.eval_components(x=spec['x'])
print(len(spec['model']))
for i, model in enumerate(spec['model']):
    ax.plot(spec['x'], components[f'm{i}_'])
plot_to_output(fig, 'xrd-fitting-xrd-complex-components.png')

def print_best_values(spec, output):
    model_params = {
        'GaussianModel':   ['amplitude', 'sigma'],
        'LorentzianModel': ['amplitude', 'sigma'],
        'VoigtModel':      ['amplitude', 'sigma', 'gamma']
    }
    best_values = output.best_values
    print('center\tmodel\tamplitude\tsigma\tgamma')
    for i, model in enumerate(spec['model']):
        prefix = f'm{i}_'
        values = ',\t'.join(f'{best_values[prefix+param]:8.3f}' for param in model_params[model["type"]])
        print(f'[{best_values[prefix+"center"]:3.3f}] {model["type"]:16}: {values}')

print_best_values(spec, output)

