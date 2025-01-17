#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from utilities import * # My functions: pair_dat_err, uncertainties_to_root_graph_errors
from functions import * 
from variables import * 

B_arr = np.array([])  
B_arr_err = np.array([])  
peak_ext_up = np.array([]) 
peak_ext_down = np.array([])  
peak_ext_up_err = np.array([])  
peak_ext_down_err = np.array([])  

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
    peaks_found = update_spec_from_peaks(spec,peak_finder(y,counter, a, dist), pw)
    
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
    
    print_best_values(spec, output, file, OUTTXTDIRPROVA)
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

def analyze(file,counter):
    peaks, peaks_err, peaks_amp = np.loadtxt(OUTTXTDIR+file+'.txt', unpack=True, skiprows=1)

    global B_arr   
    global B_arr_err   
    global B_xyz_X   
    global B_xyz_Y   
    global B_xyz_Z   
    global B_xyz_arr_err   
    global peak_ext_up  
    global peak_ext_down   
    global peak_ext_up_err   
    global peak_ext_down_err   
    peak_ext_up = np.append(peak_ext_up,peaks[-1])
    peak_ext_down = np.append(peak_ext_down,peaks[0])
    peak_ext_up_err = np.append(peak_ext_up_err,peaks_err[-1])
    peak_ext_down_err = np.append(peak_ext_down_err,peaks_err[0])
    
    B_xyz, B_xyz_err, B_str, B_arr = B_calc(file,B_arr,peaks,peaks_err)
    B_xyz_X = np.append(B_xyz_X, B_xyz[0])
    B_xyz_Y = np.append(B_xyz_Y, B_xyz[1])
    B_xyz_Z = np.append(B_xyz_Z, B_xyz[2])
    B_xyz_arr_err = np.append(B_xyz_arr_err, B_xyz_err)
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
    # plot_to_output(fig, 'deviation_normal.pdf')
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
        # '20220728-1241-44',
        '20220728-1338-08',
        '20220728-1222-32',
        '20220801-1113-48',
        '20220728-0912-42',
        # '20220728-0947-12',
        '20220728-1128-20',
        '20220801-1030-48',
        ] 
        # '20220801-1053-22',
fig, ax = plt.subplots(figsize=(8, 8))
a = [23,23,23,23,19,16,19,23,20,18,53] 
# a = [23,23,23,23,19,19,16,19,23,20,20,18,53] 
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
plt.savefig(f'{OUTIMGDIR}/splitting_normal.svg')
plt.show()

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

