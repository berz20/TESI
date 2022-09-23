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
    peaks_found = update_spec_from_peaks(spec,peak_finder(y,counter,a,dist), pw)
    
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
    ax[0].plot(spec['x'], sum + 0.05*counter, c='orange')
    ax[0].plot(p_eu,y_eu + 0.05*counter, marker='.',color='black') 
    ax[0].plot(p_ed,y_ed + 0.05*counter, marker='.',color='black') 
    ax[0].scatter(spec['x'], spec['y'] + 0.05*counter, s=4, label=b_str)
    ax[0].axes.yaxis.set_visible(False)
    ax[0].set_xlabel('[GHz]')
    # plot_to_output(fig, file+'-total.png')
    
    print_best_values(spec, output, file, OUTTXTDIRPROVA)
    print("")

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

    # SpinHamiltonian class takes mag module in [T]
    B = B_arr/1e3
    # Curve fitting with frequencies extracted form autovalues of hamiltonian
    # popt_ext, pcov_ext = curve_fit(Deviation_plotter, B, peak_ext, sigma= peak_ext_err)
    # perr_ext = np.sqrt(np.diag(pcov_ext))
    # fit_ext = Deviation_plotter(B, *popt_ext)
    popt_plus, pcov_plus = curve_fit(Deviation_plotter_plus, B, peak_ext_up, sigma= peak_ext_up_err)
    perr_plus = np.sqrt(np.diag(pcov_plus))
    fit_0_plus = Deviation_plotter_plus(B, *popt_plus)
    popt_minus, pcov_minus = curve_fit(Deviation_plotter_minus, B, peak_ext_down, sigma= peak_ext_down_err)
    perr_minus = np.sqrt(np.diag(pcov_minus))
    fit_0_minus = Deviation_plotter_minus(B, *popt_minus)

    ax[1].errorbar(B_arr, peak_ext_up, yerr=peak_ext_up_err,capsize=5, fmt='.', color='black', label="Resonance's peaks positions")
    ax[1].errorbar(B_arr, peak_ext_down, yerr=peak_ext_down_err,capsize=5, fmt='.', color='black')
    # ax[1].errorbar(B_arr, peak_ext, yerr=peak_ext_err,fmt='x', color='black', label="Peak's centers") 
    # ax[1].plot(B_arr,fit_ext,color='black', label='['+str("{:.2f}".format(popt_ext[0]))+'$\pm$'+str("{:.2f}".format(perr_ext[0]))+']'+'$\degree$')
    # ax[1].plot(B_arr,fit_0,color='orange', label='0')
    ax[1].plot(B_arr,fit_0_plus,color='orange')
    ax[1].plot(B_arr,fit_0_minus,color='orange', label='['+str("{:.2f}".format(popt_minus[0]))+'$\pm$'+str("{:.2f}".format(perr_minus[0]))+']'+'$\degree$')
    ax[1].axhline(2.87, c='black', linestyle='dotted', label='$2.87 \ GHz$')
    ax[1].legend()
    ax[1].set_xlabel('External magnetic field [mT]')
    ax[1].set_ylabel('transition frequencies [GHz]')
    # plot_to_output(fig, 'deviation.pdf')
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
fig, ax = plt.subplots(figsize=(16, 8), ncols=2)
a = [23,23,23,23,19,16,19,23,20,18,53] 
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

# ax[0].axvline(2.87, c='black', linestyle='dotted', label='$2.87 \ GHz$')
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles[::-1], labels[::-1], title='$B_i \ Field$', loc='upper left')
#ax2[0].legend(loc='upper left')

center_split()
plt.savefig(f'{OUTIMGDIR}/total_double_normal.svg')
plt.savefig(f'{OUTIMGDIR}/total_double_normal.pdf')
plt.show()
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
