#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from utilities import * # My functions: pair_dat_err, uncertainties_to_root_graph_errors
from functions import * 
from variables import * 

B_xyz_X = np.array([])  
B_xyz_Y = np.array([])  
B_xyz_Z = np.array([])  

def splitting(file, counter):

    ft = ODMR+file+'_ODMR_data_ch0_range0.dat'
    x, y = np.loadtxt(ft, unpack=True, skiprows=19 )
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
    
    b_str,p_eu,p_ed,p_iu,p_id,i_p_eu,i_p_ed,i_p_iu,i_p_id = analyze(file,counter) 

    ax.scatter(spec['x'], spec['y'] + 0.02*counter, s=4, label=b_str)
    ax.axes.yaxis.set_visible(False)
    ax.set_xlabel('[GHz]')
    # plot_to_output(fig, file+'-total.png')
    
    print_best_values(spec, output, file, OUTTXTDIRPROVA)
    print("")


def analyze(file,counter):
    peaks, peaks_err, peaks_amp = np.loadtxt(OUTTXTDIR+file+'.txt', unpack=True, skiprows=1)

    global B_arr   
    global B_xyz_X   
    global B_xyz_Y   
    global B_xyz_Z   
    global B_xyz_arr_err   
    global peak_ext_up  
    global peak_ext_down   
    global peak_int_up   
    global peak_int_down   
    global peak_ext_up_err   
    global peak_ext_down_err   
    global peak_int_up_err   
    global peak_int_down_err   
    global int_peak_ext_up  
    global int_peak_ext_down   
    global int_peak_int_up   
    global int_peak_int_down   
    global int_peak_ext_up_err   
    global int_peak_ext_down_err   
    global int_peak_int_up_err   
    global int_peak_int_down_err   
    peak_ext_up = np.append(peak_ext_up,peaks[-1])
    peak_ext_down = np.append(peak_ext_down,peaks[0])
    peak_ext_up_err = np.append(peak_ext_up_err,peaks_err[-1])
    peak_ext_down_err = np.append(peak_ext_down_err,peaks_err[0])
    peak_int_up = np.append(peak_int_up,peaks[int(len(peaks)/2)])
    peak_int_down = np.append(peak_int_down,peaks[int(len(peaks)/2)-1])
    peak_int_up_err = np.append(peak_int_up_err,peaks_err[int(len(peaks)/2)])
    peak_int_down_err = np.append(peak_int_down_err,peaks_err[int(len(peaks)/2)-1])
    int_peak_ext_up = np.append(int_peak_ext_up,peaks[-2])
    int_peak_ext_down = np.append(int_peak_ext_down,peaks[1])
    int_peak_ext_up_err = np.append(int_peak_ext_up_err,peaks_err[-2])
    int_peak_ext_down_err = np.append(int_peak_ext_down_err,peaks_err[1])
    int_peak_int_up = np.append(int_peak_int_up,peaks[int(len(peaks)/2)+1])
    int_peak_int_down = np.append(int_peak_int_down,peaks[int(len(peaks)/2)-2])
    int_peak_int_up_err = np.append(int_peak_int_up_err,peaks_err[int(len(peaks)/2)+1])
    int_peak_int_down_err = np.append(int_peak_int_down_err,peaks_err[int(len(peaks)/2)-2])
    
    B_xyz, B_xyz_err, B_str, B_arr = B_calc(file,B_arr,peaks,peaks_err)
    B_xyz_X = np.append(B_xyz_X, B_xyz[0])
    B_xyz_Y = np.append(B_xyz_Y, B_xyz[1])
    B_xyz_Z = np.append(B_xyz_Z, B_xyz[2])
    B_xyz_arr_err = np.append(B_xyz_arr_err, B_xyz_err)
    return B_str, peak_ext_up[counter], peak_ext_down[counter], peak_int_up[counter], peak_int_down[counter], int_peak_ext_up[counter], int_peak_ext_down[counter], int_peak_int_up[counter], int_peak_int_down[counter] 

def mult_vect():
    zeros= np.array([])
    for i in range(0,len(B_xyz_X)):
        zeros = np.append(zeros,0.)
    x, y, z = np.meshgrid(
                            zeros,
                            zeros,
                            zeros
                            )

    # Make the direction data for the arrows
    # u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    # v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    # w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
    #  np.sin(np.pi * z))
    # soa = np.array([[0, 0, 1, 1, -2, 0], [0, 0, 2, 1, 1, 0],
    #             [0, 0, 3, 2, 1, 0], [0, 0, 4, 0.5, 0.7, 0]])
    #
    # X, Y, Z, U, V, W = zip(*soa)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    c = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey']
    for i in range(0, len(B_xyz_X)):
        ax.quiver(x[i], y[i], z[i], B_xyz_Y[i]*1e+3,B_xyz_Z[i]*1e+3,B_xyz_X[i]*1e+3,color=c[i],arrow_length_ratio=0.1, linestyle='-', linewidth=0.9)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([  0, 10])
    # ax.quiver(X, Y, Z, U, V, W)
    # ax[1].quiver(B_xyz_X, B_xyz_Y, B_xyz_Z)
    ax.legend()  
    ax.set_xlabel('$B_y \ [mT]$')
    ax.set_ylabel('$B_z \ [mT]$')
    ax.set_zlabel('$B_x \ [mT]$')
    # plot_to_output(fig, 'deviation.pdf')
    print("")
    print('Amplitude Factor:',a,'\nPeak Width:',pw)
    print('')


        # '20220802-1225-51',
        # '20220802-1121-27',
        # '20220802-1208-48',
# files = [
#         '20220802-1101-15',
#         '20220802-1031-50',
#         '20220802-1136-53',
#         '20220802-1153-00',
#         '20220802-1238-26',
#         '20220802-1252-09',
#         '20220802-1306-04',
#         '20220802-1332-19',
#         '20220802-1407-11',
#         ]
files = [
        '20220802-1101-15',
        '20220802-1136-53',
        '20220802-1153-00',
        '20220802-1238-26',
        '20220802-1252-09',
        '20220802-1306-04',
        '20220802-1332-19',
        '20220802-1407-11',
        ]
fig = plt.figure(figsize=(16, 8))
a = [91,63,45,50,40,40,27,38,30]
# pw = [(1.5,),(1.5,),(1.5,),(1.5,),(1.5,),(1.5,),(1.5,),(1.5,),(1.5,),(1.5,),(1.5,),]
#Amplitude
# a = 1/0.0209
#Peaks width
pw = (1.5,)
#Peaks distance
dist =  4.1
#Offset
offset = 1 
ax = fig.add_subplot(1, 2, 1)
for f,i in zip(files, range(0,len(files))):
    splitting(f,i)

# ax[0].axvline(2.87, c='black', linestyle='dotted', label='$2.87 \ GHz$')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title='B Field', loc='upper left')
#ax2[0].legend(loc='upper left')

mult_vect()
plt.savefig(f'{OUTIMGDIR}/multiple_vector.svg')
# plt.show()
# File used 
# 20220802-1031-50
# 20220802-1101-15
# 20220802-1121-27
# 20220802-1136-53
# 20220802-1153-00
# 20220802-1208-48
# 20220802-1031-50
# 20220802-1225-51
# 20220802-1238-26
# 20220802-1252-09
# 20220802-1306-04
# 20220802-1332-19
# 20220802-1407-11
