#!/usr/bin/env python
import numpy as np

DATADIR = "../data"
ODMR = "../data/ODMR/"
OUTPUTDIR = "../output/"
OUTIMGDIR = "../output/prova/"
OUTTXTDIR = "../output/txt/"
OUTTXTDIRPROVA = "../output/prova/"
index_output = ["x", "y", "z"]  # z direction confocal
u1 = -1*np.sqrt(2/3)*np.array([0, 1, 1/np.sqrt(2)])
u2 = -1*np.sqrt(2/3)*np.array([0, -1, 1/np.sqrt(2)])
u3 = -1*np.sqrt(2/3)*np.array([1, 0, -1/np.sqrt(2)])
u4 = -1*np.sqrt(2/3)*np.array([-1, 0, -1/np.sqrt(2)])
# u1 = np.sqrt(1/3)*np.array([1, 1, 1])
# u2 = np.sqrt(1/3)*np.array([1, -1, -1])
# u3 = np.sqrt(1/3)*np.array([-1, 1, -1])
# u4 = np.sqrt(1/3)*np.array([-1, -1, 1])

B_arr = np.array([])  
B_xyz_X = np.array([])  
B_xyz_Y = np.array([])  
B_xyz_Z = np.array([])  
B_xyz_arr_err = np.array([])  
peak_ext_up = np.array([]) 
peak_ext_down = np.array([])  
peak_int_up = np.array([])  
peak_int_down = np.array([])  
peak_ext_up_err = np.array([])  
peak_ext_down_err = np.array([])  
peak_int_up_err = np.array([])  
peak_int_down_err = np.array([])  
int_peak_ext_up = np.array([]) 
int_peak_ext_down = np.array([])  
int_peak_int_up = np.array([])  
int_peak_int_down = np.array([])  
int_peak_ext_up_err = np.array([])  
int_peak_ext_down_err = np.array([])  
int_peak_int_up_err = np.array([])  
int_peak_int_down_err = np.array([])  

