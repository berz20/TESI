import numpy as np
TOTAL = "../data/compareB_total.csv"
NORMAL = "../data/compareB_normal.csv"

def total():
    f, B_x, B_x_err, B_y ,B_y_err ,B_z ,B_z_err, B_xH, B_zH = np.loadtxt(TOTAL, unpack=True, skiprows=1 ) 
    B_xH_err = 5*B_xH/100 
    B_zH_err = 5*B_zH/100 
    Z_x = []
    Z_z = []
    Z_x = np.append(Z_x,np.abs(B_x - B_xH)/np.abs(B_x_err**2 + B_xH_err**2))
    Z_z = np.append(Z_z,np.abs(B_z - B_zH)/np.abs(B_z_err**2 + B_zH_err**2))

    for i in range(0, len(B_x)):
        print('File:',f[i])
        print('B_x:',B_x[i],'+-',B_x_err[i],'| Hall:',B_xH[i],'+-',B_xH_err[i])
        print('B_z:',B_z[i],'+-',B_z_err[i],'| Hall:',B_zH[i],'+-',B_zH_err[i])
        print('B_y:',B_y[i],'+-',B_y_err[i])
        if Z_x[i] < 1.96:
            print('X ok')
        if Z_z[i] < 1.96:
            print('Z ok')
def normal():
    f, B_0, B_0_err, B_xH, B_zH = np.loadtxt(NORMAL, unpack=True, skiprows=1 ) 
    B_xH_err = 5*B_xH/100 
    B_zH_err = 5*B_zH/100 
    B_H = np.sqrt(B_xH**2+B_zH**2)
    B_H_err = np.sqrt((B_xH/np.sqrt(B_xH**2+B_zH**2)*B_xH_err)**2 + (B_zH/np.sqrt(B_xH**2+B_zH**2)*B_zH_err)**2)
    Z = []
    Z = np.append(Z,np.abs(B_0 - B_H)/np.abs(B_0_err**2 + B_H_err**2))

    for i in range(0, len(B_0)):
        print('File:',f[i])
        print('B_x:',B_0[i],'+-',B_0_err[i],'| Hall:',B_H[i],'+-',B_H_err[i])
        print('B_z Hall:',B_xH[i],'+-',B_xH_err[i])
        print('B_z Hall:',B_zH[i],'+-',B_zH_err[i])
        if Z[i] < 1.96:
            print('X ok')
total() 
normal()
