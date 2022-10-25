import numpy as np
TOTAL = "../data/total_B.txt"
NORMAL = "../data/normal_B.txt"
def print_table(bx,bxe,by,bye,bz,bze,hbx,hbxe,hbz,hbze):
    for i in range(0,len(bx)):
        print('$',str("{:.2f}".format(bx[i])),'\pm',str("{:.2f}".format(bxe[i])),'$','&','$',str("{:.2f}".format(by[i])),'\pm',str("{:.2f}".format(bye[i])),'$','&','$',str("{:.2f}".format(bz[i])),'\pm',str("{:.2f}".format(bze[i])),'$','&','$',str("{:.2f}".format(hbx[i])),'\pm',str("{:.2f}".format(hbxe[i])),'$','&','$',str("{:.2f}".format(hbz[i])),'\pm',str("{:.2f}".format(hbze[i])), '\\','\hline')

def main(file, digits):
    counter = 0
    f, B_x, B_x_err, B_y ,B_y_err ,B_z ,B_z_err, B_xH, B_zH = np.loadtxt(file, unpack=True, skiprows=1 ) 
    B_xH_err = 5*B_xH/100 + 10*digits    
    B_zH_err = 5*B_zH/100 + 10*digits  
    Z_x = []
    Z_z = []
    Z_x = np.append(Z_x,np.abs(B_x - B_xH)/np.sqrt(B_x_err**2 + B_xH_err**2))
    Z_z = np.append(Z_z,np.abs(B_z - B_zH)/np.sqrt(B_z_err**2 + B_zH_err**2))

    for i in range(0, len(B_x)):
        print('File:',f[i])
        print('B_x:',B_x[i],'+-',B_x_err[i],'| Hall:',B_xH[i],'+-',B_xH_err[i])
        print('B_z:',B_z[i],'+-',B_z_err[i],'| Hall:',B_zH[i],'+-',B_zH_err[i])
        print('B_y:',B_y[i],'+-',B_y_err[i])
        if Z_x[i] < 1.96:
            print('X ok')
            counter+=1
        if Z_z[i] < 1.96:
            print('Z ok')
            counter+=1
    if file == TOTAL:
        print_table(B_x, np.abs(B_x_err), B_y ,np.abs(B_y_err) ,B_z ,np.abs(B_z_err), B_xH, np.abs(B_xH_err), B_zH, np.abs(B_zH_err))
    else:
        print_table(B_x[::-1], np.abs(B_x_err[::-1]), B_y ,np.abs(B_y_err[::-1]) ,B_z[::-1] ,np.abs(B_z_err[::-1]), B_xH[::-1], np.abs(B_xH_err[::-1]), B_zH[::-1], np.abs(B_zH_err[::-1]))
    print(counter)
print('NORMAL')
main(NORMAL, 0.01)
print('TOTAL')
main(TOTAL, 0.01)
