import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
from hamiltonians import SpinHamiltonian
from scipy.optimize import curve_fit
OUTIMGDIR = "../output/prova/"

ham = SpinHamiltonian()

def Deviation_plotter():
    # B = np.array([0.00687,0.00813,0.00952,0.01252,0.01749,0.01952,0.02315,0.03063])
    B = np.linspace(0,0.140,100)
    sham = np.vectorize(ham.transitionFreqs, otypes=[np.ndarray])
    freqs = np.array(sham(B,0,0))
    freqs = np.array(freqs.tolist())
    freqs_dev_90 = np.array(sham(B,90,0))
    freqs_dev_90 = np.array(freqs_dev_90.tolist())
    freqs_dev_60 = np.array(sham(B,60,0))
    freqs_dev_60 = np.array(freqs_dev_60.tolist())
    freqs_dev_30 = np.array(sham(B,30,0))
    freqs_dev_30 = np.array(freqs_dev_30.tolist())

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot()
    
    ax.set_xlabel('$B \ [T]$')
    ax.set_ylabel('$Transition \ Frequency \ [GHz]$')
    ax.plot(B, freqs[:,0],color='black', label='$B_{//}$')
    ax.plot(B, freqs[:,1],color='black')
    ax.axhline(2.87, c='black', linestyle='dotted')
    ax.plot(B, freqs_dev_30[:,0],color='tab:blue', label='$30\degree$')
    ax.plot(B, freqs_dev_30[:,1],color='tab:blue')
    # ax.plot(B, (freqs_dev_30[:,0]+freqs_dev_30[:,1])/2,color='tab:blue', label='$30\degree$')
    ax.plot(B, freqs_dev_60[:,0],color='red', label='$60\degree$')
    ax.plot(B, freqs_dev_60[:,1],color='red')
    # ax.plot(B, (freqs_dev_60[:,0]+freqs_dev_60[:,1])/2,color='red', label='$60\degree$')
    ax.plot(B, freqs_dev_90[:,0],color='green', label='$90\degree$')
    ax.plot(B, freqs_dev_90[:,1],color='green')
    # ax.plot(B, (freqs_dev_90[:,0]+freqs_dev_90[:,1])/2,color='green', label='$90\degree$')
    ax.legend()
    plt.grid()
    # plt.savefig(f'{OUTIMGDIR}/prova_deg_0_over.svg')
    plt.show()

if __name__ == '__main__':
    Deviation_plotter()
