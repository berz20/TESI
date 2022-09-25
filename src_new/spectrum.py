import os
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
DIR = "../data/monocromatore"
PROVA = "../output/prova"
counter = 0 
c = ['orange','red']
title = ['$NV^0$','$NV^-$']
def main(file):
    global counter
    freq, count = np.loadtxt(file, unpack=True) 
    indicies = signal.find_peaks(count, prominence=600, distance=1000)[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(freq,count, c='black')

    for i in range(0,len(indicies)):
        if freq[indicies][i]<650:
            ax.scatter(freq[indicies][i],count[indicies][i], label=title[i]+':\t '+str("{:.1f}".format(freq[indicies][i]))+' nm', c=c[i])
    print(indicies)
    ax.set_xlabel('$\lambda \ [nm]$')
    ax.set_ylabel('$Counts/s$')
    ax.legend()
    plt.savefig(f'{PROVA}/prova_spectr'+str(counter)+'.svg')
    counter += 1
    return


for file in os.listdir(DIR):
    if file.endswith("0V.txt"):
        print(os.path.join(DIR, file))
        f = DIR+"/"+file
        main(f)


