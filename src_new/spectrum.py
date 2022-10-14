import os
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
DIR = "../data/monocromatore"
PROVA = "../output/prova"
counter = 0 
c = ['red','orange']
title = ['ZPL','ZPL']
def main(file):
    global counter
    lenght = np.array([])
    count = np.array([])
    lenght_d, count_d = np.loadtxt(file, unpack=True) 
    for i in range(0,len(lenght_d)):
        if lenght_d[i]>580:
            lenght = np.append(lenght,lenght_d[i])
            count = np.append(count,count_d[i])
    indicies = signal.find_peaks(count, prominence=600, distance=1000)[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(lenght,count, c='black')

    for i in range(0,len(indicies)):
        if lenght[indicies][i]<650:
            ax.scatter(lenght[indicies][i],count[indicies][i], label=title[i]+': '+str("{:.1f}".format(lenght[indicies][i]))+' nm', c=c[i])
    print(indicies)
    ax.legend( prop={'size': 15})
    plt.xlabel('$\lambda \ [nm]$', fontsize=18)
    plt.ylabel('$Fluorescence  \ [Counts/s]$', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f'{PROVA}/prova_spectr'+str(counter)+'.svg')
    counter += 1
    return


for file in os.listdir(DIR):
    if file.endswith("0V.txt"):
        print(os.path.join(DIR, file))
        f = DIR+"/"+file
        main(f)


