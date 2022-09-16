import os
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
DIR = "../data/monocromatore"
PROVA = "../output/prova"
counter = 0 
def main(file):
    global counter
    freq, count = np.loadtxt(file, unpack=True) 
    indicies = signal.find_peaks(count, threshold=20)[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(freq,count)
    ax.scatter(freq[indicies][0],count[indicies][0], label= freq[indicies][0])
    ax.set_xlabel('$\lambda \ [nm]$')
    ax.set_ylabel('$Counts/s$')
    ax.legend()
    plt.savefig(f'{PROVA}/prova_spectr'+str(counter)+'.svg')
    counter += 1
    return


for file in os.listdir(DIR):
    if file.endswith("000DAC_NV"):
        print(os.path.join(DIR, file))
        f = DIR+"/"+file
        main(f)


