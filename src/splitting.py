# from urllib.request import url2pathname
import os
import numpy as np
import scipy as sp
import sympy as sy
import pandas as pd
import scipy.signal
from matplotlib import pyplot as plt
from collections import namedtuple
import scipy.constants as cons
DATADIR = "../data"
ODMR = "../data/ODMR"
OUTPUTDIR = "../output"
index_output = ["x", "y", "z"]  # z direction confocal
u1 = np.sqrt(2/3)*np.array([0, 1, 1/np.sqrt(2)])
u2 = np.sqrt(2/3)*np.array([0, -1, 1/np.sqrt(2)])
u3 = np.sqrt(2/3)*np.array([1, 0, -1/np.sqrt(2)])
u4 = np.sqrt(2/3)*np.array([-1, 0, -1/np.sqrt(2)])
import package/module
# import glob, os
# os.chdir("/home/berz/Documents/TESI/data/ODMR")
# for file in glob.glob("*.txt"):
#     print(file)

# Constants = namedtuple('Constants', ['pi', 'e'])
# constants = Constants(3.14, 2.718)
# constants.pi
# constants.pi = 3


def print_array(title, word, bool, array, unit, index):
    # print("")
    print(title)
    if bool:
        for i in range(0, len(array)):
            print(word, index[i], ":", array[i], unit)
    else:
        for i in range(0, len(array)):
            print(word, i, ":", array[i], unit)


def diff_peaks(peaks, freq):
    freq_peaks = freq[peaks]/1e+9
    if ((len(peaks) % 2) == 0):
        diff = np.array([])
        cen = np.array([])
        print("Peaks positions:")
        for i in range(0, int(len(peaks)/2)):
            diff = np.append(diff, freq_peaks[len(peaks)-1-i] - freq_peaks[i])
            cen = np.append(
                cen, (freq_peaks[len(peaks)-1-i] + freq_peaks[i])/2)
            print("Resonance", i, ":",
                  freq_peaks[i], "Ghz -",  freq_peaks[len(peaks)-1-i], "GHz")
        return diff, cen


def splitting(file):
    # freq, count_data = np.loadtxt(f'{DATADIR}/lor_try.dat', unpack=True, skiprows=19 )
    print("")
    freq, count_data = np.loadtxt(file, unpack=True, skiprows=19)
    count = count_data / np.max(count_data)
    contrast = (max(count) - min(count))/(max(count) + min(count))
    height = max(count)*(1-contrast)/(1+contrast)
    a = 1/2.7
    height_new = max(count)*(1-contrast*a)/(1+contrast*a)
    print("height", height)
    print("height new", height_new)
    print("contrast", contrast)
    # find all the peaks that associated with the negative peaks
    # peaks_negative, properties = scipy.signal.find_peaks(count * -1., prominence=(None, 0.6))
    peaks_negative, properties = scipy.signal.find_peaks(count * -1., height=-height_new)

    sigma_peak = scipy.signal.peak_widths(count, peaks_negative, rel_height=0.5)[0]
    ampli_peak = count[peaks_negative]-max(count)
    print("sigma peak", sigma_peak)
    print("ampli peak", ampli_peak)
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(freq, count, 'b-', linewidth=2)

    ax.plot(freq[peaks_negative], count[peaks_negative],
            'ro', label='negative peaks')
    plt.show()
    # peak_widths=(0.006e+9, 0.01e+9)
    # peaks_negative = scipy.signal.find_peaks_cwt(count, peak_widths)
    print("")

    freq_peaks = freq[peaks_negative]/1e+9
    if (((len(peaks_negative) % 2) & (len(peaks_negative) < 9)) == 0):
        wid = np.array([])
        cen = np.array([])
        print("Peaks positions:")
        for i in range(0, int(len(peaks_negative)/2)):
            wid = np.append(
                wid, freq_peaks[len(peaks_negative)-1-i] - freq_peaks[i])
            cen = np.append(
                cen, (freq_peaks[len(peaks_negative)-1-i] + freq_peaks[i])/2)
            print("Resonance", i, ":",
                  freq_peaks[i], "Ghz -",  freq_peaks[len(peaks_negative)-1-i], "GHz")

        print_array("Resonance width:", "Resonance", False, wid, "GHz", [])

        print_array("Resonance center:", "Resonance", False, cen, "GHz", [])

        # Magnetic Field

        mu_b = cons.physical_constants["Bohr magneton"][0]
        h = cons.physical_constants["Planck constant"][0]
        mu_b_t = cons.physical_constants["Bohr magneton"]
        h_t = cons.physical_constants["Planck constant"]
        g = 2.002

        print(2.*g*mu_b/h)

        gamma = 28  # [GHz/T]

        # B = h*1e+3*wid/(2.*g*mu_b)

        B = wid/gamma  # [T]

        # print(mu_b_t)

        # print(h_t)

        # print(g)

        print_array("Resonance ODMR:", "Resonance", False, B*1e+3, "mT", [])

        if len(wid) > 2:

            left_side_3 = np.array([u1, -1*u2, -1*u3])

            right_side_3 = B

            B_zee = np.linalg.inv(left_side_3).dot(right_side_3)

            B_zee_module = 0

            for i in range(0, len(B_zee)):

                B_zee_module = B_zee_module + B_zee[i]**2

            B_zee_module = np.sqrt(B_zee_module)

            if len(wid) > 3:

                left_side_4 = u4

                right_side_4 = B[3]

                if (B_zee @ left_side_4) != right_side_4:

                    B_zee = -1*B_zee

            print_array("Magnetic Field Components:", "B",
                        True, B_zee*1000, "mT", index_output)

            print("")

            print("Magnetic Field Module:")

            print("|B| :", B_zee_module*1000, "mT")


for file in os.listdir(ODMR):
    if file.endswith("ch0_range0.dat"):
        print(os.path.join(ODMR, file))
        s = '"'
        f = ODMR+"/"+file
        splitting(f)
