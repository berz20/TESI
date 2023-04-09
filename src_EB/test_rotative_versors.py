#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt

from utilities import *  # My functions: pair_dat_err, uncertainties_to_root_graph_errors
from functions import *
from variables import *


def splitting(file, counter):
    ft = ODMR + file + "_ODMR_data_ch0_range0.dat"
    x, y = np.loadtxt(ft, unpack=True, skiprows=19)
    print("######################################")
    print("File:", file)
    print("")
    y = normalize(y)
    x = x / 1e9

    spec = {"x": x, "y": y, "model": []}
    spec.update
    peaks_found = update_spec_from_peaks(spec, peak_finder(y, counter, a, dist), pw)

    model, params = generate_model(spec)
    output = model.fit(spec["y"], params, x=spec["x"])
    print("Chi", output.chisqr)
    components = output.eval_components(x=spec["x"])
    sum = 0
    for i, model in enumerate(spec["model"]):
        sum = sum + components[f"m{i}_"]

    b_str, p_eu, p_ed, p_iu, p_id, i_p_eu, i_p_ed, i_p_iu, i_p_id = analyze(
        file, counter
    )
    y_eu = spec["y"][peaks_found[-1]]
    y_ed = spec["y"][peaks_found[0]]
    i_y_eu = spec["y"][peaks_found[-2]]
    i_y_ed = spec["y"][peaks_found[1]]
    y_iu = spec["y"][peaks_found[int(len(peaks_found) / 2)]]
    y_id = spec["y"][peaks_found[int(len(peaks_found) / 2) - 1]]
    i_y_iu = spec["y"][peaks_found[int(len(peaks_found) / 2) + 1]]
    i_y_id = spec["y"][peaks_found[int(len(peaks_found) / 2) - 2]]

    ax[0].plot(spec["x"], sum + 0.02 * counter, c="orange")
    ax[0].plot(p_eu, y_eu + 0.02 * counter, marker=".", color="black")
    ax[0].plot(p_ed, y_ed + 0.02 * counter, marker=".", color="black")
    ax[0].plot(p_iu, y_iu + 0.02 * counter, "r.")
    ax[0].plot(p_id, y_id + 0.02 * counter, "r.")
    ax[0].plot(i_p_eu, i_y_eu + 0.02 * counter, marker=".", color="blue")
    ax[0].plot(i_p_ed, i_y_ed + 0.02 * counter, marker=".", color="blue")
    ax[0].plot(i_p_iu, i_y_iu + 0.02 * counter, marker=".", color="green")
    ax[0].plot(i_p_id, i_y_id + 0.02 * counter, marker=".", color="green")
    ax[0].scatter(spec["x"], spec["y"] + 0.02 * counter, s=4, label=b_str)
    ax[0].axes.yaxis.set_visible(False)
    ax[0].set_xlabel("[GHz]")
    # plot_to_output(fig, file+'-total.png')

    print_best_values(spec, output, file, OUTTXTDIRPROVA)
    print("")


def check_B(B_axis, B_exp):
    left_side = B_axis
    right_side = B_exp
    left_side_3 = np.array([u1, u2, u3])
    count = 0
    product = np.dot(left_side, left_side_3)
    for i in range(0, len(product)):
        print("Product sx", product[i])
        print("dx", right_side[i])
        if product[i] == right_side[i]:
            count = count + 1
    if count == 3:
        print("Ok")


def analyze(file, counter):
    peaks, peaks_err, peaks_amp = np.loadtxt(
        OUTTXTDIR + file + ".txt", unpack=True, skiprows=1
    )

    global B_arr
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
    peak_ext_up = np.append(peak_ext_up, peaks[-1])
    peak_ext_down = np.append(peak_ext_down, peaks[0])
    peak_ext_up_err = np.append(peak_ext_up_err, peaks_err[-1])
    peak_ext_down_err = np.append(peak_ext_down_err, peaks_err[0])
    peak_int_up = np.append(peak_int_up, peaks[int(len(peaks) / 2)])
    peak_int_down = np.append(peak_int_down, peaks[int(len(peaks) / 2) - 1])
    peak_int_up_err = np.append(peak_int_up_err, peaks_err[int(len(peaks) / 2)])
    peak_int_down_err = np.append(peak_int_down_err, peaks_err[int(len(peaks) / 2) - 1])
    int_peak_ext_up = np.append(int_peak_ext_up, peaks[-2])
    int_peak_ext_down = np.append(int_peak_ext_down, peaks[1])
    int_peak_ext_up_err = np.append(int_peak_ext_up_err, peaks_err[-2])
    int_peak_ext_down_err = np.append(int_peak_ext_down_err, peaks_err[1])
    int_peak_int_up = np.append(int_peak_int_up, peaks[int(len(peaks) / 2) + 1])
    int_peak_int_down = np.append(int_peak_int_down, peaks[int(len(peaks) / 2) - 2])
    int_peak_int_up_err = np.append(
        int_peak_int_up_err, peaks_err[int(len(peaks) / 2) + 1]
    )
    int_peak_int_down_err = np.append(
        int_peak_int_down_err, peaks_err[int(len(peaks) / 2) - 2]
    )

    if ((len(peaks) % 2) & (len(peaks) < 9)) == 0:
        B_xyz, B_xyz_err, B_str, B_arr = B_calc(file, B_arr, peaks, peaks_err)
        theta = rad_to_deg(np.arctan(B_xyz[0] / B_xyz[2]))
        print("theta", theta)
        return (
            B_str,
            peak_ext_up[counter],
            peak_ext_down[counter],
            peak_int_up[counter],
            peak_int_down[counter],
            int_peak_ext_up[counter],
            int_peak_ext_down[counter],
            int_peak_int_up[counter],
            int_peak_int_down[counter],
        )


def center_split():
    global B_arr
    peak_ext = np.array([])
    peak_int = np.array([])
    peak_ext_err = np.array([])
    peak_int_err = np.array([])
    int_peak_ext = np.array([])
    int_peak_int = np.array([])
    int_peak_ext_err = np.array([])
    int_peak_int_err = np.array([])
    for i in range(0, len(peak_ext_up)):
        peak_ext = np.append(peak_ext, (peak_ext_up[i] + peak_ext_down[i]) / 2)
        peak_int = np.append(peak_int, (peak_int_up[i] + peak_int_down[i]) / 2)
        peak_ext_err = np.append(
            peak_ext_err,
            (np.sqrt(peak_ext_up_err[i] ** 2 + peak_ext_down_err[i] ** 2) / 4),
        )
        peak_int_err = np.append(
            peak_int_err,
            (np.sqrt(peak_int_up_err[i] ** 2 + peak_int_down_err[i] ** 2) / 4),
        )
        int_peak_ext = np.append(
            int_peak_ext, (int_peak_ext_up[i] + int_peak_ext_down[i]) / 2
        )
        int_peak_int = np.append(
            int_peak_int, (int_peak_int_up[i] + int_peak_int_down[i]) / 2
        )
        int_peak_ext_err = np.append(
            int_peak_ext_err,
            (np.sqrt(int_peak_ext_up_err[i] ** 2 + int_peak_ext_down_err[i] ** 2) / 4),
        )
        int_peak_int_err = np.append(
            int_peak_int_err,
            (np.sqrt(int_peak_int_up_err[i] ** 2 + int_peak_int_down_err[i] ** 2) / 4),
        )

    # SpinHamiltonian class takes mag module in [T]
    B = B_arr / 1e3
    # Curve fitting with frequencies extracted form autovalues of hamiltonian
    popt_ext, pcov_ext = curve_fit(Deviation_plotter, B, peak_ext, sigma=peak_ext_err)
    perr_ext = np.sqrt(np.diag(pcov_ext))
    popt_int, pcov_int = curve_fit(Deviation_plotter, B, peak_int, sigma=peak_int_err)
    perr_int = np.sqrt(np.diag(pcov_int))
    int_popt_int, int_pcov_int = curve_fit(
        Deviation_plotter, B, int_peak_int, sigma=int_peak_int_err
    )
    int_perr_int = np.sqrt(np.diag(int_pcov_int))
    int_popt_ext, int_pcov_ext = curve_fit(
        Deviation_plotter, B, int_peak_ext, sigma=int_peak_ext_err
    )
    int_perr_ext = np.sqrt(np.diag(int_pcov_ext))
    fit_ext = Deviation_plotter(B, *popt_ext)
    fit_int = Deviation_plotter(B, *popt_int)
    int_fit_int = Deviation_plotter(B, *int_popt_int)
    int_fit_ext = Deviation_plotter(B, *int_popt_ext)

    popt_ext_plus, pcov_ext_plus = curve_fit(
        Deviation_plotter_plus, B, peak_ext_up, sigma=peak_ext_up_err
    )
    perr_ext_plus = np.sqrt(np.diag(pcov_ext_plus))
    popt_int_plus, pcov_int_plus = curve_fit(
        Deviation_plotter_plus, B, peak_int_up, sigma=peak_int_up_err
    )
    perr_int_plus = np.sqrt(np.diag(pcov_int_plus))
    int_popt_int_plus, int_pcov_int_plus = curve_fit(
        Deviation_plotter_plus, B, int_peak_int_up, sigma=int_peak_int_up_err
    )
    int_perr_int_plus = np.sqrt(np.diag(int_pcov_int_plus))
    int_popt_ext_plus, int_pcov_ext_plus = curve_fit(
        Deviation_plotter_plus, B, int_peak_ext_up, sigma=int_peak_ext_up_err
    )
    int_perr_ext_plus = np.sqrt(np.diag(int_pcov_ext_plus))
    fit_ext_plus = Deviation_plotter_plus(B, *popt_ext_plus)
    fit_int_plus = Deviation_plotter_plus(B, *popt_int_plus)
    int_fit_int_plus = Deviation_plotter_plus(B, *int_popt_int_plus)
    int_fit_ext_plus = Deviation_plotter_plus(B, *int_popt_ext_plus)

    popt_ext_minus, pcov_ext_minus = curve_fit(
        Deviation_plotter_minus, B, peak_ext_down, sigma=peak_ext_down_err
    )
    perr_ext_minus = np.sqrt(np.diag(pcov_ext_minus))
    popt_int_minus, pcov_int_minus = curve_fit(
        Deviation_plotter_minus, B, peak_int_down, sigma=peak_int_down_err
    )
    perr_int_minus = np.sqrt(np.diag(pcov_int_minus))
    int_popt_int_minus, int_pcov_int_minus = curve_fit(
        Deviation_plotter_minus, B, int_peak_int_down, sigma=int_peak_int_down_err
    )
    int_perr_int_minus = np.sqrt(np.diag(int_pcov_int_minus))
    int_popt_ext_minus, int_pcov_ext_minus = curve_fit(
        Deviation_plotter_minus, B, int_peak_ext_down, sigma=int_peak_ext_down_err
    )
    int_perr_ext_minus = np.sqrt(np.diag(int_pcov_ext_minus))
    fit_ext_minus = Deviation_plotter_minus(B, *popt_ext_minus)
    fit_int_minus = Deviation_plotter_minus(B, *popt_int_minus)
    int_fit_int_minus = Deviation_plotter_minus(B, *int_popt_int_minus)
    int_fit_ext_minus = Deviation_plotter_minus(B, *int_popt_ext_minus)

    ax[1].plot(
        B_arr,
        fit_ext_minus,
        color="black",
        label="["
        + str("{:.2f}".format(popt_ext_minus[0]))
        + "$\pm$"
        + str("{:.2f}".format(perr_ext_minus[0]))
        + "]"
        + "$\degree$",
    )
    ax[1].plot(B_arr, fit_int_minus, color="red")
    ax[1].plot(
        B_arr,
        int_fit_ext_minus,
        color="blue",
        label="["
        + str("{:.2f}".format(int_popt_ext_minus[0]))
        + "$\pm$"
        + str("{:.2f}".format(int_perr_ext_minus[0]))
        + "]"
        + "$\degree$",
    )
    ax[1].plot(B_arr, int_fit_int_minus, color="green")
    ax[1].plot(B_arr, fit_ext_plus, color="black")
    ax[1].plot(B_arr, int_fit_ext_plus, color="blue")
    ax[1].plot(
        B_arr,
        int_fit_int_plus,
        color="green",
        label="["
        + str("{:.2f}".format(int_popt_int_plus[0]))
        + "$\pm$"
        + str("{:.2f}".format(int_perr_int_plus[0]))
        + "]"
        + "$\degree$",
    )
    ax[1].plot(
        B_arr,
        fit_int_plus,
        color="red",
        label="["
        + str("{:.2f}".format(popt_int_plus[0]))
        + "$\pm$"
        + str("{:.2f}".format(perr_int_plus[0]))
        + "]"
        + "$\degree$",
    )
    # fit_0_plus = Deviation_plotter_plus(B,0)
    # fit_0_minus = Deviation_plotter_minus(B,0)
    # ax[1].plot(B_arr,fit_0_plus,color='orange', label='0'+'$\degree$')
    # ax[1].plot(B_arr,fit_0_minus,color='orange')
    # plot peaks position (dot)
    ax[1].errorbar(
        B_arr,
        peak_ext_up,
        yerr=peak_ext_up_err,
        capsize=5,
        fmt=".",
        color="black",
        label="Resonance's peaks positions",
    )
    ax[1].errorbar(
        B_arr, peak_ext_down, yerr=peak_ext_down_err, capsize=5, fmt=".", color="black"
    )
    ax[1].errorbar(
        B_arr, peak_int_up, yerr=peak_int_up_err, capsize=5, fmt=".", color="red"
    )
    ax[1].errorbar(
        B_arr, peak_int_down, yerr=peak_int_down_err, capsize=5, fmt=".", color="red"
    )
    ax[1].errorbar(
        B_arr,
        int_peak_ext_up,
        yerr=int_peak_ext_up_err,
        capsize=5,
        fmt=".",
        color="blue",
    )
    ax[1].errorbar(
        B_arr,
        int_peak_ext_down,
        yerr=int_peak_ext_down_err,
        capsize=5,
        fmt=".",
        color="blue",
    )
    ax[1].errorbar(
        B_arr,
        int_peak_int_up,
        yerr=int_peak_int_up_err,
        capsize=5,
        fmt=".",
        color="green",
    )
    ax[1].errorbar(
        B_arr,
        int_peak_int_down,
        yerr=int_peak_int_down_err,
        capsize=5,
        fmt=".",
        color="green",
    )
    # Plot center of peaks (x)
    # ax[1].errorbar(B_arr, peak_ext, yerr=peak_ext_err,fmt='x', color='black', label="Peak's centers")
    # ax[1].errorbar(B_arr, peak_int, yerr=peak_int_err,fmt='x', color='red')
    # ax[1].errorbar(B_arr, int_peak_ext, yerr=int_peak_int_err,fmt='x', color='blue')
    # ax[1].errorbar(B_arr, int_peak_int, yerr=int_peak_ext_err,fmt='x', color='green')
    # Plot fit center with frequencies hamiltonian
    # ax[1].plot(B_arr,fit_ext,color='black', label='['+str("{:.2f}".format(popt_ext[0]))+'$\pm$'+str("{:.2f}".format(perr_ext[0]))+']'+'$\degree$')
    # ax[1].plot(B_arr,fit_int,color='red', label='['+str("{:.2f}".format(popt_int[0]))+'$\pm$'+str("{:.2f}".format(perr_int[0]))+']'+'$\degree$')
    # ax[1].plot(B_arr,int_fit_ext,color='blue', label='['+str("{:.2f}".format(int_popt_ext[0]))+'$\pm$'+str("{:.2f}".format(int_perr_ext[0]))+']'+'$\degree$')
    # ax[1].plot(B_arr,int_fit_int,color='green', label='['+str("{:.2f}".format(int_popt_int[0]))+'$\pm$'+str("{:.2f}".format(int_perr_int[0]))+']'+'$\degree$')
    # 2.87 GHz line
    ax[1].axhline(2.87, c="black", linestyle="dotted", label="$2.87 \ GHz$")
    # Labels
    ax[1].set_xlabel("$External \ magnetic \ field \ [mT]$")
    ax[1].set_ylabel("$Transition \ frequencies \ [GHz]$")
    ax[1].legend()
    # plot_to_output(fig, 'deviation.pdf')
    print("")
    print("Amplitude Factor:", a, "\nPeak Width:", pw)
    print("")


files = [
    # (3-11)
    "20220802-1407-11",
    "20220802-1332-19",
    "20220802-1306-04",
    "20220802-1252-09",
    # '20220802-1238-26',
    "20220802-1153-00",
    "20220802-1136-53",
    # '20220802-1101-15',
    # #(3-15)
    # '20220802-1407-11',
    # # '20220802-1332-19',
    # '20220802-1306-04',
    # '20220802-1252-09',
    # # '20220802-1238-26',
    # # '20220802-1153-00',
    # '20220802-1136-53',
    # '20220802-1101-15',
    # #(All)
    # '20220802-1407-11',
    # '20220802-1332-19',
    # '20220802-1306-04',
    # '20220802-1252-09',
    # '20220802-1238-26',
    # '20220802-1153-00',
    # '20220802-1136-53',
    # '20220802-1101-15',
]
files = files[::-1]
fig, ax = plt.subplots(figsize=(16, 8), ncols=2)

## PARAMETER DATA
#  Peaks amplitude-contrast
# a = [30,38,27,40,40,50,55,95]
# (3-11)
a = [30, 38, 27, 40, 50, 55]
# (3-15)
# a = [30,27,40,55,95]
a = a[::-1]
#  Peaks width
pw = (1.5,)
#  Peaks distance
dist = 4.1
#  Offset
offset = 1

# Create first plot with all ODMR resonances fitted with custom model of composite inverse lorentzians (spec)
for f, i in zip(files, range(0, len(files))):
    splitting(f, i)

handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles[::-1], labels[::-1], title="B Field", loc="upper left")

center_split()
plt.savefig(f"{OUTIMGDIR}/total_double_deg(3-11).pdf")
plt.savefig(f"{OUTIMGDIR}/total_double_deg(3-11).svg")
plt.show()

# File used
# '20220802-1031-50',
# '20220802-1101-15',
# '20220802-1121-27',
# '20220802-1136-53',
# '20220802-1153-00',
# '20220802-1208-48',
# '20220802-1031-50',
# '20220802-1225-51',
# '20220802-1238-26',
# '20220802-1252-09',
# '20220802-1306-04',
# '20220802-1332-19',
# '20220802-1407-11',
