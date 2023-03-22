from copy import deepcopy
from subprocess import check_output
from matplotlib import rcParamsDefault
from plot_energy import create_band_image
from qeoutput import check_eigenvalues
import matplotlib.pyplot as plt
import numpy as np
import os
import re

plt.rcParams["figure.dpi"]=150
plt.rcParams["figure.facecolor"]="white"
plt.rcParams["figure.figsize"]=(8, 6)

TEMPLATE = "si.bands.template"
FILENAME = "si.bandspy.in"
FILEOUTPUT = "si.bandspy.out"

PP_FILENAME = "si_bands_pp.in"
PP_FILEOUTPUT = "si_bands_pp.out"

FORMATTING_DECIMALS = 4

EPSILON_CONVERGENCE = 0.1


def format_row(row, row_num) -> str:
    FD = FORMATTING_DECIMALS
    return f"   {row[0]:.{FD}f} {row[1]:.{FD}f} {row[2]:.{FD}f} {row_num: >3}\n"


def create_file(matrix):
    string_builder = ""

    with open(TEMPLATE, "r") as f:
        string_builder = f.read() + "\n"

    string_builder += f"   {len(matrix)}\n"

    for c, row in enumerate(matrix):
        string_builder += format_row(row, c)

    with open(FILENAME, 'w') as f:
        f.write(string_builder)


def create_image(output_name):
    # load data
    data = np.loadtxt('si_bands.dat.gnu')

    k = np.unique(data[:, 0])
    bands = np.reshape(data[:, 1], (-1, len(k)))

    for band in range(len(bands)):
        plt.plot(k, bands[band, :], linewidth=1, alpha=0.5, color='k')
    plt.xlim(min(k), max(k))

    plt.savefig("images/" + output_name)
    plt.clf()
    plt.cla()


def copy_dat_file(output_name):
    os.system(f"cp si_bands.dat datfiles/{output_name}")


def check_success(espresso_output) -> bool:
    return "JOB DONE" in espresso_output


def calculate_energies() -> bool:  # Returns True if successful
    outp1 = os.popen(f"pw.x -i {FILENAME} > {FILEOUTPUT}; cat {FILEOUTPUT}")
    outp2 = os.popen(f"bands.x < {PP_FILENAME} > {PP_FILEOUTPUT}; cat {PP_FILEOUTPUT}")

    check1 = check_success(outp1.read())
    check2 = check_success(outp2.read())
    check3 = check_eigenvalues("si_bands_pp.out")

    return check1 and check2 and check3  # Check if all were successful


def grid():
    kx_max = 10
    ky_max = 10
    kz_max = 101

    start = 0
    step = 2
    mult = 0.01

    for i in range(kx_max):
        for j in range(ky_max):

            grid = []
            for k in range(start, kz_max, step):
                kx = i*0.1
                ky = j*0.1
                kz = k*mult

                grid.append([kx, ky, kz])

            create_file(grid)
            yield (i, j)


def create_grid():
    g = grid()
    created = 1
    while ret := next(g, None):
        (i,j)=ret
        print(f"\rCurrent [{i}, {j}]", end='')
        calculation_success = calculate_energies()
        create_band_image("si_bands.dat.gnu", f"images/image_{i}_{j}.png")
        copy_dat_file(f"kx_{i}_{j}.dat")
    print("\nDone!")


def check_convergence():

    ecutwfcs = [5,10,15,20,25,30,35,40]

    g = grid()
    while (kpair := next(g, None)):
        (i,j) = kpair

        energies = []
        for ecutwfc in ecutwfcs:

            with open(FILENAME, 'r') as f:
                lines = f.readlines()

            lines[8] = f"    ecutwfc = {ecutwfc}\n"

            with open(FILENAME, 'w') as f:
                f.writelines(lines)

            calculate_energies()

            # Get total energy

            str_energy_line = os.popen(f"grep ! {FILEOUTPUT}").read()

            energy_str = re.findall(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?', str_energy_line)[0]

            energy = float(energy_str)

            energies.append(energy)

            print(f"\rConvergence testing k-pair (i={i}, j={j}) - E={energy}... ", end='')

        if abs(energies[-1] - energies[-2]) < EPSILON_CONVERGENCE:
            print("  Converged!")
        else:
            print("  Not converged!")


if __name__ == "__main__":
    # left_column_values_generate()
    # create_grid()
    check_convergence()
    # create_file([[1,2,3],[4,5,6],[7,8,9]])
