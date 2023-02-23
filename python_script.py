from copy import deepcopy
import os
import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault
import numpy as np

plt.rcParams["figure.dpi"]=150
plt.rcParams["figure.facecolor"]="white"
plt.rcParams["figure.figsize"]=(8, 6)

TEMPLATE = "si.bands.template"
FILENAME = "si.bandspy.in"
FILEOUTPUT = "si.bandspy.out"

def format_row(row, row_num) -> str:
    return f"   {row[0]} {row[1]} {row[2]} {row_num: >2}\n"

def create_file(matrix):
    string_builer = ""

    with open(TEMPLATE, "r") as f:
        string_builer = f.read() + "\n"

    for c, row in enumerate(matrix):
        string_builer += format_row(row, c)

    with open(FILENAME, 'w') as f:
        f.write(string_builer)

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

def calculate_energies():
    os.system("pw.x -i si.bandspy.in")
    os.system("bands.x < si_bands_pp.in > si_bands_pp_py.out ")

def left_column_values_generate():
    M_template = [[0.5, 0.5, 0.5],
         [0.4, 0.4, 0.4],
         [0.3, 0.3, 0.3],
         [0.2, 0.2, 0.2],
         [0.1, 0.1, 0.1],
         [0.0, 0.0, 0.0]]

    variational_template = [[0.0, 0.0, 0.1],
                   [0.0, 0.0, 0.2],
                   [0.0, 0.0, 0.3],
                   [0.0, 0.0, 0.4],
                   [0.0, 0.0, 0.5],
                   [0.0, 0.0, 0.6],
                   [0.0, 0.0, 0.7],
                   [0.0, 0.0, 0.8],
                   [0.0, 0.0, 0.9],
                   [0.0, 0.0, 1.0],
                   [0.0, 0.1, 1.0],
                   [0.0, 0.2, 1.0],
                   [0.0, 0.3, 1.0],
                   [0.0, 0.4, 1.0],
                   [0.0, 0.5, 1.0],
                   [0.0, 0.6, 1.0],
                   [0.0, 0.7, 1.0],
                   [0.0, 0.8, 1.0],
                   [0.0, 0.9, 1.0],
                   [0.0, 1.0, 1.0],
                   [0.0, 0.9, 0.9],
                   [0.0, 0.8, 0.8],
                   [0.0, 0.7, 0.7],
                   [0.0, 0.6, 0.6],
                   [0.0, 0.5, 0.5],
                   [0.0, 0.4, 0.4],
                   [0.0, 0.3, 0.3],
                   [0.0, 0.2, 0.2],
                   [0.0, 0.1, 0.1],
                   [0.0, 0.0, 0.0]]

    for i in range(9):
        M = deepcopy(M_template)
        variational = deepcopy(variational_template)

        for c, row in enumerate(variational):
            row[0] = 0.1*i
            M.append(row)

        create_file(M)
        calculate_energies()
        create_image(f"image{i+1}.png")
        copy_dat_file(f"dat_file{i+1}.dat")

if __name__ == "__main__":
    left_column_values_generate()
