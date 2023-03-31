from energy import *
from qe import *
from copy import deepcopy
from subprocess import check_output
from matplotlib import rcParamsDefault
from mpl_toolkits import mplot3d
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

BANDS_GNUFILE = "si_bands.dat.gnu"

FORMATTING_DECIMALS = 4

EPSILON_CONVERGENCE = 0.05


def inputfile_row_format(row, row_num) -> str:
    FD = FORMATTING_DECIMALS
    return f"   {row[0]:.{FD}f} {row[1]:.{FD}f} {row[2]:.{FD}f} {row_num: >3}\n"


def create_file(matrix):
    string_builder = ""

    with open(TEMPLATE, "r") as f:
        string_builder = f.read() + "\n"

    string_builder += f"   {len(matrix)}\n"

    for c, row in enumerate(matrix):
        string_builder += inputfile_row_format(row, c)

    with open(FILENAME, 'w') as f:
        f.write(string_builder)


def create_image(output_name):
    # load data
    data = np.loadtxt(BANDS_GNUFILE)

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
    os.system(f"cp {BANDS_GNUFILE} gnufiles/{output_name}.gnu")


def generate_grid():
    kx_range = [-1, 1]
    ky_range = [-1, 1]
    kz_range = [-1, 1]

    kx_num_points = 40
    ky_num_points = 40
    kz_num_points = 100

    for kx in np.linspace(*kx_range, kx_num_points):
        for ky in np.linspace(*ky_range, ky_num_points):

            grid = []
            for kz in np.linspace(*kz_range, kz_num_points):
                grid.append([kx, ky, kz])

            create_file(grid)
            yield (kx, ky)


def create_grid():
    g = generate_grid()
    while ret := next(g, None):
        (i,j)=ret
        print(f"\rCurrent [{i:.4f}, {j:.4f}]", end='')
        calculation_success = calculate_energies()
        create_band_image(BANDS_GNUFILE, f"images/image_{i}_{j}.png")
        copy_dat_file(f"kx_{i}_ky_{j}.dat")
    print("\nDone!")


def check_convergence():
    ecutwfcs = [5,10,15,20,25,30,35,40,45,50,55,60]
    g = generate_grid()
    while kpair := next(g, None):
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
            if len(energies) >= 2:
                if abs(energies[-1] - energies[-2]) < EPSILON_CONVERGENCE:
                    break

        if abs(energies[-1] - energies[-2]) < EPSILON_CONVERGENCE:
            print("  Converged!")
        else:
            print("  Not converged!")


def plot_3d_intersects(emin=4.2, emax=5.2, epsilon=0.01):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xdata = []
    ydata = []
    zdata = []

    gnu_files = os.listdir("gnufiles")

    for gnu_file in gnu_files:
        splitted_no_fextension = ".".join(gnu_file.split('.')[:-2]).split("_")

        kx = float(splitted_no_fextension[1])
        ky = float(splitted_no_fextension[3])

        print(gnu_file, kx, ky)

        intersections = find_intersections(f"gnufiles/" + gnu_file, emin=emin, emax=emax, epsilon=epsilon)

        for intersection in intersections:
            xdata.append(kx)
            ydata.append(ky)
            zdata.append(intersection[0])

    ax.scatter3D(xdata, ydata, zdata)
    plt.show()


def valence_maximum():  # Should lie in Gamma - L direction
    # First create grid from (0,0,0) to (0.5, 0.5, 0.5)
    num_points = 20
    grid = np.ones((20, 3)) * np.linspace(0, 0.5, num_points)[np.newaxis].T
    create_file(grid)
    success = calculate_energies()

    # Plot bands and get the band with valence maximum and corresponding max value
    bands = get_bands(BANDS_GNUFILE)
    for c, band in enumerate(bands):
        col = "r"
        if c == 3:
            col = "g"
            valence_maximum = np.max(band[:, 1])
            print("Valence max: ", valence_maximum)
        plt.plot(band[:, 0], band[:, 1], c=col)
    plt.show()


if __name__ == "__main__":
    os.chdir("qefiles/")

    # left_column_values_generate()
    # create_grid()
    # check_convergence()
    # create_file([[1,2,3],[4,5,6],[7,8,9]])
    plot_3d_intersects()
    # valence_maximum()
