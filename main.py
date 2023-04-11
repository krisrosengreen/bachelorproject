from energy import *
from qe import *
from copy import deepcopy
from subprocess import check_output
from matplotlib import rcParamsDefault
from mpl_toolkits import mplot3d
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import os
import re
import subprocess


plt.rcParams["figure.dpi"]=150
plt.rcParams["figure.facecolor"]="white"
plt.rcParams["figure.figsize"]=(8, 6)

TEMPLATE = "si.bands.template"
FILENAME = "si.bandspy.in"
FILEOUTPUT = "si.bandspy.out"

AUTOGRID_FILENAME = "si.bands.autogrid"

PP_FILENAME = "si_bands_pp.in"
PP_FILEOUTPUT = "si_bands_pp.out"

BANDS_GNUFILE = "si_bands.dat.gnu"

FORMATTING_DECIMALS = 4


def inputfile_row_format(row, row_num) -> str:
    """
    Format row list to a string, the same way Quantum Espresso does it.

    Parameters
    ----------
    row : list
        point to be formatted

    row_num : int
        The row-number this row corresponds to
    """
    FD = FORMATTING_DECIMALS
    return f"   {row[0]:.{FD}f} {row[1]:.{FD}f} {row[2]:.{FD}f} {row_num: >3}\n"


def create_file(points):
    """
    Create a Quantum Espresso input file from given points

    Parameters
    ----------
    points - list
        Contains points to be used in Quantum Espresso to create band structure
    """
    string_builder = ""

    with open(TEMPLATE, "r") as f:
        string_builder = f.read() + "\n"

    string_builder += f"   {len(points)}\n"

    for c, row in enumerate(points):
        string_builder += inputfile_row_format(row, c)

    with open(FILENAME, 'w') as f:
        f.write(string_builder)


def create_image(output_name):
    """
    From file 'si_bands.dat.gnu' create band structure image and save file to argument 'output_name'

    Parameters
    ----------
    output_name : str
        Name of file to save image as
    """
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
    """
    Copy 'si_bands.dat' file to folder qefiles/datfiles/ and change name to contain
    kx and ky values.

    Similarly, copy 'si_bands.dat.gnu' to qefiles/gnufiles/

    Parameters
    ----------
    output_name : str
        Name of file to save image as
    """
    os.system(f"cp si_bands.dat datfiles/{output_name}")
    os.system(f"cp {BANDS_GNUFILE} gnufiles/{output_name}.gnu")


def generate_grid(kx_range=[-1, 1], ky_range=[-1, 1], kz_range=[-1, 1], kx_num_points=41, ky_num_points=41, kz_num_points=41) -> tuple:
    """
    Creates a 3D grid. For each point create Quantum Espresso file.

    Parameters
    ----------
    kx_range : list
        List of size 2. The kx-range

    ky_range : list
        Similar to kx_range but in y direction.

    kz_range : list
        Similar to kx_range but in z direction

    kx_num_points : int
        Number of points in x-direction

    ky_num_points : int
        Number of points in y-direction

    kz_num_points : int
        Number of points in z-direction
    """

    for kx in np.linspace(*kx_range, kx_num_points):
        for ky in np.linspace(*ky_range, ky_num_points):

            grid = []
            for kz in np.linspace(*kz_range, kz_num_points):
                grid.append([kx, ky, kz])

            create_file(grid)
            yield (kx, ky)


def create_grid():
    """
    Creates a grid defined by the function 'generate_grid'. Copy '.dat' and '.gnu' files to respective folders.
    Lastly, create images of the band structure.
    """

    kx_range = [-1,1]
    ky_range = [-1,1]
    kz_range = [-1,1]

    kx_num_points = 41
    ky_num_points = 41
    kz_num_points = 41

    g = generate_grid(kx_range, ky_range, kz_range, kx_num_points, ky_num_points, kz_num_points)

    count = 0
    print(f"Starting", end="")
    while ret := next(g, None):
        (i,j)=ret
        time_start = time.time()

        calculation_success = calculate_energies()
        create_band_image(BANDS_GNUFILE, f"images/image_{i}_{j}.png")
        copy_dat_file(f"kx_{i}_ky_{j}.dat")

        time_taken = time.time() - time_start
        remaining_calcs = kx_num_points*ky_num_points - count
        secs_remain = math.floor(time_taken * remaining_calcs)
        print(f"\rCurrent [{i:.4f}, {j:.4f}] - {count}/{kx_num_points*ky_num_points} - Time left {str(timedelta(seconds=secs_remain))}\t", end='')
        count += 1
    print("\nDone!")


def read_dat_file():
    with open("si_bands.dat", "r") as f:
        lines = f.readlines()

    points = []

    for i in range(1, len(lines), 2):
        kx,ky,kz = list(map(float, lines[i].split()))
        #print(kx, ky, kz)
        points.append((kx,ky,kz))

    points = np.array(points)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_zlabel("kz")
    plt.show()


def init_scf_calculation():
    print("Beginning initial scf calculation!")
    process = subprocess.Popen(['pw.x', '-i', AUTOGRID_FILENAME],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout,stderr = process.communicate()
    print("Done!")


def check_convergence(epsilon_convergence=0.05):
    """
    Check that energies converge

    Parameters
    ----------
    epsilon_convergence : float
        The difference between points must be below this threshold to converge
    """
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
                if abs(energies[-1] - energies[-2]) < epsilon_convergence:
                    break

        if abs(energies[-1] - energies[-2]) < epsilon_convergence:
            print("  Converged!")
        else:
            print("  Not converged!")


def plot_3d_intersects(emin=5.1, emax=5.19, epsilon=0.01):
    """
    Plot points where bands cross or overlap, within energies emin (Energy-minimum) and emax (Energy-max)

    Parameters
    ----------
    emin : float
        The minimum energy

    emax : float
        The maximum energy

    epsilon : float
        The threshold energy difference between bands
    """
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

        intersections = find_intersections(f"gnufiles/" + gnu_file, emin=emin, emax=emax, epsilon=epsilon)

        for intersection in intersections:
            xdata.append(kx)
            ydata.append(ky)
            zdata.append(intersection[0])

    ax.scatter3D(xdata, ydata, zdata)
    plt.show()


def plot_3d_energy(energy, epsilon=0.01):
    """
    Plot all points on bands calculated near 'energy' and within 'epsilon'

    Parameters
    ----------
    energy : float
        The energy used to find points on bands

    epsilon : float
        The energy difference threshold
    """
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

        intersections = within_energy(f"gnufiles/" + gnu_file, energy)

        for intersection in intersections:
            xdata.append(kx)
            ydata.append(ky)
            zdata.append(intersection[0])

    ax.scatter3D(xdata, ydata, zdata)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_zlabel("kz")
    plt.show()


def valence_maximum(show=False):
    """
    Finds the valence band maximum in the Gamma-L direction.
    """

    # Should lie in Gamma - L direction
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
            valence_max = np.max(band[:, 1])
        if show:
            plt.plot(band[:, 0], band[:, 1], c=col)
    if show:
        plt.show()
    return valence_max


def conduction_minimum(show=False):
    """
    Finds the conduction minimum in the Gamma - X direction
    """

    # Should lie in Gamma - L direction
    # First create grid from (0,0,0) to (0.5, 0.0, 0.0)
    num_points = 20
    grid = np.zeros((num_points, 3))
    grid[:, 0] = np.linspace(-1, 1, num_points)
    create_file(grid)
    success = calculate_energies()

    # Plot bands and get the band with valence maximum and corresponding max value
    bands = get_bands(BANDS_GNUFILE)
    for c, band in enumerate(bands):
        col = "r"
        if c == 4:
            col = "g"
            conduction_min = np.max(band[:, 1])
        if show:
            plt.plot(band[:, 0], band[:, 1], c=col)
    if show:
        plt.show()
    return conduction_min


if __name__ == "__main__":
    os.chdir("qefiles/")

    # check_convergence()
    # plot_3d_intersects()
    # plot_3d_energy(5.33)
    # print(valence_maximum(show=True))
    # conduction_minimum(show=True)
    # init_scf_calculation()
    create_grid()
    # read_dat_file()
