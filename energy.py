from settings import *
from utils import *
from copy import deepcopy
from scipy.optimize import fmin
from subprocess import check_output
from mpl_toolkits import mplot3d
from utils import inputfile_row_format
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math
import re
import subprocess


"""
Useful values
"""


class symmetry_points:
    L = [0.5,0.5,0.5]
    gamma = [0,0,0]
    X = [0,1,0]
    W = [0.5,1,0]
    U = [0.25,1,0.25]


"""
Interface Quantum Espresso software
"""


def get_string_within(s, char1, char2) -> str:
    """
    Find substring in string 's' that have starting char 'char1' and ending char 'char2'

    Parameters
    ----------
    s : str
        String to find substring from

    char1 : str
        The starting character

    char2 : str
        The ending character
    """
    string_builder = ""

    start = -1
    end = -1

    for i in range(len(s)):
        if s[i] == char1:
            start = i

        if s[i] == char2:
            end = i
            break

    return s[start+1:end]


def check_eigenvalues(filename) -> bool:  # There has to be 8 eigenvalues
    """
    Check that there are no more or no less than 8 eigenvalues in file 'filename'

    Parameters
    ----------
    filename : str
        Name and path to file
    """
    with open(filename, 'r') as f:
        file_content = f.read()

        lines = file_content.split("\n")
        energies = list(filter(lambda s: "e(" in s, lines))

        current = 0
        bad_value = False

        for index, energy in enumerate(energies):
            energy_eigenvalue_count = get_string_within(energy, '(', ')')
            count1_char = energy_eigenvalue_count.strip()[0]
            count2_char = energy_eigenvalue_count.strip()[-1]

            count1 = int(count1_char)
            count2 = int(count2_char)

            if count1 == current+1 or count2 == current+1:
                current+= count2-count1
                current+=1
            else:
                if not bad_value:
                    print("Bad eigenvalue!")
                    print("...")
                    print("\n".join(energies[index-5:index]))
                    print(energies[index], count1, count2, "expected", current+1 ,"\r **")
                    print("\n".join(energies[index+1:index+5]))
                    print("...")
                    bad_value = True
                else:
                    bad_value = False
                    current = max(count1, count2)
            if current == 8 or count2 == 8:
                current = 0
    return not bad_value


def check_success(espresso_output) -> bool:
    """
    Check that the output given from Quantum Espresso indicates that the job
    finished successfully

    Parameters
    ----------
    espresso_output : str
        Quantum Espresso command output
    """
    return "JOB DONE" in espresso_output


def calculate_energies() -> bool:  # Returns True if successful
    """
    Run Quantum Espresso console commands to calculate energies from Quantum Espresso
    input file 'si.bandspy.in'
    """

    # First make sure that TMP folder exists
    assert os.path.isdir("tmp"), "No tmp folder found! Remember to do initial scf calculation!"

    process1 = subprocess.Popen(["pw.x", "-i", FILENAME], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process1.wait()
    process2 = subprocess.Popen(["bands.x", "-i", PP_FILENAME], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outp1,_ = process1.communicate()
    outp2,_ = process2.communicate()

    with open(FILEOUTPUT, "wb") as f:
        f.write(outp1)
    with open(PP_FILEOUTPUT, "wb") as f:
        f.write(outp2)

    check1 = check_success(outp1.decode())
    check2 = check_success(outp2.decode())
    check3 = check_eigenvalues("si_bands_pp.out")

    # return check1 and check2 and check3  # Check if all were successful
    return check1 and check2 and check3  # Check if all were successful


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

    kx_num_points = 81
    ky_num_points = 81
    kz_num_points = 81

    g = generate_grid(kx_range, ky_range, kz_range, kx_num_points, ky_num_points, kz_num_points)

    count = 0
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


def init_scf_calculation():
    print("Beginning initial scf calculation!")
    stdout, stderr = scf_calculation(AUTOGRID_FILENAME)
    print("Done!")


def scf_calculation(filename):
    process = subprocess.Popen(['pw.x', '-i', filename],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    return process.communicate()


"""
Get useful stuff out of code above
"""


def get_energy(filename):
    (stdout, stderr) = scf_calculation(filename)
    lines = stdout.decode().split("\n")
    assert check_success(stdout.decode()), "Not successful in calculating energy!"
    for line in lines:
        if "total energy" in line:
            values = get_values(line)
            return values[0]


def file_change_line(filename, numline, newline):
    with open(filename, "r") as f:
        lines = f.readlines()
        lines[numline] = newline

    with open(filename, "w") as f:
        f.writelines(lines)


def optimize_lattice_constant(max_iterations=30) -> float:
    """
    Find the optimal lattice constant that gives the lowest energy.
    Returns the lattice constant that gives the lowest energy.

    Parameters
    ----------
    max_iterations : int
        Maximum number of iterations to go through to find the best lattice constant

    Returns
    -------
    float
        The best lattice constant the program could find
    """

    def get_lattice_energy(lattice_const):
        lattice_const = lattice_const[0]
        LATTICE_CONST_LINE = 9
        const_format_line = lambda val: f"    celldm(1)={val},\n"
        file_change_line(OPTLATTICE_FILENAME, LATTICE_CONST_LINE, const_format_line(lattice_const))
        energy = get_energy(OPTLATTICE_FILENAME)

        print("LC:", lattice_const, "E:", energy)

        return energy

    best_val = fmin(get_lattice_energy, x0=11, maxiter=max_iterations)

    return best_val[0]


def create_band_image(filename, output):
    """
    From file given by argument 'filename' create band structure image
    and save file to argument 'output'

    Parameters
    ----------
    filename : str
        Name and path to gnu file to be used to create image

    output : str
        Name of file to save image as
    """
    with open(filename) as f:
        bands_data = f.read()

    bands = bands_data.strip().split("\n\n")
    floatify = lambda L: [float(L[0]), float(L[1])]

    for band in bands:
        lines = band.split("\n")

        xy_data = list(map(lambda s: floatify(s.strip().split()), lines))

        xy_data_np = np.array(xy_data)
        plt.plot(xy_data_np[:, 0], xy_data_np[:, 1], linewidth=1, alpha=0.5, color='k')

    plt.savefig(output)
    plt.clf()
    plt.cla()


def read_dat_file(filename) -> list:
    """
    Reads a dat file from argument 'filename' to get bands

    Parameters
    ----------
    filename : str
        Name of file
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    bands = []
    band = []

    for line in lines:
        if line == "\n":
            bands.append(band)
            band = []
        else:
            text_energies = re.findall(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?', line)
            energies = list(map(float, text_energies))
            band.append(energies)

    return bands


def find_intersections(filename, epsilon=0.1, emin=1, emax=3) -> list:
    """
    Find intersections in 'filename' within energy-min and energy-max that are within
    an energy threshold

    Parameters
    ----------
    filename : str
        Name of file to load

    epsilon : float
        Energy difference threshold

    emin : float
        Minimum energy

    emax : float
        Maximum energy
    """
    bands = np.array(read_dat_file(filename))

    points_intersect = []

    for c1, band1 in enumerate(bands):

        for c2, band2 in enumerate(bands):
            if c1 == c2:
                continue

            # idxs = np.argwhere(np.diff(np.sign(band1[:, 1] - band2[:, 1]))).flatten()
            idxs = np.where(np.abs(band1[:, 1] - band2[:, 1]) < epsilon)[0]

            for idx in idxs:
                if emin <= (yval := band1[idx][1]) and yval <= emax:
                    points_intersect.append(band1[idx])

    return points_intersect


def within_energy(filename, energy, epsilon=0.1) -> list:
    """
    Find all points that are within energy and energy threshold

    Parameters
    ----------
    filename : str
        Name of file

    energy : float
        Points with energy similar to this is returned

    epsilon : float
        Energy threshold
    """
    bands = np.array(read_dat_file(filename))

    points_intersect = []

    for band1 in bands:

        # idxs = np.argwhere(np.diff(np.sign(band1[:, 1] - band2[:, 1]))).flatten()
        idxs = np.where(np.abs(band1[:, 1] - energy) < epsilon)[0]

        for idx in idxs:
            points_intersect.append(band1[idx])

    return points_intersect


def get_bands(filename) -> list:
    """
    From filename get bands

    Parameters
    ----------
    filename : str
        Name of file
    """
    with open(filename) as f:
        bands_data = f.read()

    txt_bands = bands_data.strip().split("\n\n")
    floatify = lambda L: [float(L[0]), float(L[1])]

    bands = []

    for txt_band in txt_bands:
        lines = txt_band.split("\n")

        xy_data = list(map(lambda s: floatify(s.strip().split()), lines))

        xy_data_np = np.array(xy_data)
        bands.append(xy_data_np)

    return bands


def plot_bands_and_intersections(filename):
    """
    Plot bands and band intersections

    Parameters
    ----------
    filename : str
        File to load bands from
    """
    with open(filename) as f:
        bands_data = f.read()

    bands = bands_data.strip().split("\n\n")
    floatify = lambda L: [float(L[0]), float(L[1])]

    for band in bands:
        lines = band.split("\n")

        xy_data = list(map(lambda s: floatify(s.strip().split()), lines))

        xy_data_np = np.array(xy_data)
        plt.plot(xy_data_np[:, 0], xy_data_np[:, 1], linewidth=1, alpha=0.5, color='k')

    intersects = find_intersections(filename)
    if len(intersects) != 0:
        np_intersecs = np.array(intersects)

        xs_int = np_intersecs[:,0]
        ys_int = np_intersecs[:,1]

        plt.scatter(xs_int, ys_int)


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


def plot_brillouin_zone(ax):
    for xx in fcc_points():
        ax.plot(xx[:, 0], xx[:, 1], xx[:, 2], color='k', lw=1.0)


def plot_3d_intersects(emin=4, emax=5, epsilon=0.01):
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
            zdata.append(intersection[0] - 1)  # Offset by -1

    # plot_brillouin_zone(ax)
    plot_symmetry_points(ax)

    ax.scatter3D(xdata, ydata, zdata, s=2)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_zlabel("kz")


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

    # plot_brillouin_zone(ax)

    ax.scatter3D(xdata, ydata, zdata)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_zlabel("kz")


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
    grid[:, 0] = np.linspace(0, 1, num_points)
    create_file(grid)
    success = calculate_energies()

    # Plot bands and get the band with valence maximum and corresponding max value
    bands = get_bands(BANDS_GNUFILE)
    for c, band in enumerate(bands):
        col = "r"
        if c == 4:
            col = "g"
            conduction_min = np.min(band[:, 1])
        if show:
            plt.plot(band[:, 0], band[:, 1], c=col)
    if show:
        plt.show()
    return conduction_min


def band_gap():
    valence_max = valence_maximum()
    conduct_min = conduction_minimum()
    band_gap = conduct_min - valence_max
    print("Band gap:", band_gap)  # This becomes 2.23 eV - Which is very weird? This value should be underestimated


def size_point(matrix, point: int) -> float:
    """
    Find quantum espresso representational value to a given point in a matrix
    Parameters
    ----------
    matrix : list
        List containing the points to calculate energies of
    point : int
        The index of the point to which the representational value is to be calculated
    Return
    ------
    float : The representational value
    """
    summed = 0
    for i in range(1, point):
        vec = matrix[i] - matrix[i-1]
        summed += np.sqrt(vec.dot(vec))

    return summed


def plot_symmetry_points(ax):
    points = np.array([symmetry_points.gamma,
    symmetry_points.L,
    symmetry_points.U,
    symmetry_points.X,
    symmetry_points.W])
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c="r", s=5)


if __name__ == "__main__":
    os.chdir("qefiles/")

    val = optimize_lattice_constant(max_iterations=100)
    print("Optimized lattice constant:", val)
