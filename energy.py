from settings import (FILENAME, PP_FILENAME, FILEOUTPUT,
                      BANDS_GNUFILE, AUTOGRID_FILENAME, OPTLATTICE_FILENAME,
                      TEMPLATE, VALENCE_MAX, PP_FILEOUTPUT,
                      FILENAME30)
from utils import (get_values, PlottingRange, file_change_line, check_within_BZ, IntersectsResponse)
from scipy.optimize import fmin
from utils import inputfile_row_format
from datetime import timedelta  # Calculation ETAs
import matplotlib.pyplot as plt
import numpy as np
import os  # File directory functions
import time
import math
import re  # Parsing numbers from text
import subprocess  # Running CLI QE commands
import uuid  # Create unique filename
import json  # Config file


"""
Useful values
"""


class symmetry_points:
    # kx, ky, kz
    L = [0.5, 0.5, 0.5]
    gamma = [0, 0, 0]
    X = [0, 1, 0]
    W = [0.5, 1, 0]
    U = [0.25, 1, 0.25]


"""
Interface Quantum Espresso software
"""


def get_string_within(s, char1, char2) -> str:
    """
    Find substring in string 's' that have starting char 'char1' and
    ending char 'char2'

    Parameters
    ----------
    s : str
        String to find substring from

    char1 : str
        The starting character

    char2 : str
        The ending character
    """

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
    Check that there are no more or no less than 8 eigenvalues
    in file 'filename'

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
                current += count2-count1
                current += 1
            else:
                if not bad_value:
                    print("Bad eigenvalue!")
                    print("...")
                    print("\n".join(energies[index-5:index]))
                    print(energies[index], count1, count2,
                          "expected", current+1, "\r **")
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
    Run Quantum Espresso console commands to calculate
    energies from Quantum Espresso input file 'si.bandspy.in'
    """

    # First make sure that TMP folder exists
    assert os.path.isdir("tmp"), "Remember to do initial scf calculation!"

    process1 = subprocess.Popen(["pw.x", "-i", FILENAME],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process1.wait()
    process2 = subprocess.Popen(["bands.x", "-i", PP_FILENAME],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outp1, _ = process1.communicate()
    outp2, _ = process2.communicate()

    with open(FILEOUTPUT, "wb") as f:
        f.write(outp1)
    with open(PP_FILEOUTPUT, "wb") as f:
        f.write(outp2)

    check1 = check_success(outp1.decode())
    check2 = check_success(outp2.decode())
    # check3 = check_eigenvalues("si_bands_pp.out")

    # return check1 and check2 and check3  # Check if all were successful
    return check1 and check2  # and check3  # Check if all were successful


def calculate_energies30() -> bool:  # Returns True if successful
    """
    Run Quantum Espresso console commands to calculate
    energies from Quantum Espresso input file 'si.bandspy.in'
    """

    # First make sure that TMP folder exists
    assert os.path.isdir("tmp"), "Remember to do initial scf calculation!"

    process1 = subprocess.Popen(["pw.x", "-i", FILENAME30],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process1.wait()
    process2 = subprocess.Popen(["bands.x", "-i", PP_FILENAME],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outp1, _ = process1.communicate()
    outp2, _ = process2.communicate()

    with open(FILEOUTPUT, "wb") as f:
        f.write(outp1)
    with open(PP_FILEOUTPUT, "wb") as f:
        f.write(outp2)

    check1 = check_success(outp1.decode())
    check2 = check_success(outp2.decode())
    # check3 = check_eigenvalues("si_bands_pp.out")

    # return check1 and check2 and check3  # Check if all were successful
    return check1 and check2  # and check3  # Check if all were successful


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


def create_file_ry30(points):
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

    with open(FILENAME30, 'w') as f:
        f.write(string_builder)


def copy_dat_file(gridname, kx, ky):
    """
    Copy 'si_bands.dat' file to folder qefiles/datfiles/ and change name
    to contain kx and ky values.

    Similarly, copy 'si_bands.dat.gnu' to qefiles/gnufiles/

    Parameters
    ----------
    output_name : str
        Name of file to save image as
    """

    unique_filename = uuid.uuid4().hex

    os.system(f"cp si_bands.dat datfiles/{unique_filename}")
    os.system(f"cp {BANDS_GNUFILE} gnufiles/{unique_filename}.gnu")

    if not os.path.isdir("config"):
        os.makedirs("config")

    with open(f"config/{gridname}", "a") as f:
        f.write(f"{kx} {ky} {unique_filename}\n")


def generate_grid(kx_range=[-1, 1], ky_range=[-1, 1],
                  kz_range=[-1, 1], kx_num_points=41,
                  ky_num_points=41, kz_num_points=41) -> tuple:
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


def generate_grid_BZ(kx_range=[-1, 1], ky_range=[-1, 1],
                     kz_range=[-1, 1], kx_num_points=41,
                     ky_num_points=41, kz_num_points=41):
    """
    Creates a 3D grid. For each point create Quantum Espresso file.
    Similar to generate_grid function, but only generates points
    within the Brillouin zone.

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
            if not check_within_BZ([kx, ky, kz_range[0]]):
                continue

            grid = []
            for kz in np.linspace(*kz_range, kz_num_points):
                if check_within_BZ([kx, ky, kz]):
                    grid.append([kx, ky, kz])

            create_file_ry30(grid)
            yield (kx, ky)


def create_grid(gridname, kx_range=[0, 1], ky_range=[0, 1],
                kz_range=[0, 1], kx_num_points=6,
                ky_num_points=6, kz_num_points=6):
    """
    Creates a grid defined by the function 'generate_grid'.
    Copy '.dat' and '.gnu' files to respective folders.
    Lastly, create images of the band structure.
    """

    with open(f"config/{gridname}_config.json", "w") as f:
        print(f"Writing config file to {gridname}_config.json")
        data = {"kz_offset": kz_range[0]}
        f.write(json.dumps(data))

    g = generate_grid(kx_range, ky_range, kz_range,
                      kx_num_points, ky_num_points, kz_num_points)

    count = 0
    while ret := next(g, None):
        (i, j) = ret
        time_start = time.time()

        calculate_energies()
        create_band_image(BANDS_GNUFILE,
                          f"images/{gridname}_image_{i}_{j}.png")
        copy_dat_file(gridname, kx=i, ky=j)

        time_taken = time.time() - time_start
        remaining_calcs = kx_num_points*ky_num_points - count
        secs_remain = math.floor(time_taken * remaining_calcs)

        print(f"\rCurrent [{i:.4f}, {j:.4f}] - \
              {count}/{kx_num_points*ky_num_points}\
              - Time left {str(timedelta(seconds=secs_remain))}\t", end='')

        count += 1
    print("\nDone!")


def create_quad_BZ_grid(gridname, kx_range=[0, 1], ky_range=[0, 1],
                        kz_range=[0, 1], kx_num_points=6,
                        ky_num_points=6, kz_num_points=6):
    """
    Creates a grid defined by the function 'generate_grid'.
    Copy '.dat' and '.gnu' files to respective folders.
    Lastly, create images of the band structure.

    This is different from create_grid function in that this
    limits the points to within the first Brillouin zone.
    And, this function only uses ecutwfc of 30 Ry instead
    of the 50 Ry used in the other function. 30 Ry should
    be sufficient as the total energy converges before 30
    and is roughly as flat as it is in 50 Ry.

    This should be faster than create_grid
    function when calculating throughout the entire
    Brillouin zone.
    """

    with open(f"config/{gridname}_config.json", "w") as f:
        print(f"Writing config file to {gridname}_config.json")
        data = {"kz_offset": kz_range[0]}
        f.write(json.dumps(data))

    g = generate_grid_BZ(kx_range, ky_range, kz_range,
                         kx_num_points, ky_num_points, kz_num_points)

    # Number of points within this grid
    total_count = 0
    for kx in np.linspace(*kx_range, kx_num_points):
        for ky in np.linspace(*ky_range, ky_num_points):
            if check_within_BZ([kx, ky, kz_range[0]]):
                total_count += 1

    count = 0
    while ret := next(g, None):
        (i, j) = ret
        time_start = time.time()

        calculate_energies30()
        create_band_image(BANDS_GNUFILE,
                          f"images/{gridname}_image_{i}_{j}.png")
        copy_dat_file(gridname, kx=i, ky=j)

        time_taken = time.time() - time_start
        remaining_calcs = total_count - count
        secs_remain = math.floor(time_taken * remaining_calcs)

        print(f"\rCurrent [{i:.4f}, {j:.4f}] - \
              {count}/{total_count}\
              - Time left {str(timedelta(seconds=secs_remain))}\t", end='')

        count += 1
    print("\nDone!")


def init_scf_calculation():
    """
    Do an initial SCF calculation.
    This must be done before any bands calculations.
    """
    print("Beginning initial scf calculation!")
    stdout, stderr = scf_calculation(AUTOGRID_FILENAME)
    print("Done!")


def scf_calculation(filename):
    """
    Perform an SCF calculation on QE input file

    Parameters
    ----------
    filename : str
        Name of QE input file
    """
    process = subprocess.Popen(['pw.x', '-i', filename],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    return process.communicate()


"""
Get useful stuff out of code above
"""


def get_total_energy(filename):
    """
    Get the total energy of from a QE scf calculation in
    a given output file "filename"

    Parameters
    ----------
    filename : str
        Name of output file form which the total energy can be read

    Output
    ------
    float : The total energy
    """
    (stdout, stderr) = scf_calculation(filename)
    lines = stdout.decode().split("\n")
    assert check_success(stdout.decode()), "Unsuccessful in calculating energy"
    for line in lines:
        if "total energy" in line:
            values = get_values(line)
            return values[0]


def get_lattice_energy(lattice_const):
    lattice_const = lattice_const[0]
    LATTICE_CONST_LINE = 9

    def const_format_line(val):
        return f"    celldm(1)={val},\n"

    file_change_line(OPTLATTICE_FILENAME, LATTICE_CONST_LINE,
                     const_format_line(lattice_const))
    energy = get_total_energy(OPTLATTICE_FILENAME)

    print("LC:", lattice_const, "E:", energy)

    return energy


def optimize_lattice_constant(max_iterations=30) -> float:
    """
    Find the optimal lattice constant that gives the lowest energy.
    Returns the lattice constant that gives the lowest energy.

    Parameters
    ----------
    max_iterations : int
        Maximum number of iterations to go through
        to find the best lattice constant

    Returns
    -------
    float
        The best lattice constant the program could find
    """
    best_val = fmin(get_lattice_energy, x0=11, maxiter=max_iterations)

    return best_val[0]


def plot_bands_data(filename, zero_vbm=True):
    """
    From filename plot gnu data

    Parameters
    ----------
    filename : str
        Name of file to plot from
    """
    with open(filename) as f:
        bands_data = f.read()

    bands = bands_data.strip().split("\n\n")

    def floatify(L):
        return [float(L[0]), float(L[1])]

    for band in bands:
        lines = band.split("\n")

        # Convert line of two seperate, spaced numbers into numbers in List
        xy_data = list(map(lambda s: floatify(s.strip().split()), lines))

        xy_data_np = np.array(xy_data)

        if zero_vbm:
            plt.plot(xy_data_np[:, 0], xy_data_np[:, 1] - VALENCE_MAX,
                     linewidth=1, alpha=0.5, color='k')
        else:
            plt.plot(xy_data_np[:, 0], xy_data_np[:, 1],
                     linewidth=1, alpha=0.5, color='k')


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
    plot_bands_data(filename)

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
            Es = re.findall(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?', line)
            energies = list(map(float, Es))
            band.append(energies)

    return bands


def find_intersections(filename, epsilon=0.1,
                       emin=1, emax=VALENCE_MAX,
                       include_conduction=True) -> list:
    """
    Find intersections in 'filename' within energy-min and energy-max
    that are within an energy threshold.

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
        if not include_conduction and c1 > 3:
            continue

        for c2, band2 in enumerate(bands):
            if c1 == c2:
                continue

            if not include_conduction and c2 > 3:
                continue

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

    def floatify(L):
        return [float(L[0]), float(L[1])]

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

    def floatify(L):
        return [float(L[0]), float(L[1])]

    for band in bands:
        lines = band.split("\n")

        xy_data = list(map(lambda s: floatify(s.strip().split()), lines))

        xy_data_np = np.array(xy_data)
        plt.plot(xy_data_np[:, 0], xy_data_np[:, 1],
                 linewidth=1, alpha=0.5, color='k')

    intersects = find_intersections(filename)
    if len(intersects) != 0:
        np_intersecs = np.array(intersects)

        xs_int = np_intersecs[:, 0]
        ys_int = np_intersecs[:, 1]

        plt.scatter(xs_int, ys_int)


def create_image(output_name):
    """
    From file 'si_bands.dat.gnu' create band structure image
    and save file to argument 'output_name'

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
    ecutwfcs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    g = generate_grid()
    while kpair := next(g, None):
        (i, j) = kpair
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
            print(f"\rConvergence testing k-pair (i={i}, j={j}) - E={energy}...", end='')
            if len(energies) >= 2:
                if abs(energies[-1] - energies[-2]) < epsilon_convergence:
                    break

        if abs(energies[-1] - energies[-2]) < epsilon_convergence:
            print("  Converged!")
        else:
            print("  Not converged!")


def plot_brillouin_zone(ax):
    """
    Plot the brillouin zone

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    for xx in fcc_points():
        ax.plot(xx[:, 0], xx[:, 1], xx[:, 2], color='k', lw=1.0)


def get_grid_files(gridname):
    """
    Get the files for the grid

    Parameters
    ----------
    gridname : str
        Name of grid

    Returns
    -------
    list : List of files
    """

    with open(f"config/{gridname}", "r") as f:
        lines = f.readlines()

    # Remove empty lines
    lines = list(filter(lambda x: len(x) != 0, lines))
    data = []

    for line in lines:
        vals = line.split()
        data.append([(float(vals[0]), float(vals[1])), vals[2]])

    return data


def get_grid_kz_offset(gridname):
    """
    Get the kz offset for the grid

    Parameters
    ----------
    gridname : str
        Name of grid

    Returns
    -------
    float : The kz offset
    """
    with open(f"config/{gridname}_config.json", "r") as f:
        data = json.load(f)

    return data["kz_offset"]


def get_3d_intersects_points(gridname, emin=4, emax=VALENCE_MAX, epsilon=0.0001,
                             plotrange=PlottingRange.standard(), include_conduction=True):
    xdata = []
    ydata = []
    zdata = []

    grid_files = get_grid_files(gridname)
    kz_offset = get_grid_kz_offset(gridname)

    L_colors = []

    for grid_file in grid_files:
        kx = grid_file[0][0]
        ky = grid_file[0][1]

        gnu_file = grid_file[1]

        intersections = find_intersections("gnufiles/" + gnu_file + ".gnu",
                                           emin=emin, emax=emax,
                                           epsilon=epsilon,
                                           include_conduction=include_conduction)

        for intersection in intersections:
            # Offset by because of way QE represents this
            kz = intersection[0] + kz_offset

            if plotrange.check_within((kx, ky, kz)):
                xdata.append(kx)
                ydata.append(ky)
                zdata.append(kz)  # Offset by -1

                energy = intersection[1]
                absoluted = abs(energy + 5)

                # Colors
                R = np.clip(1 * (20 - absoluted) / 20, 0, 1)
                G = np.clip(1 * (absoluted) / 20, 0, 1)
                B = 0

                L_colors.append((R, G, B))

    return IntersectsResponse(xdata, ydata, zdata, L_colors)



def plot_3d_intersects(gridname, emin=4, emax=VALENCE_MAX, epsilon=0.0001,
                       colors=False, plotrange=PlottingRange.standard(),
                       include_conduction=True):
    """
    Plot points where bands cross or overlap, within
    energies emin (Energy-minimum) and emax (Energy-max)

    Parameters
    ----------
    emin : float
        The minimum energy

    emax : float
        The maximum energy

    epsilon : float
        The threshold energy difference between bands

    Returns
    -------
    Axis : The axis used to plot the grid
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    resp = get_3d_intersects_points(gridname, emin, emax, epsilon, plotrange, include_conduction)
    (xdata, ydata, zdata) = resp.get_points()
    L_colors = resp.get_colors()

    # plot_brillouin_zone(ax)
    plot_symmetry_points(ax)
    psize = 0.5

    if colors:
        ax.scatter3D(xdata, ydata, zdata, s=psize, c=L_colors)
    else:
        ax.scatter3D(xdata, ydata, zdata, s=psize)
    ax.set_xlabel(r"kx [$\frac{2\pi}{a}$]")
    ax.set_ylabel(r"ky [$\frac{2\pi}{a}$]")
    ax.set_zlabel(r"kz [$\frac{2\pi}{a}$]")

    return (fig, ax)


def plot_3d_energy(gridname, energy):
    """
    Plot all points on bands calculated near 'energy' and within 'epsilon'

    Parameters
    ----------
    energy : float
        The energy used to find points on bands

    epsilon : float
        The energy difference threshold
    """
    ax = plt.axes(projection='3d')

    xdata = []
    ydata = []
    zdata = []

    data_files = get_grid_files(gridname)
    kz_offset = get_grid_kz_offset(gridname)

    print(data_files)

    for data_file in data_files:
        kx = data_file[0][0]
        ky = data_file[0][1]

        gnu_file = data_file[1] + ".gnu"

        intersections = within_energy("gnufiles/" + gnu_file, energy)

        for intersection in intersections:
            xdata.append(kx)
            ydata.append(ky)
            zdata.append(intersection[0] + kz_offset)

    # plot_brillouin_zone(ax)
    plot_symmetry_points(ax)

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
    calculate_energies()

    # Plot bands and get the band with valence maximum and corresponding
    # max value
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
    calculate_energies()

    # Plot bands and get the band with valence maximum and corresponding
    # max value
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
    """
    Finds the band gap of the material
    """
    valence_max = valence_maximum()
    conduct_min = conduction_minimum()
    band_gap = conduct_min - valence_max
    print("Band gap:", band_gap)


def size_point(matrix, point: int) -> float:
    """
    Find quantum espresso representational value to a given point in a matrix
    Parameters
    ----------
    matrix : list
        List containing the points to calculate energies of
    point : int
        The index of the point to which the
        representational value is to be calculated
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
    """
    Plot and label symmetry points in the first Brillouin zone.
    """
    L = symmetry_points.L
    X = symmetry_points.X
    W = symmetry_points.W
    U = symmetry_points.U
    gamma = symmetry_points.gamma

    points = [L, X, W, U, gamma]
    labels = ["L", "X", "W", "U", r"$\Gamma$"]
    for point, label in zip(points, labels):
        ax.scatter3D(point[0], point[1], point[2], c="r", s=5)
        ax.text(point[0], point[1], point[2], label)


if __name__ == "__main__":
    os.chdir("qefiles/")

    val = optimize_lattice_constant(max_iterations=100)
    print("Optimized lattice constant:", val)
