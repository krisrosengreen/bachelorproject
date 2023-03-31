import matplotlib.pyplot as plt
import numpy as np
import re
import os


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

    for band1 in bands:

        for band2 in bands:
            if np.array_equal(band1, band2):
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
    plt.show()


if __name__ == "__main__":
    os.chdir("qefiles/")

    plot_bands_and_intersections("si_bands.dat.gnu")
