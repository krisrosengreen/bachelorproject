import matplotlib.pyplot as plt
import numpy as np
import re
import os


def create_band_image(filename, output):
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

def read_dat_file(filename):
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


def find_intersections(filename, epsilon=0.1, emin=1, emax=3):
    epsilon = 0.1

    bands = read_dat_file(filename)

    points_intersect = []

    for band1 in bands:
        for band2 in bands:
            if band1 == band2:
                continue

            for i in range(len(band1)):
                if (e1 := band1[i][1]) >= emin and e1 <= emax and (e2 := band2[i][1]) >= emin and e2 <= emax:
                    if abs(band1[i][1] - band2[i][1]) < epsilon:
                        points_intersect.append(band1[i])

    return points_intersect

def plot_bands_and_intersections(filename):
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
