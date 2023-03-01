import matplotlib.pyplot as plt
import numpy as np

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
