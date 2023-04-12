from energy import *
from settings import *
import numpy as np
import os

def silicon_band_structure():
    """
    Look at key silicon band structure.
    Path: L - Gamma - X
    """
    # init_scf_calculation()

    # Create Grid:
    num_points = 20
    grid1 = np.zeros((num_points, 3))
    grid1[:, 2] = np.linspace(1, 0, num_points)
    grid2 = np.ones((20, 3)) * np.linspace(0, 0.5, num_points)[np.newaxis].T
    combined_grid = np.vstack((grid1, grid2))

    # Create file
    create_file(combined_grid)

    # Calculate energies
    calculate_energies()

    # Create image
    create_band_image(BANDS_GNUFILE, "si_band.png")


if __name__ == "__main__":
    os.chdir("qefiles/")

    silicon_band_structure()
