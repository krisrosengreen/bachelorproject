from energy import *
from settings import *
import numpy as np
import os


def generate_points_between(pos1, pos2, num_points):
    x1 = pos1[0]
    x2 = pos2[0]

    y1 = pos1[1]
    y2 = pos2[1]

    z1 = pos1[2]
    z2 = pos2[2]

    Lx = np.linspace(x1, x2, num_points)
    Ly = np.linspace(y1, y2, num_points)
    Lz = np.linspace(z1, z2, num_points)

    return np.c_[Lx, Ly, Lz]


def delete_duplicate_neighbors(matrix):
    indeces = []
    for i in range(len(matrix) - 1):
        c = (matrix[i] - matrix[i+1])
        if c.dot(c) < 0.0001:
            indeces.append(i)
    new_array = []
    for c,row in enumerate(matrix):
        if c not in indeces:
            new_array.append(row)

    return np.array(new_array)


def silicon_band_structure(init_scf_calc=True):
    """
    Look at key silicon band structure.
    Path: L - Gamma - X
    """
    if init_scf_calc:
        init_scf_calculation()

    # Create Grid:

    num_points=10
    L = [0.5,0.5,0.5]
    gamma = [0,0,0]
    X = [0,1,0]
    W = [0.5,1,0]
    U = [0.25,1,0.25]

    LGAMMA = generate_points_between(L, gamma, num_points)
    GAMMAX = generate_points_between(gamma, X, num_points)
    XW = generate_points_between(X, W, num_points)
    WU = generate_points_between(W, U, num_points)
    UGAMMA = generate_points_between(U, gamma, num_points)
    combined_grid = np.vstack((LGAMMA, GAMMAX, XW, WU, UGAMMA))

    combined_grid = delete_duplicate_neighbors(combined_grid)

    # Create file
    create_file(combined_grid)

    # Calculate energies
    calculate_energies()

    # Create image
    create_band_image(BANDS_GNUFILE, "images/si_band.png")


if __name__ == "__main__":
    os.chdir("qefiles/")

    silicon_band_structure(False)
