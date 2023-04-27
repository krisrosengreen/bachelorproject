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
        if c.dot(c) < 0.00001:
            indeces.append(i)
    new_array = []
    for c,row in enumerate(matrix):
        if c not in indeces:
            new_array.append(row)

    return np.array(new_array)


def silicon_band_structure(init_scf_calc=True):
    """
    Look at key silicon band structure.
    Path: L - Gamma - X - W - U - Gamma
    """
    if init_scf_calc:
        init_scf_calculation()

    # Create Grid:

    num_points=50

    LGAMMA = generate_points_between(symmetry_points.L, symmetry_points.gamma, num_points)
    GAMMAX = generate_points_between(symmetry_points.gamma, symmetry_points.X, num_points)
    XW = generate_points_between(symmetry_points.X, symmetry_points.W, num_points)
    WU = generate_points_between(symmetry_points.W, symmetry_points.U, num_points)
    UGAMMA = generate_points_between(symmetry_points.U, symmetry_points.gamma, num_points)
    combined_grid = np.vstack((LGAMMA, GAMMAX, XW, WU, UGAMMA))

    combined_grid = delete_duplicate_neighbors(combined_grid)

    # Create file
    create_file(combined_grid)

    # Calculate energies
    calculate_energies()

    xval_L = size_point(combined_grid, 0)
    xval_gamma = size_point(combined_grid, 49)
    xval_X = size_point(combined_grid, 98)
    xval_W = size_point(combined_grid, 147)
    xval_U = size_point(combined_grid, 196)
    xval_gamma_last = size_point(combined_grid, len(combined_grid))

    plt.xticks([xval_L, xval_gamma, xval_X, xval_W, xval_U, xval_gamma_last], ["L", r"$\Gamma$", "X", "W", "U", r"$\Gamma$"])
    for x in [xval_L, xval_gamma, xval_X, xval_W, xval_U, xval_gamma_last]:
        plt.axvline(x, lw=0.5, linestyle="--", c='k')

    # Create image
    create_band_image(BANDS_GNUFILE, "figures/si_band.png")


def VBM_figure():
    vbm_energy = valence_maximum()
    plot_3d_energy(vbm_energy)
    plt.savefig("figures/valence_maximum_3d_plot.png")


def CBM_figure():
    cbm_energy = conduction_minimum()
    plot_3d_energy(cbm_energy)
    plt.savefig("figures/conduction_3d_plot.png")


def create_figures():
    print("Creating figure for silicon band structure")
    silicon_band_structure(True)

    print("Creating figure for valence band maximum")
    VBM_figure()

    print("Creating figure for conduction band minimum")
    CBM_figure()


def dispersion_XW():
    """
    Look at the dispersion in the X-W symmetry line
    """
    X = symmetry_points.X
    W = symmetry_points.W

    XW = generate_points_between(X.copy(), W.copy(), 50)
    WX= generate_points_between(W.copy(), X.copy()*2, 50)

    XW2 = generate_points_between(X.copy()*2, W.copy()*2, 50)
    WX2 = generate_points_between(W.copy()*2, X.copy()*3, 50)

    combined_grid = np.vstack((XW, WX, XW2, WX2))
    combined_grid = delete_duplicate_neighbors(combined_grid)

    create_file(combined_grid)

    calculate_energies()

    create_band_image(BANDS_GNUFILE, "figures/dispersion_XW.png")


def nodal_lines():
    # Nodal line in Gamma-L direction
    plot_3d_intersects(emin=4, emax=6)
    plt.title(r"Energy range: [4, 5]. Nodal line in direction $\Gamma$-L")
    plt.savefig("figures/NL_gamma-L.png")


if __name__ == "__main__":
    os.chdir("qefiles/")

    # silicon_band_structure(init_scf_calc=False)
    # create_figures()
    # dispersion_XW()
    nodal_lines()
