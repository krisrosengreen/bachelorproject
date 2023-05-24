from energy import *
from settings import *
from utils import limit_first_quad
import numpy as np
import matplotlib.pyplot as plt
import os


def create_figures():
    print("Creating figure for silicon band structure")
    silicon_band_structure(True)

    print("Creating figure for valence band maximum")
    VBM_figure()

    print("Creating figure for conduction band minimum")
    CBM_figure()

    print("Creating figure for dispresion along X-W direction")
    dispersion_XW()

    print("Creating figure for nodal lines along high-symmetry lines")
    trivial_nodal_lines()

    print("Creating figure for nodal lines across the entire Brillouin zone")
    nodal_lines()

    print("Creating figure for optimized lattice constant")
    lattice_constant_optimize()

    print("Creating figure for high-res nodal lines around symmetry points")
    highres_symmetry_points()

    print("Creating figure for kinetic energy cutoff and total energy")
    energy_convergence()


def generate_points_between(pos1, pos2, num_points):
    """
    Generate a series of points along the line connecting,
    vector pos1 and vector pos2 with num_points number of points.

    Parameters
    ----------
    pos1 : list
        Vector of pos1
    pos2 : list
        Vector of pos2
    num_points : int
        Number of points alon the connecting line

    Returns
    -------
    list : List of points along the connecting line
    """
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
    """
    Delete duplicate points in the matrix

    Parameters
    ----------
    matrix : 2D-list
        Matrix containing the points to be filtered

    Returns
    -------
    Returns matrix without any duplicate neighboring points
    """
    indeces = []
    for i in range(len(matrix) - 1):
        c = (matrix[i] - matrix[i+1])
        if c.dot(c) < 0.00001:
            indeces.append(i)
    new_array = []
    for c, row in enumerate(matrix):
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

    # Valence maximum energy
    valence_max = valence_maximum()

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
    # create_band_image(BANDS_GNUFILE, "figures/si_band.pdf")
    bands = get_bands(BANDS_GNUFILE)

    for band in bands:
        plt.plot(band[:, 0], band[:, 1] - valence_max, linewidth=1, alpha=0.5, color='k')

    plt.ylabel("E [eV]")
    plt.savefig("figures/si_band.pdf")
    plt.clf()
    plt.cla()


def VBM_figure():
    vbm_energy = valence_maximum()
    plot_3d_energy(vbm_energy)
    plt.savefig("figures/valence_maximum_3d_plot.pdf")


def CBM_figure():
    cbm_energy = conduction_minimum()
    plot_3d_energy(cbm_energy)
    plt.savefig("figures/conduction_3d_plot.pdf")


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

    create_band_image(BANDS_GNUFILE, "figures/dispersion_XW.pdf")


def nodal_lines():
    # Nodal line in Gamma-L direction
    plot_3d_intersects(emin=4, emax=6)
    plt.title(r"Energy range: [4, 5]. Nodal line in direction $\Gamma$-L")
    plt.savefig("figures/NL_gamma-L.pdf")


def trivial_nodal_lines():
    (fig, ax) = plot_3d_intersects("grid100points", emin=-15,
                       emax=VALENCE_MAX + 0.1, colors=False, epsilon=0.01)
    plot_fcc(ax)
    limit_first_quad(ax)
    plt.savefig("figures/trivial_nodal_lines.pdf")
    plt.cla()
    plt.clf()


def lattice_constant_optimize():
    L = []
    wrap_func = get_lattice_energy

    def wrapper(x):
        resp = wrap_func(x)
        L.append([x[0], resp])

        return resp

    resp = fmin(wrapper, x0=11, maxiter=30)
    npL = np.array(L)

    plt.scatter(npL[:, 0], npL[:, 1])
    plt.ylabel("Total Energy [Ry]")
    plt.xlabel("Lattice Constant")
    plt.savefig("figures/lattice_constant.pdf")


def highres_symmetry_points():
    files = ["aroundGAMMA", "aroundL", "aroundW", "aroundX", "aroundU"]
    filetitle = [r"Around $\Gamma$", "Around L",
                 "Around W", "Around X", "Around U"]
    points = [symmetry_points.gamma, symmetry_points.L,
              symmetry_points.W, symmetry_points.X,
              symmetry_points.U]

    offset = 0.1

    def Loffset(x):
        return [x-offset, x+offset]

    emin = -15
    for file, filetitle, point in zip(files, filetitle, points):
        (fig, ax) = plot_3d_intersects(file, emin=emin, emax=VALENCE_MAX,
                                       include_conduction=False, epsilon=0.001)
        ax.set_title(f"{filetitle} at energy interval [{emin}, {VALENCE_MAX}]")
        # plt.show()

        xlim = Loffset(point[0])
        ylim = Loffset(point[1])
        zlim = Loffset(point[2])
        
        ax.axes.set_xlim3d(xlim[0], xlim[1])
        ax.axes.set_ylim3d(ylim[0], ylim[1])
        ax.axes.set_zlim3d(zlim[0], zlim[1])

        fig.savefig(f"figures/{file}.pdf")
        plt.clf()
        plt.cla()
    print("Done!")


def energy_convergence():
    ECONVERGE_FILENAME = "si.scf.Econverge"
    
    Es = [5, 6, 7, 8, 9, 10, 12, 14,  15, 20, 25, 30, 35, 40, 45, 50]
    points = []
    for ecutwfc in Es:
        start = time.time()
        file_change_line(ECONVERGE_FILENAME, 12, "    ecutwfc" + f" = {ecutwfc},\n")
        Etot = get_total_energy(ECONVERGE_FILENAME)
        points.append([ecutwfc, Etot])
        Ttaken = time.time() - start
        print(f"ecutwfc {ecutwfc} total energy {Etot} time taken {Ttaken}")

    print(points)
    with open("../output/Econverge.txt", "w") as f:
        f.write(str(points))

    np_points = np.array(points)
    plt.plot(np_points[:, 0], np_points[:, 1])
    plt.xlabel("Kinetic energy cutoff [Ry]")
    plt.ylabel("Total energy [Ry]")
    plt.savefig("figures/ecutwfc.pdf")


if __name__ == "__main__":
    os.chdir("qefiles/")

    # energy_convergence()
    # highres_symmetry_points()
    # lattice_constant_optimize()
    # trivial_nodal_lines()
    # silicon_band_structure(init_scf_calc=False)
    trivial_nodal_lines()
    # create_figures()
    # dispersion_XW()
    # nodal_lines()
