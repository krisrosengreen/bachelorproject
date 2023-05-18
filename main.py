from energy import *
from matplotlib import rcParamsDefault
import matplotlib.pyplot as plt
import os


plt.rcParams["figure.dpi"]=150
plt.rcParams["figure.facecolor"]="white"
plt.rcParams["figure.figsize"]=(8, 6)


def init_setup():
    required_folders = ["images", "datfiles", "gnufiles"]
    items = os.listdir()
    for rf in required_folders:
        if rf not in items:
            os.mkdir(rf)


def create_grids_around_points():
    GAMMA = symmetry_points.gamma
    X = symmetry_points.X
    U = symmetry_points.U
    W = symmetry_points.W
    L = symmetry_points.L

    num_points=13
    offset = 0.05
    Loffset = lambda x: [x-0.05, x+0.05]

    points = {"kx_num_points": num_points, "ky_num_points": num_points, "kz_num_points": num_points}
    print("Now aroundL 1/5")
    create_grid("aroundL", kx_range=Loffset(L[0]), ky_range=Loffset(L[1]), kz_range=Loffset(L[2]), **points)
    print("Now aroundX 2/5")
    create_grid("aroundX", kx_range=Loffset(X[0]), ky_range=Loffset(X[1]), kz_range=Loffset(X[2]), **points)
    print("Now aroundU 3/5")
    create_grid("aroundU", kx_range=Loffset(U[0]), ky_range=Loffset(U[1]), kz_range=Loffset(U[2]), **points)
    print("Now aroundW 4/5")
    create_grid("aroundW", kx_range=Loffset(W[0]), ky_range=Loffset(W[1]), kz_range=Loffset(W[2]), **points)
    print("Now aroundGAMMA 5/5")
    create_grid("aroundGAMMA", kx_range=Loffset(GAMMA[0]), ky_range=Loffset(GAMMA[1]), kz_range=Loffset(GAMMA[2]), **points)


if __name__ == "__main__":
    """
    Basic setup
    """
    os.chdir("qefiles/")
    init_setup()

    """
    Conduction and valence, band minimum and maximum, respectively.
    """

    valence_max = valence_maximum()
    # print("valence max:", valence_max)
    # conduct_min = conduction_minimum()
    # print("conduction min:", conduct_min)
    # Plot only values within this range

    """
    Look at nodal lines, energy etc
    """

    # create_grids_around_points()

    plotrange = PlottingRange([-0, 1], [-0, 1], [-0, 1])  # (xlim, ylim, zlim)
    plotrange = PlottingRange.standard()
    
    # check_convergence()
    plot_3d_intersects("aroundGAMMA", emin=3, emax=valence_max+0.1, plotrange=plotrange, colors=False, epsilon=0.001)
    # plot_3d_energy("aroundL", 5, epsilon=1)
    # init_scf_calculation()
    # create_grid("aroundL", kx_num_points=12, ky_num_points=12, kz_num_points=12)
    # read_dat_file()


    plt.show()
