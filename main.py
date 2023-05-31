from energy import *
from matplotlib import rcParamsDefault
import matplotlib.pyplot as plt
from utils import limit_first_quad, plot_fcc
import os


plt.rcParams["figure.dpi"]=150
plt.rcParams["figure.facecolor"]="white"
plt.rcParams["figure.figsize"]=(8, 6)


def init_setup():
    required_folders = ["images", "datfiles", "gnufiles", "config"]
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

    num_points=61
    offset = 0.1
    Loffset = lambda x: [x-offset, x+offset]

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


def create_L_grid_Ry30():
    L = symmetry_points.L
    num_points=4
    offset = 0.1
    Loffset = lambda x: [x-offset, x+offset]
    points = {"kx_num_points": num_points, "ky_num_points": num_points, "kz_num_points": num_points}
    create_quad_BZ_grid("ry30_Lgrid", kx_range=Loffset(L[0]), ky_range=Loffset(L[1]), kz_range=Loffset(L[2]), **points)



if __name__ == "__main__":
    """
    Basic setup
    """
    os.chdir("qefiles/")
    init_setup()

    """
    Conduction and valence, band minimum and maximum, respectively.
    """

    # valence_max = valence_maximum()
    # print("valence max:", valence_max)
    # conduct_min = conduction_minimum()
    # print("conduction min:", conduct_min)
    # Plot only values within this range

    """
    Look at nodal lines, energy etc
    """

    Lrange = [0, 1]

    # create_L_grid_Ry30()
    # create_grid("test_aroundL", kx_num_points=101, ky_num_points=101, kz_num_points=101)
    # create_quad_BZ_grid("test_aroundL", kx_num_points=101, ky_num_points=101, kz_num_points=101)
    # print("Creating test2 grid")
    # create_quad_BZ_grid("grid150", kx_num_points=150, ky_num_points=150,
                        # kz_num_points=150, kx_range=Lrange, ky_range=Lrange, kz_range=Lrange)

    plotrange = PlottingRange([-0, 1], [-0, 1], [-0, 1])  # (xlim, ylim, zlim)
    plotrange = PlottingRange.standard()
    # init_scf_calculation()
    # check_convergence()
    # plot_3d_intersects("test5", emin=-12, emax=VALENCE_MAX+0.1, plotrange=plotrange, colors=False, epsilon=0.0001)
    # print("Plotting now!")
    (fig,ax) = plot_3d_intersects("grid150", emin=-12, emax=VALENCE_MAX+0.1, plotrange=plotrange, colors=False, epsilon=0.001, include_conduction=False)
    # plot_3d_energy("aroundL", 5, epsilon=1)
    # create_grid("aroundL", kx_num_points=12, ky_num_points=12, kz_num_points=12)
    # read_dat_file()

    plot_fcc(ax)

    from utils import remove_ticks_and_grid
    remove_ticks_and_grid(ax)
    limit_first_quad(ax)

    plt.show()
