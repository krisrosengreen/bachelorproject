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
    init_scf_calculation()
    aroundL = [symmetry_points.L[0]-0.05, symmetry_points.L[1]+0.05]
    aroundX = [symmetry_points.X[0]-0.05, symmetry_points.X[1]+0.05]
    aroundW = [symmetry_points.W[0]-0.05, symmetry_points.W[1]+0.05]
    aroundU = [symmetry_points.U[0]-0.05, symmetry_points.U[1]+0.05]
    aroundGAMMA = [symmetry_points.gamma[0]-0.05, symmetry_points.gamma[1]+0.05]

    num_points=24

    print("Now aroundL 1/5")
    create_grid("aroundL", kx_num_points=num_points, ky_num_points=num_points, kz_num_points=num_points, kx_range=aroundL, ky_range=aroundL, kz_range=aroundL)
    print("Now aroundL 2/5")
    create_grid("aroundX", kx_num_points=num_points, ky_num_points=num_points, kz_num_points=num_points, kx_range=aroundX, ky_range=aroundX, kz_range=aroundX)
    print("Now aroundL 3/5")
    create_grid("aroundW", kx_num_points=num_points, ky_num_points=num_points, kz_num_points=num_points, kx_range=aroundW, ky_range=aroundW, kz_range=aroundW)
    print("Now aroundL 4/5")
    create_grid("aroundU", kx_num_points=num_points, ky_num_points=num_points, kz_num_points=num_points, kx_range=aroundU, ky_range=aroundU, kz_range=aroundU)
    print("Now aroundL 5/5")
    create_grid("aroundGAMMA", kx_num_points=num_points, ky_num_points=num_points, kz_num_points=num_points, kx_range=aroundGAMMA, ky_range=aroundGAMMA, kz_range=aroundGAMMA)


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

    create_grids_around_points()

    plotrange = PlottingRange([-0, 1], [-0, 1], [-0, 1])  # (xlim, ylim, zlim)

    # check_convergence()
    # plot_3d_intersects("aroundL", emin=-15, emax=15, plotrange=plotrange, colors=False, epsilon=0.001)
    # plot_3d_energy("aroundL", 5, epsilon=1)
    # init_scf_calculation()
    # create_grid("aroundL", kx_num_points=12, ky_num_points=12, kz_num_points=12)
    # read_dat_file()


    # plt.show()
