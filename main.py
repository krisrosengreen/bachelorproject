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
    print("valence max:", valence_max)
    conduct_min = conduction_minimum()
    print("conduction min:", conduct_min)
    # Plot only values within this range

    """
    Look at nodal lines, energy etc
    """

    plotrange = PlottingRange([-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1])  # (xlim, ylim, zlim)

    # check_convergence()
    plot_3d_intersects(emin=4, emax=valence_max+0.1, plotrange=plotrange, colors=False, epsilon=0.01)
    # plot_3d_energy(5.75)
    # init_scf_calculation()
    # create_grid()
    # read_dat_file()


    plt.show()
