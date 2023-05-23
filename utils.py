import re
import numpy as np
from scipy.spatial import Voronoi


FORMATTING_DECIMALS = 4


class PlottingRange():
    """
    Class used to define ranges wherein values should be plotted
    """
    @staticmethod
    def standard():
        """
        A standard instance of this class with high values
        in the limits such that all values are plotted

        Return
        ------
        PlottingRange : Returns standard instance of
                        this class that allows for all values
        """
        return PlottingRange([-100, 100], [-100, 100], [-100, 100])

    def __init__(self, xlim: list, ylim: list, zlim: list):
        """
        Parameters
        ----------
        xlim : list
            Limit for x-values
        ylim : list
            Limit for y-values
        zlim : list
            Limit for z-values
        """
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

    @staticmethod
    def _within(val: float, lim: list) -> bool:
        """
        Check if value within limit

        Parameters
        ----------
        val : float
            Check if this value is within the limit "lim"
        lim : list
            List to define allowed upper and lower values

        Return
        ------
        bool : Returns whether or not point is within "lim"
        """
        return (val >= lim[0]) and (val <= lim[1])

    def check_within(self, point: tuple) -> bool:
        """
        Check if point is within limits

        Parameters
        ----------
        point : tuple
            Tuple containing x, y, z values to check if it is within limits

        Return
        ------
        bool : Returns whether or not point is within limit
        """
        cond1 = self._within(point[0], self.xlim)
        cond2 = self._within(point[1], self.ylim)
        cond3 = self._within(point[2], self.zlim)
        return cond1 and cond2 and cond3


def inputfile_row_format(row, row_num) -> str:
    """
    Format row list to a string, the same way Quantum Espresso does it.

    Parameters
    ----------
    row : list
        point to be formatted

    row_num : int
        The row-number this row corresponds to
    """
    FD = FORMATTING_DECIMALS
    return f"\t{row[0]:.{FD}f} {row[1]:.{FD}f} {row[2]:.{FD}f} {row_num: >3}\n"


def fcc_points() -> list:
    # Code taken from
    # http://staff.ustc.edu.cn/~zqj/posts/howto-plot-brillouin-zone/
    cell = np.array([[0.0, 1, 1],
                    [1, 0.0, 1],
                    [1, 1, 0.0]])

    cell = np.asarray(cell, dtype=float)
    assert cell.shape == (3, 3)

    px, py, pz = np.tensordot(cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    vor = Voronoi(points)

    bz_facets = []
    bz_ridges = []
    bz_vertices = []

    # for rid in vor.ridge_vertices:
    #     if( np.all(np.array(rid) >= 0) ):
    #         bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
    #         bz_facets.append(vor.vertices[rid])

    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):
        # WHY 13 ????
        # The Voronoi ridges/facets are perpendicular to the lines drawn
        # between the input points. The 14th input point is [0, 0, 0].
        if (pid[0] == 13 or pid[1] == 13):
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid

    bz_vertices = list(set(bz_vertices))

    return bz_ridges


def get_values(text) -> list:
    """
    Given a text return a list of values contained within
    Parameters
    ----------
    text : str
        Str fromwhich values are found
    """
    reg_str = r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    text_values = re.findall(reg_str, text)
    values = list(map(float, text_values))
    return values


def file_change_line(filename, numline, newline):
    """
    Change a line in a given file

    Parameters
    ----------
    filename : str
        Name of file for which a line will be changed
    numline : int
        Line number
    newline : str
        Content of the new line
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        lines[numline] = newline

    with open(filename, "w") as f:
        f.writelines(lines)


def list_to_formatted_string(L):
    newL = list(map(lambda x: " ".join(x), L))
    print(newL)
    finishL = "\n".join(newL)
    print(finishL)


if __name__ == "__main__":
    print(list_to_formatted_string([[1,2,3],[4,5,6]]))
