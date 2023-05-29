import re
import numpy as np
import matplotlib.pyplot as plt


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
        cond4 = check_within_BZ(point)
        return cond1 and cond2 and cond3 and cond4


def check_within_BZ(point_check) -> bool:
    """
    Check if a given points lies within the first Brillouin zone

    Parameters
    ----------
    point_check : list
        List containing the coordinates for the point to be checked

    Returns
    -------
    bool : Whether point lies 
    """
    points = np.array([[2, 0, 0], [-2, 0, 0],
                       [0, 2, 0], [0, -2, 0],
                       [0, 0, 2], [0, 0, -2],
                       [1, 1, 1], [-1, 1, -1],
                       [1, 1, -1], [-1, 1, 1],
                       [1, -1, 1], [-1, -1, -1],
                       [1, -1, -1], [-1, -1, 1]])
    point_check = np.array(point_check)
    dist_center = np.sqrt(point_check.dot(point_check))

    within = True

    for point in points:
        diff = point_check - point 
        dist_point = np.sqrt(diff.dot(diff))
        if dist_point < dist_center:
            within = False
    return within


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


def connect_lines(points):
    lines = []

    for p1 in points:
        for p2 in points:
            dist = p1-p2
            if np.sqrt(dist.dot(dist)) <= 0.9:
                lines.append(np.array([p1, p2]))
    return lines


def rotate_points(points, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle), 0],
                        [np.sin(angle), -np.cos(angle), 0],
                        [0, 0, 1]])
    return points@rot_mat


def plot_lines(ax, lines):
    for line in lines:
        # Et lille hack til at tilføje dybde
        camera = np.array([1, -1, 1])
        p1 = (line[1] + line[0]) / 2
        diff = camera - p1
        dist = np.sqrt(diff.dot(diff))
        alpha = 1 - min(1, (dist-1)/1.69)

        ax.plot3D(line[:, 0], line[:, 1], line[:, 2], c='k', lw=0.5, alpha=alpha)


def plot_first_quad_fcc(ax):
    points = [[0.5, 1, 0], [0, 1, 0.5],  # X-W s
              [1, 0.5, 0], [1, 0, 0.5],
              [0.5, 0, 1], [0, 0.5, 1]]
    points = np.array(points)
    lines = connect_lines(points)
    plot_lines(ax, lines)


def limit_first_quad(ax):
    ax.axes.set_xlim3d(0, 1)
    ax.axes.set_ylim3d(0, 1)
    ax.axes.set_zlim3d(0, 1)


def plot_fcc(ax):
    points = [[0.5, 1, 0], [0, 1, 0.5],  # X-W s
              [1, 0.5, 0], [1, 0, 0.5],
              [0.5, 0, 1], [0, 0.5, 1]]
    points = np.array(points)
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        rot_points = rotate_points(points, angle)
        lines = connect_lines(rot_points)
        plot_lines(ax, lines)

    inverted = np.array([1, 1, -1]) * points
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        rot_points = rotate_points(inverted, angle)
        lines = connect_lines(rot_points)
        plot_lines(ax, lines)

    plt.axis("equal")
