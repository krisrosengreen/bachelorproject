from utils import (rotate_points, connect_lines)
import numpy as np
from math import floor, ceil
import matplotlib.pyplot as plt


def get_all_lines():
    points = [[0.5, 1, 0], [0, 1, 0.5],  # X-W s
              [1, 0.5, 0], [1, 0, 0.5],
              [0.5, 0, 1], [0, 0.5, 1]]
    all_lines = []
    points = np.array(points)
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        rot_points = rotate_points(points, angle)
        lines = connect_lines(rot_points)
        for _ in lines:
            all_lines.append(_)

    inverted = np.array([1, 1, -1]) * points
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        rot_points = rotate_points(inverted, angle)
        lines = connect_lines(rot_points)
        for _ in lines:
            all_lines.append(_)

    return all_lines


def round_nearest(x):
    f = floor(x*2)
    c = ceil(x*2)

    fdiff = abs(f-x)
    cdiff = abs(x-c)

    if fdiff < cdiff:
        return f/2
    else:
        return c/2



def formate_coordinate(L):
    return f"({round_nearest(L[0])},{round_nearest(L[1])},{round_nearest(L[2])})"


if __name__ == "__main__":
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    lines = get_all_lines()
    with open("tikz.tex", "w") as f:
        f.write("\\usetikzlibrary{3d}\n\\begin{tikzpicture}\n")
        for line in lines:
            coord1 = formate_coordinate(line[0])
            coord2 = formate_coordinate(line[1])
            txt_line = f"\\draw {coord1} -- {coord2};\n"
            print(line, coord1, coord2)
            print()
            print()
            f.write(txt_line)

            ax.plot3D(line[:,0], line[:,1], line[:,2], c="k")
        f.write("\\end{tikzpicture}\n")
    
    plt.show()
