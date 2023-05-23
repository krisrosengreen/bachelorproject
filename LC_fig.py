from utils import get_values
import matplotlib.pyplot as plt
import numpy as np

with open("output/optimization_LC.txt") as f:
    lines = f.readlines()

points = []

for line in lines[1:][::2]:
    vals = get_values(line)
    points.append(vals)

nppoints = np.array(points)
plt.scatter(nppoints[:, 0], nppoints[:, 1], s=2)
plt.ylabel("Total Energy [Ry]")
plt.xlabel("Lattice Constant")
plt.savefig("qefiles/figures/lattice_constant.pdf")
