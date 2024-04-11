import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DPI = 100

a, b, c, d = 0, 0, 1, 0

x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
x, y = np.meshgrid(x, y)

z = (-d - a * x - b * y) / c

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    x, y, z, alpha=0.6, rstride=100, cstride=100, color="#b98b99", edgecolor="none"
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
ax.set_title("")

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

plt.savefig(
    "notes/_media/hyperplane-cover.png", dpi=DPI, bbox_inches="tight", pad_inches=0
)
