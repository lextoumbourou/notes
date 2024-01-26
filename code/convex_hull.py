import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

DPI = 100

np.random.seed(0)
points = np.random.rand(30, 2)

hull = ConvexHull(points)

plt.figure(figsize=(4,4))
plt.plot(points[:,0], points[:,1], 'o', color="#333")
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], color="#b9b28b")

plt.xticks([])
plt.yticks([])

plt.box(False)

file_path = "notes/_media/convex-hull.png"
plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
plt.close()
