import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

DPI = 100

iris = load_iris()
X = iris.data[:, :2]

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_

centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

colors = ['#b9b28b', '#8ba7b9', '#b98b99']
for i in range(len(X)):
    ax.scatter(X[i, 0], X[i, 1], c=colors[labels[i]], marker='o')

ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100)

ax.set_title('K-means Clustering for Iris (k=3)', fontsize=14)

plt.savefig("notes/_media/kmeans.png", dpi=DPI, bbox_inches='tight', pad_inches=0)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_title('')

plt.savefig("notes/_media/kmeans-cover.png", dpi=DPI, bbox_inches='tight', pad_inches=0)
