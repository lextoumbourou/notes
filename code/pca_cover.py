from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

iris = datasets.load_iris()

X = iris.data
y = iris.target

# Note that the input is not scaled here, which is consistent with the sklearn implementation.
X_centered = (X - np.mean(X, axis=0))

cov_matrix = np.cov(X_centered.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:,sorted_indices]

top_eigenvectors = sorted_eigenvectors[:,:2]

X_pca = X_centered @ top_eigenvectors

explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
print("Explained Variance Ratio:", explained_variance_ratio)

plt.figure(figsize=(8, 6))
for i, species in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=species)

plt.axis('off')
plt.savefig('notes/_media/pca-cover.png', bbox_inches='tight', pad_inches=0)
