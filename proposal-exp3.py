import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


## center points
## must be of size n_clusters, n_features
## n_features = 2
## the problem is, we correspond centroids to n_clusters
## so we need to figure out the optimal number of clusters
## Results will be basically +1, -1, etc (1, 0)
## maybe we can map the weights as the coefficients
## count how many have +1 set, THIS is n_clusters
## These will also be the ideal center points, collect them
## run kmeans/kmedoids with these points

# predfined centers, 3 x 2 nparray


#Blob centers
centers = np.array([ [1,6], [2,4], [3,5] ])

tabu_centers = np.array([ [1,6], [2,4], [3,5] ])
simanneal_centers = np.array([ [3,5], [2,4], [1,7] ])
hybrid_centers = np.array([ [3,7], [2,6], [5,4] ])

X, y = make_blobs(
    n_samples=20, n_features=2,
    centers=centers, cluster_std=0.5,
    shuffle=True, random_state=0)

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=10000,
    tol=1e-04, random_state=0
)

km_tabu = KMeans(
        n_clusters=3, init=tabu_centers,
        n_init=10, max_iter=10000,
        tol=1e-04, random_state=0
)

km_simanneal = KMeans(
        n_clusters=3, init=simanneal_centers,
        n_init=10, max_iter=10000,
        tol=1e-04, random_state=0
)

km_hybrid = KMeans(
        n_clusters=3, init=hybrid_centers,
        n_init=10, max_iter=10000,
        tol=1e-04, random_state=0
)


y_km = km.fit_predict(X)
y_km_tabu = km_tabu.fit_predict(X)
y_km_simanneal = km_simanneal.fit_predict(X)
y_km_hybrid = km_hybrid.fit_predict(X)

#plot
# plot the 3 clusters
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)

print("\n--- K-means (Random) ---\n")
print("K-Means (Random) Centers: ", km.cluster_centers_)
print("K-Means (Random) Iterations: ", km.n_iter_)
print("K-Means (Random) Inertia: ", km.inertia_)

print("\n--- K-means (Tabu) ---\n")
print("K-Means (Tabu) Centers: ", km_tabu.cluster_centers_)
print("K-Means (Tabu) Iterations: ", km_tabu.n_iter_)
print("K-Means (Tabu) Inertia: ", km_tabu.inertia_)

print("\n--- K-means (Sim. Annealing) ---\n")
print("K-Means (Sim. Annealing) Centers: ", km_simanneal.cluster_centers_)
print("K-Means (Sim. Annealing) Iterations: ", km_simanneal.n_iter_)
print("K-Means (Sim. Annealing) Inertia: ", km_simanneal.inertia_)

print("\n--- K-means (Hybrid BQM) ---\n")
print("K-Means (Hybrid BQM) Centers: ", km_hybrid.cluster_centers_)
print("K-Means (Hybrid BQM) Iterations: ", km_hybrid.n_iter_)
print("K-Means (Hybrid BQM) Inertia: ", km_hybrid.inertia_)
print()












#plt.legend(scatterpoints=1)
#plt.grid()
#plt.show()



'''
3 Centers for this file

[[-1.53142893  3.15495352]
 [ 1.75789654  0.81392937]
 [ 1.15053223  4.31268924]]
'''

