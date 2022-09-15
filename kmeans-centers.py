import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


num_samples = 100
num_clusters = 3

### Default K-Means ###

X, y = make_blobs(
    n_samples=num_samples, n_features=2,
    cluster_std=1,
    shuffle=True, random_state=0)

default_km = KMeans(
    n_clusters=num_clusters, init='random',
    n_init=10, max_iter=10000,
    tol=1e-04, random_state=0
)

default_y_km = default_km.fit_predict(X)

default_centers = default_km.cluster_centers_
default_iterations = default_km.n_iter_


### Tabu K-Means

tabu_centers_20 = np.array([ [2,12], [6,8], [8,12] ])

tabu_centers_50 = np.array([ [0,8], [4,8], [4,12] ])

tabu_centers_100 = np.array([ [12,8], [4,12], [-10,2] ])

tabukm = KMeans(
    n_clusters=num_clusters, init=tabu_centers_20,
    max_iter=10000, n_init=10,
    tol=1e-04
)

tabukm_fit = tabukm.fit_predict(X)

tabu_centers = tabukm.cluster_centers_
tabu_iter = tabukm.n_iter_

print("Num Samples: ", num_samples)
print("Num Clusters: ", num_clusters)
print("Centroids: ", tabu_centers)
print("Iterations: ", tabu_iter)



