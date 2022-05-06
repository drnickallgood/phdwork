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


centers = np.array([ [1,6], [2,4], [3,5] ])
centers2 = np.array([ [3,5], [2,4], [1,7] ])
centers3 = np.array([ [3,7], [2,6], [5,4] ])

X, y = make_blobs(
    n_samples=20, n_features=2,
    cluster_std=0.5,
    shuffle=True, random_state=0)

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=10000,
    tol=1e-04, random_state=0
)

y_km = km.fit_predict(X)

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

print(km.cluster_centers_)
print(km.inertia_)
#plt.legend(scatterpoints=1)
#plt.grid()
#plt.show()



'''
3 Centers for this file

[[-1.53142893  3.15495352]
 [ 1.75789654  0.81392937]
 [ 1.15053223  4.31268924]]
'''

