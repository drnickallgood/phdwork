import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


#[3,2,2,0]
#[1,-4,3,3]

'''
X, y, centers = make_blobs(
    n_samples=20, n_features=2,
    centers=[[3,2,2,0],[1,-4,3,3] ], cluster_std=0.05,
    shuffle=True, random_state=0, return_centers=True
)
'''

X, y, centers = make_blobs(
    n_samples=10, n_features=2,
    centers=[ [5,5], [3,2] ], cluster_std=0.5,
    shuffle=True, random_state=0, return_centers=True
)


plt.scatter(
    X[y == 0, 0], X[y == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y == 1, 0], X[y == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)


# plot the centroids
plt.scatter(
    centers[:, 0], centers[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)

plt.legend(scatterpoints=1)
#plt.grid()
#plt.show()

#print(centers)
#plt.scatter(X[:,0],X[:,1], c=y);
plt.grid()
plt.show()
#print(X)
#print("\n----\n")
#print(Xnew)
#print("\n---\n")
#print(y)
#print("\n----\n")
#print(ynew)


