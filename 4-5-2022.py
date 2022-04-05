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
    centers=[ [2,2], [5,5] ], cluster_std=0.5,
    shuffle=True, random_state=0, return_centers=True
)

start_centers=np.array([(2,2),(5,5)])

Xnew, ynew = make_blobs(
    n_samples=10, n_features=2,
    centers=start_centers, cluster_std=0.5,
    shuffle=True, random_state=0
)

print(centers)
#plt.scatter(X[:,0],X[:,1], c=y);
#plt.show()
print(X)
print("\n----\n")
print(Xnew)
print("\n---\n")
print(y)
print("\n----\n")
print(ynew)


