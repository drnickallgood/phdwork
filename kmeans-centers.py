import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy.linalg as LA


num_samples = 45 
num_clusters = 3
tabu_centers_20 = np.array([ [2,12], [6,8], [8,12] ])
tabu_centers_30 = np.array([ [4,12], [6,10], [4,8] ])
tabu_centers_35 = np.array([ [4,12], [4,8], [2,12] ])
tabu_centers_40 = np.array([ [4,12], [8,8], [2,6]  ])
tabu_centers_45 = np.array([ [0,12], [4,12], [4,8] ])
tabu_centers_50 = np.array([ [0,8], [4,8], [4,12] ])
tabu_centers_100 = np.array([ [12,8], [4,12], [-10,2] ])


sim_centers_20 = np.array([ [4,12], [6,8], [2, 10] ])
sim_centers_30 = np.array([ [4, 10], [6,10], [-4, 14] ])
sim_centers_35 = np.array([ [-2, 12], [2, 14], [6, 14]   ])
sim_centers_40 = np.array([ [6, 14], [-2, 14], [10, 10]    ])
sim_centers_45 = np.array([ [-2, 14], [-4, -2], [10, 14]   ])

hybrid_centers_20 = np.array([ [2, 10], [6, 10], [6, 14] ])
hybrid_centers_30 = np.array([ [10, 14], [10, 12], [-2, 14] ])
hybrid_centers_35 = np.array([ [10, 14], [-2, 14], [-4, -2] ])
hybrid_centers_40 = np.array([ [4, -2], [6, -2], [-2, -2] ])
hybrid_centers_45 = np.array([ [-2, -2], [-2, 14], [-2, -6] ])






### Default K-Means ###

X, y = make_blobs(
    n_samples=num_samples, n_features=2,
    cluster_std=1,shuffle=True, random_state=0
    )

default_km = KMeans(
    n_clusters=num_clusters, 
    n_init=10, max_iter=10000,
    tol=1e-04, init=hybrid_centers_45 
)

default_y_km = default_km.fit_predict(X)

default_centers = default_km.cluster_centers_
default_iterations = default_km.n_iter_

default_km_sscore = metrics.silhouette_score(X, default_y_km, metric='euclidean')
default_km_homog = metrics.homogeneity_score(y, default_y_km)
default_km_compl = metrics.completeness_score(y, default_y_km)
default_km_vm = metrics.v_measure_score(y, default_y_km)



print("Num Samples: ", num_samples)
print("Num Clusters: ", num_clusters)
print("Centroids: ", default_km.cluster_centers_)
print("Iterations: ", default_km.n_iter_)
#print("Inertia: ", LA.norm(v - np.matmul(W, H))**2)
print("Inertia: ", default_km.inertia_)
print("Silhouette Score: ", default_km_sscore)
print("Homogeneity Score: ", default_km_homog)
print("Completeness Score: ", default_km_compl)
print("V-measure: ", default_km_vm)


