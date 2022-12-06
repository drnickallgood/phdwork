import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy.linalg as LA
from sklearn import metrics
from ember import read_vectorized_features
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import ember
import random


num_samples = 50 
num_clusters = 3 
rand_seed = 0 
random.seed(a=0)

centers_1725_5 = np.array([ [1, 6], [2, 4], [3, 5], [-1, -4], [8, -3] ])

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


tabu20_5_1725 = np.array([ [4, 10], [0, -8], [14, -6], [8, 12], [-2, 12] ])
tabu30_5_1725 = np.array([ [0, -8], [0, 10], [0, 12], [14, -6], [4, 8] ])
tabu35_5_1725 = np.array([ [14,-8], [0, 12], [8, 8], [0, -8], [4, 8] ])
tabu40_5_1725 = np.array([ [14, 0], [2, 10], [12, -4], [0, 8], [0, -8] ])
tabu45_5_1725 = np.array([ [0, -8], [0, -12], [12, -8], [0, 8], [-2, -10] ])
tabu50_5_1725 = np.array([ [12, -6], [0, -8], [12, 6], [-2, -8], [0, 8] ])
tabu100_5_1725 = np.array([ [8, 0], [0, -16], [14, -2], [4, 0], [0, 8] ])


sim20_5_1725 = np.array([ [-4, -8], [2, 8], [14,-6], [0, 12], [8, 12] ])
sim30_5_1725 = np.array([ [-2, 10], [6, -2], [10, -2], [-4, 8], [14, -2] ])
sim35_5_1725 = np.array([ [0, 12], [-2, -2], [10, -2], [-4, -2], [12, 12] ])
sim40_5_1725 = np.array([ [14, -2], [-2, -2], [6, 12], [-2, 10], [-2, 14] ])
sim45_5_1725 = np.array([ [14, -2], [10, -2], [12, -2], [-2, 14], [-2, -2] ])
sim50_5_1725 = np.array([ [14, -2], [-2, 14], [-2, -2], [-4, 14], [12, 6] ])


hybrid20_5_1725 = np.array([ [-6, 14], [-2, 2], [14, -8], [4, 10], [8, 10] ])
hybrid30_5_1725 = np.array([ [-2, -2], [6, 2], [14, -2], [-2, 14], [-2, -4] ])
hybrid35_5_1725 = np.array([ [14, -2], [8, -4], [-4, 14], [-2, -2], [-2, 6] ])
hybrid40_5_1725 = np.array([ [-2, 6], [-2, 14], [-2, -6], [-2, -2], [14, -2] ])
hybrid45_5_1725 = np.array([ [-2, 14], [-2, -2], [-10, -2], [14, -2], [-2, 6] ])
hybrid50_5_1725 = np.array([ [-2, -2], [14, -2], [-2, 14], [0, -2], [2, -2] ])
 


## Default K-Means ###

X, y = make_blobs(
    n_samples=num_samples, n_features=2, centers=centers_1725_5,
    cluster_std=1,shuffle=True, random_state=rand_seed
    )

## Motif Data Set ##

# Get EMBER vectors for MOTIF dataset
ember_X, ember_y = read_vectorized_features("/media/data1/malware/MOTIF/dataset/", subset="train")


# Normalize EMBER vectors and apply PCA
ember_norm_X = ember_X.copy()
ember_norm_X = make_pipeline(MinMaxScaler(), PCA(n_components=2)).fit_transform(ember_norm_X)

motif = np.zeros([num_samples, 2])
motif_y = np.zeros([num_samples,])

for i in range(0,num_samples):
    rand_sample = random.randint(0,num_samples)
    #print(ember_norm_X[rand_sample])
	# Get random sample for motif X
    motif[i] = ember_norm_X[i]
	# get random sample for motif y
    motif_y[i] = ember_y[i]


## Kmeans
default_km = KMeans(
    n_clusters=num_clusters, 
    n_init=10, max_iter=10000,
    tol=1e-04, init='random', random_state=0 
)

#default_y_km = default_km.fit_predict(X)
default_y_km = default_km.fit_predict(motif)


default_centers = default_km.cluster_centers_
default_iterations = default_km.n_iter_

## for motif we need a matching yin the correct dimension?

default_km_sscore = metrics.silhouette_score(motif, default_y_km, metric='euclidean')
default_km_homog = metrics.homogeneity_score(motif_y, default_y_km)
default_km_compl = metrics.completeness_score(motif_y, default_y_km)
default_km_vm = metrics.v_measure_score(motif_y, default_y_km)



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


