import qubo
import numpy as np
import copy
import dimod
import random
import math
from math import log
import pprint
import neal
import numpy.linalg as LA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from datetime import datetime
import tabu
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from sklearn import metrics

num_samples = 20
k = 3

# Test Data
v, blob_labels, blob_centers = make_blobs(
    n_samples=num_samples, n_features=2,centers=k,
    cluster_std=0.5,center_box=(-8, 7),
    shuffle=True, random_state=7, return_centers=True
)

v = np.transpose(v)

'''
W:  [[-1  7. -7.]
    [ 3.  0  3.]]

H:  [0 0 0 0 1 0 1 0 0 0 1 1 0 1 0 1 1 0 0 0]
    [1 0 0 1 0 1 0 0 0 1 0 0 1 0 0 0 0 0 1 0]
    [0 1 1 0 0 0 0 1 1 0 0 0 0 0 1 0 0 1 0 1]]
'''

w = np.array([[-1, 7, -7], [3, 0, 3]])

h = np.array([[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
[1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]])

### 

computed_centers = np.transpose(w)

computed_labels = []

transposed_h = np.transpose(h)
#print(transposed_h)
#for i in range(0,computed_labels.size):
for j in range(0, transposed_h.shape[0]):       # row
    for k in range(0, transposed_h.shape[1]):   # col
        #print(transposed_h[j][k])
        # k is right val for label
        if transposed_h[j][k] == 1:
            #print("Label for: ", transposed_h[j][k] , "is: ", str(k))
            computed_labels.append(k)


#print(type(computed_labels))
#print(computed_labels)
computed_labels = np.array(computed_labels)
#print(type(computed_labels))
#print(computed_labels)
#print(computed_labels.shape)
#print(computed_labels.size)

#print(np.transpose(v))

## Inertia ##
# Sum of squared errors, uses L2 Norm
inertia = LA.norm(v - np.matmul(w, h)) ** 2
print("Inertia: ", inertia)

## Silohouette ##

# (b - a) / max(a, b)
	#a: The mean distance between a sample and all other points in the same class.
	#b: The mean distance between a sample and all other points in the next nearest cluster.
    # For a set of samples Silohouette score is mean of each coefficient scores
    	# Calculate scores
        # Sum Scores
        # divide by # of samples

    # sklearn - metrics.silhouette_score(X, labels, metric='euclidean')

# V is our samples, H is our labels
# H needs to be 1D

# Need to convert H to labels 1D array, 3 clustsers = 0, 1, 2
# Transpose H, iterate through rows of H, where there is a 1, that position is a label


s_score = metrics.silhouette_score(np.transpose(v), computed_labels, metric='euclidean')
print("Silhouette score: ", s_score)


# Homogeneity

homogeneity_score = metrics.homogeneity_score(blob_labels, computed_labels)  

print("Homogeneity Score: ", homogeneity_score)

# Completeness

completeness_score =  metrics.completeness_score(blob_labels, computed_labels) 
print("Completeness Score: ", completeness_score)


# V-measure

vmeasure = metrics.v_measure_score(blob_labels, computed_labels)    
print("V-measure: ", vmeasure)

















# Do kmeans first with centers





# 


