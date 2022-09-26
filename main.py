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

start_time = datetime.now()
# In a 2x2 situation, we basicallay have to send one quadratic expression at a time
Q = {}
Q_alt = {}
index = {}
Q_total = {}



num_samples = 45
k = 3
centers = np.array([ [1,6], [2,4], [3,5] ])


# Test Data
V, blob_labels, blob_centers = make_blobs(
    n_samples=num_samples, n_features=2,centers=centers,
    cluster_std=1,center_box=(-8, 7),
    shuffle=True, random_state=0, return_centers=True
)

# Transpose matrix
v = np.transpose(V)

qubo_vars = qubo.parser.Parser(v,k)
v_dict, x_dict, x_dict_rev, p, n = qubo_vars.get_vars()
Q_total = {}

prec_list = [2, 1, 0]   #-8 to +7
#prec_list_str = ['null']
# Get string versions of prec_list_stirngs
#prec_strings = [prec_list_str.append(str(x)) for x in prec_list]


# Create Qubo Object
#myqubo = qubo.Qubo(v, k, num_samples, prec_list)

delta1 = 75
delta2 = 100

myqubo = qubo.Qubo(v, v_dict, x_dict, x_dict_rev, prec_list, k, p, n, delta1, delta2)

#Q_total = myqubo.get_qtotal()

#print(Q_total)

num_sweeps = 75000
num_reads = 65000
#tabu_timeout =   60000  # 1 min
#tabu_timeout = 300000  #ms  #5min
#tabu_timeout = 600000  #ms  #10min
#tabu_timeout = 900000  #ms  #15min
#tabu_timeout = 1200000  #ms  #20min
tabu_timeout = 1800000  #ms  #30min

#tabu_timeout = 3600000  #ms #1hr
#tabu_timeout = 7200000  #ms  #2hr
#tabu_timeout = 28800000       #8hr
#tabu_timeout = 57600000       #16hr

#solver = "tabu"
solver = "tabu"
myqubo.qubo_submit(num_sweeps, num_reads, tabu_timeout, solver)

#pprint.pprint(myqubo.get_solution_dict())


W, H = myqubo.get_w_h()

myqubo.get_results()
qcenters = myqubo.get_quantum_centers()

computed_labels = []

transposed_h = np.transpose(H)

for j in range(0, transposed_h.shape[0]):       # row
    for k in range(0, transposed_h.shape[1]):   # col
        #print(transposed_h[j][k])
        # k is right val for label
        if transposed_h[j][k] == 1:
            computed_labels.append(k)

computed_labels = np.array(computed_labels)

print("Blob label size", blob_labels.size)
print("Computed Label size", computed_labels.size)


# Set V back to the way it was format wise with transpose
s_score = metrics.silhouette_score(np.transpose(v), computed_labels, metric='euclidean')
#print("Silhouette score: ", s_score)
# Homogeneity
homogeneity_score = metrics.homogeneity_score(blob_labels, computed_labels)

# Completeness
completeness_score =  metrics.completeness_score(blob_labels, computed_labels)

# V-measure
vmeasure = metrics.v_measure_score(blob_labels, computed_labels)


print("Num Samples: ", num_samples)
print("Initial Centers: ", blob_centers)
#print("V: ", V)
#print("blob labels", blob_labels)
#print("W: ", W, "\n")
#print("H: ", H)

print("\n Norm: ", LA.norm(v - np.matmul(W, H)))
print("Inertia: ", LA.norm(v - np.matmul(W, H))**2)
print("Silhouette Score: ", s_score)
print("Homogeneity Score: ", homogeneity_score)
print("Completeness Score: ", completeness_score)
print("V-measure: ", vmeasure)
print("Frobenius Norm: ", LA.norm(v, 'fro')**2, "\n")

print("\nComputed Centers")
#print(qcenters)

for coords in qcenters:
    print(coords)


print("\nRunning time: ", datetime.now()-start_time, "\n")








