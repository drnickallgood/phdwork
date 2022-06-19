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



start_time = datetime.now()
# In a 2x2 situation, we basicallay have to send one quadratic expression at a time
Q = {}
Q_alt = {}
index = {}
Q_total = {}

    

num_samples = 100
k = 3
centers = np.array([ [1,6], [2,4], [3,5] ])


# Test Data
V, y, blob_centers = make_blobs(
    n_samples=num_samples, n_features=2,
    cluster_std=0.5,center_box=(-8, 7),
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
delta1 = 10
delta2 = 50

myqubo = qubo.Qubo(v, v_dict, x_dict, x_dict_rev, prec_list, k, p, n, delta1, delta2)

#Q_total = myqubo.get_qtotal()

#print(Q_total)

num_sweeps = 10000
num_reads = 2000
tabu_timeout = 300000   #ms
solver = "sim"˜
myqubo.qubo_submit(num_sweeps, num_reads, tabu_timeout, solver)

#pprint.pprint(myqubo.get_solution_dict())


W, H = myqubo.get_w_h()

#print(W)
#print(H)

myqubo.get_results()

print("\nInitial Centers: ", blob_centers)
print("\nRunning time: ", datetime.now()-start_time, "\n")






