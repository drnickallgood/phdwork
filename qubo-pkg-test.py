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

    

num_samples = 20
k = 3
centers = np.array([ [1,6], [2,4], [3,5] ])



V, y, blob_centers = make_blobs(
    n_samples=num_samples, n_features=2,
    cluster_std=0.5,center_box=(-8, 7),
    shuffle=True, random_state=0, return_centers=True
)



qubo_vars = qubo.parser.Parser(V,k)
v_dict, x_dict, x_dict_rev, p, n = qubo_vars.get_vars()

pprint.pprint(v_dict)
#qubo.Qubo(V, k, num_samples)





