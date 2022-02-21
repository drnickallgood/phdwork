import numpy as np
import copy
import dimod
import random
import math
from math import log
import dimod
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

sampler = dimod.ExactSolver()
# 5 unknowns, x0 - x5
#
# (2 - x0 x1 x2)^2
# 
#
'''
Expanded

4
-4 x0 x1 x2
+x0^2 x1^2 x2^2


'''

#linear coefficients

q1 = {}
q1['x0','x0'] = 1
q1['x1','x1'] = 1
q1['x2','x2'] = 1

q1['x0','x1'] = -4+1
q1['x1','x2'] = -4+1

#q1['x0','x1x2'] = -4+1    # result is {'x0': 1, 'x1': 0, 'x1x2': 1, 'x2': 0}
#q1['x0x1','x2'] = -4+1    #  Result is:  {'x0': 0, 'x0x1': 1, 'x1': 0, 'x2': 1}



sampleset1 = sampler.sample_qubo(q1)

print(sampleset1)

print("Result is: ", sampleset1.first.sample)




