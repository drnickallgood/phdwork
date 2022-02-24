import numpy as np
import copy
import dimod
import random
import math
from math import log
import dimod

# v = our V matrix which is p x n matrix
# w = our W matrix which is n x k
# h = our H matrix which is k x p
# k = # of clusters

def approx(v):

    return np.floor(v)

    
    
# Equation is good, but we only will have matrix to work with
# do not make equation and parse it as a string
#
'''


    V        W        H
[v0, v1]  [w0, w1]  [h0, h1]
[v2, v3]  [w2, w3]  [h2, h3]

(v0 - (w0 * h0) + (w1 * h2) )^2 +
(v1 - (w0 * h1) + (w1 * h3) )^2 +
(v2 - (w2 * h0) + (w3 * h2) )^2 + 
(v3 - (w2 * h1) + (w3 * h3) )^2


---

(2 - x2 - x1 + x0)^2

coefficents x0: 1, x1: -1, x2: -1

x0 =  1 * x0
x1 = -1 * x1
x2 = -1 * x2

a       b

[1]    [x0]
[-1]   [x1]
[-1]   [x2]

1 * x0 - 1 * x1 - 1 * x2


'''
def make_qubo(a):

    Q = {}
    varname = 'x'

    # Linear coefficients
    # still need to do squaring , subtracting, adding..etc
    # see other notebook, need to do those things..
    for i in range(a.size):
        Q[varname+str(i), varname+str(i)] = a[i][0]**2


    # Go through and do quadratic coefficients


    # Do linearization

    return Q

    

    
###

#V = np.array([[2.5,7.3],[3.5,2]]

a = np.array([[1], [-1], [-1]])

Q = make_qubo(a)

print(Q)




             
