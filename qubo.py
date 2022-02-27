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

# Annealer will return binary value, need to convert back to reals
# using approximation 
def approx_binstr(binstr):
    return 0
    
def make_qubo(v,k):
    
    Q = {}
    penalty_var_name = 'x'

    # Linear coefficients
    # still need to do squaring , subtracting, adding..etc
    # see other notebook, need to do those things..
    # We need to match existing coefficients..
    # i.e x0,x1 = x1,x0
    # V = p x n
    # W = n x k
    # H = k x n
    # k = # of clusters
    
    p = v.shape[0]
    n = v.shape[1]

    # Init W and H matrices
    w = np.zeros([n,k])
    h = np.zeros([k,n])

    # initialize w and h to all 1's
    # This will allow us to do numerical stuff on it
    for i in range(0,len(w)):
        for j in range(0,len(w)):
            w[i,j] = 1

    for i in range(0,len(w)):
        for j in range(0,len(w)):
            w[i,j] = 1
    

    # (v0 - (w0 * h0) + (w1 * h2) )^2 +
    for i in range(0,len(a)):
        for j in range(0,len(a)):
            Q[i,j] = v[i,j]
            # Do we have this already?, if so add it to existing
           # if (j,i) in Q.keys():
               # Q[j,i] += a[i][j]
            #else:
                #Q[i,j] = a[i][j] 
        
    #for i in range(a.size):
     #   Q[varname+str(i), varname+str(i)] = a[i][0]**2


    # Go through and do quadratic coefficients
    # In the event we have something like w11h11, we apply penalties
    # and we map them 

    # Do linearization, peanlty stuf
    #Penalty coeff for w11h11 substitution to x1 : 2(w11h11 - 2x1(w11 + h11) + 3x1)
    '''
      x0 = w11h11
      x1 = w12h21
      x2 = w11h12
      x3 = w12h22
      
      x4 = w21h11
      x5 = w22h21
      x6 = w21h12
      x7 = w22h22


    '''

    return Q

    

    
###

#V = np.array([[2.5,7.3],[3.5,2]]

a = np.array([[1,2], [3,4]])

Q = make_qubo(a, 2)

print(Q)




             
