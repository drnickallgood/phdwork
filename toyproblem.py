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



# Convert ||V-WH|| problem to QUBO
# v = Input Matrix of our samples as np.array 
# w = Input matrix that contains Clustered Samples, each column vector is a cluster
# h = Binary vector, such that every every "1" value is the centroid
# n = length of column vector x
# bitspower - dict containg powers of 2, applied in 2's compliment-like
# Output - QDict a dictionary containing the QUBO formulation
# dict will be (u,v) : b where (u,v) are binary variables and b is bias

### W Candidate centers, ###
## Each column is a cluster, every two rows = x and y coordinates
#####
#### SAMPLES ###

'''
Example pcode for paper

Given V
Find W, H

Radix-2
----


QUBOPrep 
---



---- 

main
-----

Q := quboPrep(W,H,bitsPower)

S := QuantumQUBOSolver(Q)

S := S.first_sample

S := PostProcess(S)




'''

'''

--- Sampple V Matrix ---

p x n == 10 x 2

[[ 1.04829186  5.03092408]
 [ 1.35678894  4.36462484]
 [ 2.21180137  0.47061579]
 [ 0.92466065  4.50908658]
 [-1.09468591  2.54679975]
 [-0.3920267   2.19069942]
 [-2.80339892  3.24469156]
 [ 2.80230706  0.79508453]
 [ 2.27719914  1.06450082]
 [ 1.45131429  4.22810872]]



--- Sample W Matrix ---

n x k = 2 x 2

-1.3049724,   0.92466, 
 3.08471943,  4.50908, 

w11 w12 
w21 w22 


--- Sample H Matrix ---

k x p = 2 x 10

** BINARY MATRIX ***
** Only a single '1' per column
** Each '1' is the 'vote' on the best center

1 0 0 1 0 0 1 0 0 1 
0 1 0 0 1 0 0 1 1 0 


h00 h01 h02 h03 h04 h05 h06 h07 h08 h09
h10 h11 h12 h13 h14 h15 h16 h17 h18 h19



'''


def make_qubo(v,w,h,n,bitspower):

    powersTwo = np.zeros(len(bitspower)+1)

    # Deal with powers of two
    for i in range(0, len(bitspower)+1):
        if i == 0:
            powersTwo[i] = (2**(bitspower[i]+1)) * (-1)
        else:
            powersTwo[i] = 2**(bitspower[i-1])

    Qinit = np.zeros([n,n])
    Qdict = {}

    # W is our unknowns to be found by annealing
    # Each entry in W has our radix-2 technique applied
    # Each entry in W = [-8w 4w 2w 1w ]
    # the QUBO form is -8w + 4w + 2w + 1w
    
                    


    


# Convert dict to binary string



#Put binary string into np.array vector





# scikit learn stuff.. prep

X, y = make_blobs(
    n_samples=10, n_features=2,
    centers=2, cluster_std=0.5,
    shuffle=True, random_state=0
)



