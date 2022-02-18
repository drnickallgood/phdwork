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


multiplication == dot product

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

start_centers=np.array([(2,2),(5,5)])

X, y = make_blobs(
    n_samples=10, n_features=2,
    centers=start_centers, cluster_std=0.5,
    shuffle=True, random_state=0
)

'''
# Plotting

plt.scatter(X[:,0], X[:,1], s=50,c='blue', marker='o', edgecolor='black')

#Plot centroids
plt.scatter(centers[0,:], centers[1,:], s=250, marker='*', c='red', edgecolor='black', label='centroids')

plt.legend(scatterpoints=1)
plt.grid()
plt.show()
'''

# -- Testing area -- #
## Ax - b ##

## In this first example know the solution for x1 and x2, which is x1=0, x1=1

A = np.array([ [2,1], [1,3] ])
b = np.array([1,3])

# Qubi dictionary, qn means qubibi #
Q = {}

# Individual coefficients here
# Based on the L2 Norm, where , we square the xample above
## ||Ax - b||^2
#
# 2 * (2 - (2*1) ) + (1 * (1 - (2*3)))

Q['q0','q0'] = A[0,0] * (A[0,0] - 2*b[0]) + A[1,0] * (A[1,0] - 2*b[1])
Q['q1','q1'] = A[0,1] * (A[0,1] - 2*b[0]) + A[1,1] * (A[1,1] - 2*b[1])

# Interation between qubits 0 and 1 as QUBO
Q['q0','q1'] = 2 * (A[0,0] * A[0,1] + A[1,0] * A[1,1])

print("-- Our current Qubo setup ---")
print(Q)

# Let us try the excat solver

sampler = dimod.ExactSolver()

sampleset = sampler.sample_qubo(Q)

# Get best energy

print("Result is: ", sampleset.first.sample)

### Second example ###

## First was easy, now we need to focus on quadratics

# Find x0, x1, x2 such that we minimize (2 - x2 - x1 -x0)^2
# We know going in, the best solution will have 2 variables = 1

# (2 - x2 - x1 - x0)^2 =
# x0^2 + x1^2 + x2^2 + 2(-x0)(-x1) + 2(-x1)(-x2) + 2(-x0)(-x2) + 2(2)(-x0) + 2(2)(-x1) + 2(2)(-x2) + 4

# x0^2 = x0, x1^2 = x1, x2^2 = x2 .. so this is the same as

# 1(1-2(2)x0 - 1(1-2(2))x1 - 1(1-2(2))x2 + 2x0x1 + 2x1x2 + 2x0x2 + 4

## Lets make this into a QUBO

Q2 = {}

Q2['x0','x0'] = 1 * (1-2*(2))
Q2['x1','x1'] = 1 * (1-2*(2))
Q2['x2','x2'] = 1 * (1-2*(2))

# Our quadratic coefficients  / interactions
Q2['x0','x1'] = 2
Q2['x0','x2'] = 2
Q2['x1','x2'] = 2

print(Q2)

# let us solve with exactsolver

sampleset2 = sampler.sample_qubo(Q2)

print(sampleset2)

print("Result is: ", sampleset2.first.sample)

for k,v in Q.items():
    print(k,v)


print("\n\n ----- Binary vector ----\n")

# now we have a binary vector with 3 unknowns, x0, x1, and x2
# Let us minimize (3 + 4x2 - 2x1 - x0)^2
# We can tell our answer by looking, x0 = 1, x1 = 1 , x2 = 0
## But convert this to QUBO

# (3 + 4x2 - 2x1 - x0)^2 =
# (4x2)^2 + 2x1^2 + x0^2 + 2(4x2(-2x1) + 2(-2x1)(-x0) + 2(4x2)(-x0) + 2(3)(4x2) + 2(3)(-2x1) + 2(3)(-x0) + 9
# Qubo we can ignore the consant, which is 9
#
Q3 = {}
Q3['x0','x0'] = 1*(1-2*(3))
Q3['x1','x1'] = 2*(2-2*(3))
Q3['x2','x2'] = 4*(4+2*(3))

Q3['x0','x1'] = 2 * (2)
Q3['x0','x2'] = -2 * (4)
Q3['x1','x2'] = -2 * (8)

for k,v in Q3.items():
    print(k,v)

sampleset3 = sampler.sample_qubo(Q3)
print(sampleset3)
print(sampleset3.first.sample)

## Ok now let us try this for V=WH
## W and H are unknowns ##

'''

  V          W                H
[1,2]    [w11 w12]      [h11, h12]
[3,4]    [w21 w22]      [h21, h22]


          WH
[w1h1 + w2h3  w1h2 + w2h4 ]
[w3h1 + w4h3  w3h2 + w4h4 ]

( (w1h1 + w2h3) + (w1h2 + w2h4) + (w3h1 + w4h3) + (w3h2 + w4h4) )^2 = 

# thanks Mathematica Expand!

w1h1^2 + 2*w1h1 w1h2 + w1h2^2 + 2*w1h1 w2h3 + 2*w1h2*w2h3 + w2h3^2 + 
 2*w1h1 w2h4 + 2*w1h2*w2h4 + 2*w2h3 w2h4 + w2h4^2 + 2*w1h1 w3h1 + 
 2*w1h2*w3h1 + 2*w2h3 w3h1 + 2*w2h4 w3h1 + w3h1^2 + 2*w1h1 w3h2 + 
 2*w1h2*w3h2 + 2*w2h3 w3h2 + 2*w2h4 w3h2 + 2*w3h1 w3h2 + w3h2^2 + 
 2*w1h1 w4h3 + 2*w1h2*w4h3 + 2*w2h3 w4h3 + 2*w2h4 w4h3 + 
 2*w3h1 w4h3 + 2*w3h2*w4h3 + w4h3^2 + 2*w1h1 w4h4 + 2*w1h2*w4h4 + 
 2*w2h3 w4h4 + 2*w2h4 w4h4 + 2*w3h1 w4h4 + 2*w3h2*w4h4 + 
 2*w4h3 w4h4 + w4h4^2


'''

# 8 unknowns

Q4 = {}
Q4['x0', 'x0'] =
Q4['x1', 'x1'] =
Q4['x2', 'x2'] =
Q4['x3', 'x3'] =

Q4['x4', 'x4'] =
Q4['x5', 'x5'] =
Q4['x6', 'x6'] =
Q4['x7', 'x7'] =

#Coefficients
Q4['x0', 'x1'] =
Q4['x0', 'x2'] =
Q4['x0', 'x3'] =
Q4['x0', 'x4'] =
Q4['x0', 'x5'] =
Q4['x0', 'x6'] =
Q4['x0', 'x7'] =
Q4['x1', 'x2'] =
Q4['x1', 'x3'] =
Q4['x1', 'x4'] =
Q4['x1', 'x5'] =
Q4['x1', 'x6'] =
Q4['x1', 'x7'] =
Q4['x2', 'x3'] =
Q4['x2', 'x4'] =
Q4['x2', 'x5'] =
Q4['x2', 'x6'] =
Q4['x2', 'x7'] =
Q4['x3', 'x4'] =
Q4['x3', 'x5'] =
Q4['x3', 'x6'] =
Q4['x3', 'x7'] =
Q4['x4', 'x5'] =
Q4['x4', 'x6'] =
Q4['x4', 'x7'] =
Q4['x5', 'x6'] =
Q4['x5', 'x7'] =
Q4['x6', 'x7'] =



