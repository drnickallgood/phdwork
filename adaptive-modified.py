import numpy as np
import copy
import dimod
import random
import math
from math import log
import dimod
import numpy.linalg as LA 
import qubo 
import pprint
import neal
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from datetime import datetime
import tabu
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from sklearn import metrics
from qubo.penalizer import Penalizer


def build_qtotal(v_dict, x_dict_rev,k, prec_list, prec_list2, delta1, delta2, n, bits_no, scale_list, offset_list):
 
    for key, val in v_dict.items():
        #print(v_dict[key]['wh'])
        # Go through individual list of tuples
        varnames = []
        for item in v_dict[key]['wh']:
            # Get our corresponding x values to WH values
            varnames.append(x_dict_rev[item])

        # Build a row vector of 1's for A
        A = np.zeros([1,k])
        A += 1
        b = float(v_dict[key]['v_val'])

        ## Replace this with the adaptive version 

        Q, Q_alt,index = qubo_prep_adaptive(A,b,n,scale_list,offset_list,bits_no,varnames=varnames)
        # Put everything from each Q_alt dict into master Q_total
        for key, val in Q_alt.items():
            # Check if key is already here, if so add to it
            if key in Q_total:
                Q_total[key] += val
            else:
                Q_total[key] = val
                
    penal = Penalizer(x_dict, delta1, delta2, Q_total, prec_list_str)
    print("Applying linearization penalties...")
    penal.linearization()
                
    # Make as many q_alt2s as there are columns in H
    prec_list2 = [0] #all variables are binary, DO NOT CHANGE VALUE
    b2 = np.array([1]) # This 1 enforces only one variable to be a 1 :D
    
    print("Applying penalties to H...")
    for h_i in range(0, n):        # row
        varnames2 = []
        for h_j in range(0, k):    # col
            varnames2.append('h'+str( (h_j+1) ) + str( (h_i+1) ))
            #pprint.pprint(varnames2)

        A = np.zeros([1,k])
        A += 1
        #pprint.pprint(varnames2)    
        Q2, Q2_alt, index = qubo_prep_nonneg(A, b2, k, prec_list2, varnames=varnames2)
    
        penal.h_penalty(Q2_alt)


####

# n = number variables 
num_samples = 20
k = 3
seed=0
upper_limit = 100
lower_limit  = -100
bits_no = 3
num_centroids = k * 2

#(6 - x1 - x2)^2
#Inorder to repurpose a code for ||Ax - b|| for individual quadratic equations for n variables,
# THe Dimensions for A is 1 x n,
# for x : n x 1 and
# b : 1 x 1


# Test Data
V, blob_labels, blob_centers = make_blobs(
    n_samples=num_samples, n_features=2,
    cluster_std=1, center_box=(-8, 7),
    random_state=seed, return_centers=True
)

# Transpose matrix
v = np.transpose(V)

# Get Qubo Vars 

qubo_vars = qubo.parser.Parser(v,k)

# n is number of variables 
v_dict, x_dict, x_dict_rev, p, n = qubo_vars.get_vars()

s = (upper_limit - lower_limit)/(2**(bits_no) - 1)
scale_list = [s for i in range(0,k)]
offset_list = [lower_limit for i in range(0,k)]


# Q_total[key] += v_dict[val]

## try to do this one at a time... instead of looping just yet
# Example sampler sampleset = sampler.sample_qubo(Q_total, timeout=tabu_timeout)
#Q,Q_alt,index = qubo_prep_adaptive(A,b,n,scale_list,offset_list,bits_no,varnames=None)

# V = p x n    , 2 x 20 transpose = 20 x 2
# W = p x k    , 2 x 3 transpose = 3 x 2
# H = k x N    , 3 x 20 transpose = 20 x 3
#
# A = 1 x k
# b = 1   (1,)  NOT 1 x 1 
# 


delta1 = 10
delta2 = 10

prec_list = [2, 1, 0]   #-8 to +7
#prec_list_str = ['null']
prec_list_str = []
prec_strings = [prec_list_str.append(str(x)) for x in prec_list]

W = np.zeros([p, k])
H = np.zeros([k, n])

x_cur = [0 for x in range(0,k)]
x_cur = np.array(x_cur)
itr = 0
w_itr = 0


### Probably need to put the above stuff down in this loop below, maybe?
## But if we do, then we have to be careful when submitting, submitting to the 
## QC multiple times.. we'd have to basically limit it for TABU, sim annealing
## to much shorter times
## also , n is the number of variables we have, so we have to keep that in mind
## when getting our results

## Previous solution dictionary has W and H entries all with binary values, not sure if we
## can use this here

## 

Q_total = {}
tolerance = 10**-2
solution_dict = {}

index_dict = {}


A = np.zeros([1,k])
A += 1

# Starting val for B we fix this later

b = np.array([1])

# while(w_itr < 5):
while LA.norm(np.matmul(A,x_cur)- b) > tolerance:

    # Build Q_total 
    for key, val in v_dict.items():
        #print(v_dict[key]['wh'])
        # Go through individual list of tuples
        varnames = []
        for item in v_dict[key]['wh']:
            # Get our corresponding x values to WH values
            varnames.append(x_dict_rev[item])

        # Build a row vector of 1's for A
        A = np.zeros([1,k])
        A += 1

        ## Will need to maybe look into this for b
        b = float(v_dict[key]['v_val'])

        Q, Q_alt,index = qubo_prep_adaptive(A,b,k,scale_list,offset_list,bits_no,varnames=varnames)

        for ikey, ival in index.items():
            index_dict[ikey] = ival

        #print(index)

        # We're out of the iteration so lets add it to Q_Total to be used for WH later
        for key, val in Q_alt.items():
            # Check if key is already here, if so add to it
            if key in Q_total:
                Q_total[key] += val
            else:
                Q_total[key] = val

    # Linearization penalties
    penal = Penalizer(x_dict, delta1, delta2, Q_total, prec_list_str)
    penal.linearization()



    prec_list2 = [0] #all variables are binary, DO NOT CHANGE VALUE
    b2 = np.array([1]) # This 1 enforces only one variable to be a 1 :D

   # H Penalties 
    for h_i in range(0, n):        # row
        varnames2 = []
        for h_j in range(0, k):    # col
            varnames2.append('h'+str( (h_j+1) ) + str( (h_i+1) ))
 
        Q2, Q2_alt, index2 = qubo_prep_nonneg(A, b2, k, prec_list2, varnames=varnames2)

        for ikey, ival in index2.items():
            index_dict[ikey] = ival

        #print(index2)
        penal.h_penalty(Q2_alt)
    
    # Submit

    ## return penalizer q_total
    Q_total2 = penal.get_penalized_qtotal()

    sampler = tabu.TabuSampler()
    sampleset = sampler.sample_qubo(Q_total2, timeout=10)

    #print(sampleset.first.sample)
    #print(index_dict)
    
    # get results

    # interpret new range and scale 

    ## Issue is here: converting result

    #soln_dict = convert_result(sampleset.first.sample, index)

    soln_dict = sampleset.first.sample 

    #convert solution into binary string
    binstr = get_bin_str(soln_dict,isising=False)
    binstr_vec = ['' for i in range(0,k)]
    temp_ctr = 0

    for i in range(0,k):
        for j in range(0,bits_no):
            binstr_vec[i]+= binstr[temp_ctr]
            temp_ctr += 1

    # Converts binary string returned into a real value based on scale offsets
    x_cur = qubo_to_real_adaptive(binstr,k,scale_list,offset_list,bits_no)
    x_cur = np.array(x_cur)
    print("Iteration: ",itr, " x_cur: ",x_cur, " cur norm: ",LA.norm(np.matmul(A,x_cur)- b))
    new_scale_list = []
    new_offset_list = []

    for i in range(0,k):
        temp_scale,temp_offset = get_scale_offset(binstr_vec[i],scale_list[i],offset_list[i],upper_limit,lower_limit,bits_no,2)
        new_scale_list.append(temp_scale)
        new_offset_list.append(temp_offset)

    scale_list = new_scale_list
    offset_list = new_offset_list
    w_itr += 1
    print("--------------------")



## END of Q_total loop


