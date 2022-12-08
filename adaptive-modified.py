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

def qubo_prep_nonneg(A,b,n,bitspower, varnames=None):
    #Same as qubo_prep but only for non-negative values
    #bitspower = [0] for binary values
    
    
    n_i_ctr = 0
    i_powerctr = 0
    i_twosymb = 0
    
    str_bitspower =[str(item) for item in bitspower]
    powersoftwo = np.zeros(len(bitspower))
    
    for i in range(0,len(bitspower)):
        powersoftwo[i] = 2**(bitspower[i])
        
    Qinit = np.zeros([n,n])
    Qdict = {} #The dictionary for our qubo
    
    Qdict_alt = {} #This dictionary saves the alternate QUBO with alphanumeric variable (keys) names
    index_dict = {} #Dictionary that maps alphanumeric variables to the integer ones
    
    for i in range(0,n):
        for j in range(i,n):
            Qinit[i,j] = 2*sum(A[:,i]*A[:,j])
            #print("qinit_i_j: ", Qinit[i,j])
            #print("A_:,i",i,":", A[:,i])
            # print("A_:,j",j,":", A[:,j])
    bnew = 2*b
    # print("b: ", b)
    # print("2*b:", bnew)

    
    for i in range(0,n*len(powersoftwo)):
        if i%len(powersoftwo)==0 and i>0:
            n_i_ctr = n_i_ctr + 1
            i_powerctr=0
        n_j_ctr = n_i_ctr
        j_powerctr = i_powerctr
        for j in range(i,n*len(powersoftwo)):
            if i==j: #Linear coefficient
                Qdict[i,i] = (powersoftwo[i_powerctr]**2)*(sum(A[:,n_i_ctr]**2)) - powersoftwo[i_powerctr]*sum(A[:,n_i_ctr]*bnew)

                # print("qdict_i_i: ", Qdict[i,i])
                if varnames != None:
                    tempvar1 = varnames[n_i_ctr] + '_' + str_bitspower[i_powerctr]
                    index_dict[tempvar1] = i
                    Qdict_alt[tempvar1,tempvar1] = Qdict[i,i]
            else: #Quadratic coefficient
                if j%len(powersoftwo)==0 and j>0:
                    n_j_ctr = n_j_ctr + 1
                    j_powerctr = 0
                Qdict[i,j] = powersoftwo[i_powerctr]*powersoftwo[j_powerctr]*Qinit[n_i_ctr,n_j_ctr]
                #print("qdict_i_j:", Qdict[i,j])
                if varnames != None:
                    tempvar2 = varnames[n_j_ctr] + '_' + str_bitspower[j_powerctr]
                    Qdict_alt[tempvar1,tempvar2] = Qdict[i,j]
            
            j_powerctr = j_powerctr + 1
        i_powerctr = i_powerctr + 1
    
    if varnames != None:
        return Qdict, Qdict_alt, index_dict
    else:
        return Qdict #just return the bare bones if varnames not requested

def qubo_prep_adaptive(A,b,n,scale_list,offset_list,bits_no,varnames=None):
    n_i_ctr = 0
    i_powerctr = 0
    i_twosymb = 0
    offset_vector = np.array(offset_list)
    offset_total = np.matmul(A,offset_vector)
    Qinit = np.zeros([n,n])
    Qdict = {}
    powersoftwo = np.zeros(bits_no)
    
    Qdict_alt = {} #This dictionary saves the alternate QUBO with alphanumeric variable (keys) names
    index_dict = {} #Dictionary that maps alphanumeric variables to the integer ones


    #powersoftwo
    for i in range(0,bits_no):
        powersoftwo[i] = 2**(bits_no - i-1)
    
    str_bitspower = [str(item) for item in range(bits_no-1,-1,-1)] #For the labelled variables, goes from [bits_no -1 ... 0]

    #prepare Qinit
    for i in range(0,n):
        for j in range(i,n):
            Qinit[i,j] = 2*sum(A[:,i]*A[:,j])    
    
    bnew = float(copy.deepcopy(b))
    bnew -= offset_total #we subtract offset here coz the whole bnew is gonna be subtracted from other parts
    bnew = 2*bnew

    
    ### Check ipad notebook, the math is written down, sorta

    for i in range(0,n*bits_no):
        if i%bits_no==0 and i>0:
            n_i_ctr = n_i_ctr + 1
            i_powerctr=0
        n_j_ctr = n_i_ctr
        j_powerctr = i_powerctr

        for j in range(i,n*bits_no):
            if i==j: #Linear coefficients
                Qdict[i,i] = (scale_list[n_i_ctr]**2)*(powersoftwo[i_powerctr]**2)*(sum(A[:,n_i_ctr]**2)) - scale_list[n_i_ctr]*powersoftwo[i_powerctr]*sum(A[:,n_i_ctr]*bnew)

                ## THIS IS WHERE IT BREAKS DUE TO INDEXING ISSUE
                ## Probably need to use better counters

                if varnames != None:
                    tempvar1 = varnames[n_i_ctr] + '_' + str_bitspower[i_powerctr]
                    #print("tempvar1", tempvar1)
                    index_dict[tempvar1] = i
                    Qdict_alt[tempvar1,tempvar1] = Qdict[i,i]
                    #print("Qdict[i][i]:", Qdict[i,i])
            else:
                if j%bits_no==0 and j>0:
                    n_j_ctr = n_j_ctr + 1
                    j_powerctr = 0
                
                Qdict[i,j] = scale_list[n_i_ctr]*scale_list[n_j_ctr]*powersoftwo[i_powerctr]*powersoftwo[j_powerctr]*Qinit[n_i_ctr,n_j_ctr]

            ## Here is where it dies out due to indexing issues
                if varnames != None:
                    #print("J: ", j)
                    #print("n_j_ctr", n_j_ctr)
                    #print("j_pwr_ctr", j_powerctr)
                    #print("varnames[n_j_ctr", varnames[n_j_ctr])
                    #print("str_bitspower[j_powerctr]", str_bitspower[j_powerctr])
                    tempvar2 = varnames[n_j_ctr] + '_' + str_bitspower[j_powerctr]
                    #print("tempvar2:", tempvar2)
                    #print("Qdict[i,j]", Qdict[i,j])
                    Qdict_alt[tempvar1,tempvar2] = Qdict[i,j]


            j_powerctr = j_powerctr + 1
            #print("i: ", i)
            #print("n_i_ctr", n_i_ctr)
            #print("i_pwr_ctr", i_powerctr)
                            
            #print("len of bitspower", len(str_bitspower))
            #print("Len of varnames", len(varnames))

        i_powerctr = i_powerctr + 1

    if varnames != None:
        return Qdict, Qdict_alt, index_dict
    else:
        return Qdict #just return the bare bones if varnames not requested

def get_scale_offset(binstr,scale,offset,upper_limit,lower_limit,bits_no,scale_factor):
    num_binstr = int(binstr,2)
    new_scale = None
    new_offset = None
    highest_num = (2**(bits_no) - 1)
    
    if (num_binstr == 0) and (offset == lower_limit): #if number returned is equivalent to 0 + offset chosen and if offset is the original lower_limit
        new_scale = scale/scale_factor
        new_offset = offset
    elif (num_binstr == highest_num) and (upper_limit == highest_num*scale + lower_limit): #if number returned is equivalent to 2^(num_bits) + offset and if max value the original high
        new_scale = scale/scale_factor
        new_offset = (num_binstr)*scale + offset - (num_binstr)*new_scale
    else:
        new_scale = scale/scale_factor
        new_offset = (num_binstr)*scale + offset - 3*new_scale #we want an offset and scale s.t the old number can be represented by 3*new_scale + offset
        #NOTE : the original limits may not be followed 100%
    
    return new_scale,new_offset

def qubo_to_real_adaptive(binstr,n,scale_list,offset_list,bits_no):
    powersoftwo = np.zeros(bits_no)
    
    #powersoftwo
    for i in range(0,bits_no):
        powersoftwo[i] = 2**(bits_no - i-1)
        
    bin_ctr=0
    cur_real = np.zeros(n)
    for i in range(0,n):
        for j in range(0,bits_no):
            cur_real[i] += scale_list[i]*powersoftwo[j]*int(binstr[bin_ctr])
            bin_ctr += 1
        cur_real[i] += offset_list[i]
    
    return cur_real

#Function to convert the solution dictionary from alphanumeric variables to integer 
def convert_result(soln_dict,index):
    new_dict = {}
    for key,value in soln_dict.items():
        new_dict[index[key]] = value
    return new_dict

#This is just to convert a dictionary based result into a binary string based result
def get_bin_str(config,isising=True):
    #Input:
    #config is a dictionary
    #isising is True if config has -1 or +1 and False if config has 0 or 1
    #Output:
    # a binary string of 0s and 1s
    binstr = ""
    if isising == True:
        for i in range(0,len(config)):
            if config[i] == 1:
                binstr += str(1)
            elif config[i] == -1:
                binstr += str(0)
    else:
        for i in range(0,len(config)):
            if config[i] == 1:
                binstr += str(1)
            elif config[i] == 0:
                binstr += str(0)
    return binstr

# parsed v_dict
# parsed x_dict_rev , 
# prec_list -- for W
# prec_list str -- for W
# prec_list2 -- for H
# delta1, delta2 = lagarange params
# n 
# bits_no
# scale_list
# offset_list
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

def linearization(self):
    # print("Applying linearization penalties...\n")
    # linearization
    # Delta == lagaragne param
    #delta = 50
    for x_key, x_val in self.x_dict.items():
        temp_h = x_val[1]
        #print(temp_h)
        for prec in self.prec_list_str:
            temp_x = x_key + "_" + prec
            #print(test)
            temp_w = x_val[0] + "_" + prec
            #print(temp_w)
            self.Q_total[temp_w, temp_h] = 2 * self.delta1
            self.Q_total[temp_x, temp_w] = -4 * self.delta1
            self.Q_total[temp_x, temp_h] = -4 * self.delta1
            self.Q_total[temp_x, temp_x] += 6 * self.delta1
    
def h_penalty(self, Q2_alt):
    #print("Applying H penalties...\n")
    Q_alt2 = {} # new dict for Q_alt but diff key names

    for key,value in Q2_alt.items():
        #Erase all the characters in the key after underscore
        temp_key = (key[0].split('_')[0],key[1].split('_')[0])
        Q_alt2[temp_key] = value * self.delta2

    #print("H Penalties: \n")
    #pprint.pprint(Q_alt2)

    #print("Adding all data to Q_total...\n")
    # Add all to Q_total for H
    for key, val in Q_alt2.items():
        if key in self.Q_total:
            self.Q_total[key] += val
        else:
            self.Q_total[key] = val

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

## First part, build q_Total 
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

    # Now we have our varnames and A and B built, lets iterate 
    #while LA.norm(np.matmul(A,x_cur)- b) > tolerance:
    while(w_itr < 5):
        print("scale_list: ",scale_list," offset_list: ",offset_list)
        print("A:\n", A)
        print("b:\n", b)
        print("x_cur:\n", x_cur)

        Q, Q_alt,index = qubo_prep_adaptive(A,b,k,scale_list,offset_list,bits_no,varnames=varnames)

        # Lets run it against the annealer
        sampler = tabu.TabuSampler()

        sampleset = sampler.sample_qubo(Q_alt, timeout=10)

        # result
        soln_dict = convert_result(sampleset.first.sample, index)

        #convert solution into binary string
        binstr = get_bin_str(soln_dict,isising=False)
        binstr_vec = ['' for i in range(0,k)]
        temp_ctr = 0

        for i in range(0,k):
            for j in range(0,bits_no):
                binstr_vec[i]+= binstr[temp_ctr]
                temp_ctr += 1

        x_cur = qubo_to_real_adaptive(binstr,k,scale_list,offset_list,bits_no)
        x_cur = np.array(x_cur)
        #print("Iteration: ",itr, " x_cur: ",x_cur, " cur norm: ",LA.norm(np.matmul(A,x_cur)- b))
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

    # We're out of the iteration so lets add it to Q_Total to be used for WH later
    for key, val in Q_alt.items():
        # Check if key is already here, if so add to it
        if key in Q_total:
            Q_total[key] += val
        else:
            Q_total[key] = val
    


## END of Q_total loop

print("Applying linearization penalties")

penal = Penalizer(x_dict, delta1, delta2, Q_total, prec_list_str)
prec_list2 = [0] #all variables are binary, DO NOT CHANGE VALUE
b2 = np.array([1]) # This 1 enforces only one variable to be a 1 :D

print("Applying penalties to H...")
for h_i in range(0, n):        # row
    varnames2 = []
    for h_j in range(0, k):    # col
        varnames2.append('h'+str( (h_j+1) ) + str( (h_i+1) ))
        #pprint.pprint(varnames2)

    A = np.zeros([1, k])
    A += 1
    #pprint.pprint(varnames2)    
     # def qubo_prep_nonneg(self, A,b,n,bitspower, varnames=None):
    Q2, Q2_alt, index = qubo_prep_nonneg(A, b2, k, prec_list2, varnames=varnames2)

    penal.h_penalty(Q2_alt)


print("Sending Q_Total to Solver...")

sol = {}
sampleset = sampler.sample_qubo(Q_total, timeout=10)

sol = sampleset.first.sample

#pprint.pprint(sol)

## This doesn't work as we expect, as we don't get floating point stuff..

for skey, sval in sol.items():
    print(skey, ":", sval)

exit(1)
print("Creating verification W and H...\n")
for i in range(0,k):
    for j in range(0,n):
        temp_h = "h" + str(i+1) + str(j+1)
        H[i,j] = sol[temp_h]
        
for i in range(0,p):
    for j in range(0,k):
        temp_w = "w" + str(i+1) + str(j+1)
        for sol_key, sol_val in sol.items():
            if temp_w in sol_key:
                #print(temp_w, sol_key)
                temp_str = sol_key.split('_')[1]
                #print(temp_str)
                if temp_str == "null":
                    W[i,j] += -(2**(prec_list[0]+1))*sol_val
                else:
                    W[i,j] += (2**int(temp_str))*sol_val




### Create W and H based on soln_dict or really sampleset.first.sample
    


