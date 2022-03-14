import numpy as np
import copy
import dimod
import random
import math
from math import log
import dimod

# v = our V matrix which is p x n matrix
# w = our W matrix which is p x k
# h = our H matrix which is k x n
# k = # of clusters

'''
    V        W        H
[v11, v12]  [w11, w12]  [h11, h12]
[v21, v22]  [w21, w22]  [h21, h22]

(v11 - (w11 * h11) + (w12 * h21) )^2 +
(v12 - (w11 * h12) + (w12 * h22) )^2 +
(v21 - (w21 * h11) + (w22 * h21) )^2 + 
(v22 - (w21 * h12) + (w22 * h22) )^2
'''

# Annealer will return binary value, need to convert back to reals
# using approximation
# Convert real value by taking floor of float value then make binary
def approx_real(r):
    return bin(math.floor(r).split('0b')[1])

def bin_to_real(binstr):
    return int(binstr,2)

# Figure out all v_ij - sum_h w_ih * h_hj
# pass in k , # of clusters

def find_vars(v,k):

    v_list = list()

    # Store our wh values and in reverse
    x_dict = {}
    x_dict_rev = {}

    # store our v values
    # This will essentially be a set of nested dictionaries
    '''
       (x, y) : { 
                   v_val: { v} , 
                   wh: { (wh_ik, wh_kj) }
                }

    '''
    # Dict for our V position and values in V
    v_dict = {}

    # 
    wh_dict = {}
 
    # Get correct dimensions
    # V is p x n 
    p = v.shape[0]
    n = v.shape[1]

    # W is p x k
    w_rows = p
    w_cols = k

    # H is k x n
    h_rows = k
    h_cols = n


    #Get the V's
    # V is p x n 
    for i in range(0,p):
        for j in range(0, n):
            # stringify what is in V at this location
            i_idx = i+1
            j_idx = j+1
            v_dict[i_idx,j_idx] = {}
            v_dict[i_idx,j_idx]['v_val'] = str(v[i][j])
            v_dict[i_idx,j_idx]['wh'] = []
            #v_str = str(v[i][j]) + "-"
            #v_list.append(v_str)

    # Build WH
    # WH will be same as V , so p x n
    wh_cnt = 1
    for i in range(0,w_rows):
        for j in range(0,h_cols):
            # This is just indexing to make it match and not index from 0
            i_idx = i+1
            j_idx = j+1
            for l in range(0,w_cols):   # This is the column vector selection
                #print("w" + str(i+1) + str(l+1) + "h" + str(l+1) + str(j+1))
                #x_dict['x'+str(wh_cnt)] = ("w" + str(i+1) + str(l+1) + "h" + str(l+1) + str(j+1))
                x_dict['x'+str(wh_cnt)] = ("w" + str(i+1) + str(l+1), "h" + str(l+1) + str(j+1))
                x_dict_rev[("w" + str(i+1) + str(l+1), "h" + str(l+1) + str(j+1))] = 'x'+str(wh_cnt)
                v_dict[i_idx,j_idx]['wh'].append( ("w"+str(i+1) + str(l+1), "h"+str(l+1)+str(j+1)) )
                wh_cnt += 1
               # x_dict['x'+str(i)] = "w" + str(i+1) + str(l+1) + "h" + str(l+1) + str(j+1)

    '''
    for k,v in x_dict.items():
        print(k,":", v)

    for k,v in x_dict_rev.items():
        print(k,":", v)

    for k,v in v_dict.items():
        print(k,":",v)
    '''
    return v_dict, x_dict, n
            

# Ax-b is similar to V-WH
# V = b here
# W = A
# H = x ??
## Code from Ajinkya Borle in nick_quadratic_automated

#Converts a ||Ax-b|| problem into a QUBO
def qubo_prep(A,b,n,bitspower,varnames=None):
    #Input:
    #'A' is the matrix which is a np.array(), a matrix which is m x n
    #b is the vector in the expression : Ax-b, an np.array() of length m
    #n is an integer value that is the length of the column vector x
    #bitspower is a list of powers that would be applied in a 2s complement way, 
    #eg : [2,1,0] would be applied with another bit tacked on front as : -2^(len(bitspower) + 2^(2) + 2^(1) + 2^(0)
    #varnames if provided is a list of variable name. eg : varnames = [x1,x2,x3]
    #Note : the implementation requires the list to be a set of decreasing and consecutive numbers, sorry!
    #Output:
    #A dictionary Qdict which is the QUBO for this problem
    #A dictionary Qdict_alt which the same QUBO but with alphanumeric variable (keys) names
    n_i_ctr = 0
    i_powerctr = 0
    i_twosymb = 0
    
    str_bitspower = ['null'] + [str(item) for item in bitspower]
    powersoftwo = np.zeros(len(bitspower)+1)
    
    for i in range(0,len(bitspower)+1):
        if i==0:
            powersoftwo[i] = (2**(bitspower[i]+1))*(-1)
        else:
            powersoftwo[i] = 2**(bitspower[i-1])
    Qinit = np.zeros([n,n]) #A proto-QUBO that preprocesses some values for the actual QUBO. Initial QUBO if you will
    Qdict = {} #A dictionary that stores the final QUBO
    
    Qdict_alt = {} #This dictionary saves the alternate QUBO with alphanumeric variable (keys) names
    index_dict = {} #Dictionary that maps alphanumeric variables to the integer ones
    
    for i in range(0,n):
        for j in range(i,n):
            Qinit[i,j] = 2*sum(A[:,i]*A[:,j])    
    bnew = 2*b
    
    
    for i in range(0,n*len(powersoftwo)):
        if i%len(powersoftwo)==0 and i>0:
            n_i_ctr = n_i_ctr + 1
            i_powerctr=0
        n_j_ctr = n_i_ctr
        j_powerctr = i_powerctr
        for j in range(i,n*len(powersoftwo)):
            if i==j:#Linear coefficient
                Qdict[i,i] = (powersoftwo[i_powerctr]**2)*(sum(A[:,n_i_ctr]**2)) - powersoftwo[i_powerctr]*sum(A[:,n_i_ctr]*bnew)
                if varnames != None:
                    tempvar1 = varnames[n_i_ctr] + '_' + str_bitspower[i_powerctr]
                    index_dict[tempvar1] = i
                    Qdict_alt[tempvar1,tempvar1] = Qdict[i,i]
            else:#Quadratic coefficient
                if j%len(powersoftwo)==0 and j>0:
                    n_j_ctr = n_j_ctr + 1
                    j_powerctr = 0
                Qdict[i,j] = powersoftwo[i_powerctr]*powersoftwo[j_powerctr]*Qinit[n_i_ctr,n_j_ctr]
                if varnames != None:
                    tempvar2 = varnames[n_j_ctr] + '_' + str_bitspower[j_powerctr]
                    Qdict_alt[tempvar1,tempvar2] = Qdict[i,j]
            
            j_powerctr = j_powerctr + 1
        i_powerctr = i_powerctr + 1
    
    if varnames != None:
        return Qdict, Qdict_alt, index_dict
    else:
        return Qdict #just return the bare bones if varnames not requested

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

#processes the binary string into a np.array vector.
def qubo_to_real(binstr,n,prec_list):
    #prepare the powers_of_two list
    
    powers_of_two = []
    powers_of_two.append(-2**(prec_list[0]+1))
    
    
    for i in range(0,len(prec_list)):
        powers_of_two.append(2**(prec_list[i]))
    #Now the actual number
    bin_ctr=0
    cur_real = np.zeros(n)
    for i in range(0,n):
        for j in range(0,len(powers_of_two)):
            cur_real[i] += powers_of_two[j]*int(binstr[bin_ctr])
            bin_ctr += 1
    
    return cur_real

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
    bnew = 2*b   
    
    for i in range(0,n*len(powersoftwo)):
        if i%len(powersoftwo)==0 and i>0:
            n_i_ctr = n_i_ctr + 1
            i_powerctr=0
        n_j_ctr = n_i_ctr
        j_powerctr = i_powerctr
        for j in range(i,n*len(powersoftwo)):
            if i==j: #Linear coefficient
                Qdict[i,i] = (powersoftwo[i_powerctr]**2)*(sum(A[:,n_i_ctr]**2)) - powersoftwo[i_powerctr]*sum(A[:,n_i_ctr]*bnew)
                if varnames != None:
                    tempvar1 = varnames[n_i_ctr] + '_' + str_bitspower[i_powerctr]
                    index_dict[tempvar1] = i
                    Qdict_alt[tempvar1,tempvar1] = Qdict[i,i]
            else: #Quadratic coefficient
                if j%len(powersoftwo)==0 and j>0:
                    n_j_ctr = n_j_ctr + 1
                    j_powerctr = 0
                Qdict[i,j] = powersoftwo[i_powerctr]*powersoftwo[j_powerctr]*Qinit[n_i_ctr,n_j_ctr]
                if varnames != None:
                    tempvar2 = varnames[n_j_ctr] + '_' + str_bitspower[j_powerctr]
                    Qdict_alt[tempvar1,tempvar2] = Qdict[i,j]
            
            j_powerctr = j_powerctr + 1
        i_powerctr = i_powerctr + 1
    
    if varnames != None:
        return Qdict, Qdict_alt, index_dict
    else:
        return Qdict #just return the bare bones if varnames not requested

#Function to convert the solution dictionary from alphanumeric variables to integer 
def convert_result(soln_dict,index):
    new_dict = {}
    for key,value in soln_dict.items():
        new_dict[index[key]] = value
    return new_dict

    


'''
p x n 
2 x 3

p x k 
2 x 2

k x n 
2 x 3

[v11, v12]  [w11, w12]  [h11, h12]
[v21, v22]  [w21, w22]  [h21, h22]

(v11 - (w11 * h11) + (w12 * h21) )^2 +
(v12 - (w11 * h12) + (w12 * h22) )^2 +
(v21 - (w21 * h11) + (w22 * h21) )^2 + 
(v22 - (w21 * h12) + (w22 * h22) )^2


2 x 3 

V          W             H
[1,2,3]  [w11, w12]     [h11, h12, h13]
[4,5,6]  [w21, w22]     [h21, h22, h23] 

w11h11 + w12h21, w11h12 + w12h22, w11h13 + w12h23
w21h11 + w22h21, w21h12 + w22h22, w21h13 + w22h23

V - WH

(1 - (w11h11 + w12h21))^2 + 
(2 - (w11h12 + w12h22))^2 + 
(3 - (w11h13 + w12h23))^2 + 
(4 - (w21h11 + w22h21))^2 + 
(5 - (w21h12 + w22h22))^2 + 
(6 - (w21h13 + w22h23))^2 


3 x 3 

V (3x3)         W (3 x 2)        H(2 x 3)

[1,2,3]       [w11, w12]        [h11, h12, h13]
[4,5,6]       [w21, w22]        [h21, h22, h23]
[7,8,9]       [w31, w32]

w11h11 + w12h21, w11h12 + w12h22, w11h13 + w12h23
w21h11 + w22h21, w21h12 + w22h22, w21h13 + w22h23
w31h11 + w32h21, w31h12 + w32h22, w31h13 + w32h23

(1- (w11h11 + w12h21))^2 + 
(2- (w11h12 + w12h22))^2 + 
(3-
(4-
(5-
(6-
(7-
(8-
(9- (w31h13 + w32h23))^2

'''





#print("\n2x2\n")
#find_vars(v, k)

#print("\n2x3\n")
#find_vars(a, k)

#print("\n3x3\n")
#find_vars(b,k)


#for k,v in x_dict.items():
   # print(k,":",v)



             
#(6 - x1 - x2)^2
#Inorder to repurpose a code for ||Ax - b|| for individual quadratic equations for n variables,THe Dimensions for A is 1 x n, for x : n x 1 and b : 1 x 1
#A = np.array([[1,1]]) 
#prec_list = [1,0]
#b = np.array([6])
#n=2
#varnames = ['x1','x2']
###Q,Q_alt,index = qubo_prep(A,b,n,prec_list,varnames=varnames)
#sampler = dimod.ExactSolver()
#sampleset = sampler.sample_qubo(Q_alt)
#print(sampleset.first.sample)


#multiply Ax
# Ax is the sum of w times h vars
# wh elements are substituded by x variables
# do b - Ax for the values)
# A is a matrix
#   rows are all 1's
#   columns are # of variables


'''
    V        W        H
[v11, v12]  [w11, w12]  [h11, h12]
[v21, v22]  [w21, w22]  [h21, h22]

(v11 - (w11 * h11) + (w12 * h21) )^2 +   #v11 - (w11h11) + (w12h21) 
(v12 - (w11 * h12) + (w12 * h22) )^2 +   #v12 - (w11h12) + (w12h22)
(v21 - (w21 * h11) + (w22 * h21) )^2 +   #v21 - (w21h11) + (w22h21)
(v22 - (w21 * h12) + (w22 * h22) )^2     #v22 - (w21h12) + (w22h22)

w11h11(w11h11 + w12h21) + w12h21(w11h11 + w12h21) 

w11h11^2 + w11h11w12h21 + w12h21w11h11 + w12h21^2 = w11h11^2 + 2(w11h11w12h12) + w12h21^2 

w11h12(w11h12 + w12h22) + w12h22(w11h12 + w12h22) = w11h12^2 + 2(w11h12w12h22) + w12h22^2

w21h11(w21h11+w22h21) + w22h21(w21h11+w22h21) = w21h11^2 + 2(w21h11w22h21) + w22h21^2

w21h12(w21h12 + w22h22) + w22h22(w21h12+w22h22) = w21h12^2 + 2(w21h12w22h22) + w22h22^2



(v11 - w11h11^2 + 2(w11h11w12h12) + w12h21^2) + (v12 -  w11h12^2 + 2(w11h12w12h22) + w12h22^2 ) +
 (v21 - w21h11^2 + 2(w21h11w22h21) + w22h21^2) + (v22 - w21h12^2 + 2(w21h12w22h22) + w22h22^2 )


(v11 - x1^2 + 2(x1 * x2) + x2^2) + (v12 - x3^2 + 2(x3 * x4) + x4^2 ) + (v21 - x5^2 + 2(x5 * x6) + x6^2 ) + (v22 - x7^2 + 2(x7 * x8) + x8^2)


(a+b)^2 = (a+b)(a+b) = a^2 + b^2 + 2ab

x1 : ('w11', 'h11')
x2 : ('w12', 'h21')
x3 : ('w11', 'h12')
x4 : ('w12', 'h22')
x5 : ('w21', 'h11')
x6 : ('w22', 'h21')
x7 : ('w21', 'h12')
x8 : ('w22', 'h22')




'''

# In a 2x2 situation, we basicallay have to send one quadratic expression at a time
Q = {}
Q_alt = {}
index = {}
Q_total = {}
    
###

#V = np.array([[2.5,7.3],[3.5,2]]

v = np.array([[1,2], [3,4]])   #2x2
#a = np.array([ [1,2,3], [3,4,5] ])  # 2 x 3
#b = np.array([ [1,2,3], [4,5,6], [7,8,9] ])   # 3x3


#for make qubo, V = B as our input
prec_list = [1,0]
b = v

# v = our V matrix which is p x n matrix
# w = our W matrix which is p x k
# h = our H matrix which is k x n
# k = # of clusters


k = 2
A = np.zeros([1,k])
A += 1  # make matrix all 1's

v_rows = v.shape[0]
v_cols = v.shape[1]

v_dict, x_dict, n = find_vars(v,k)

#print(v_dict)
#print(x_dict)
varnames = []

# Get x vector names/symbols
for k,v in x_dict.items():
    varnames.append(k)



Q,Q_alt,index = qubo_prep_nonneg(A,b,n,prec_list,varnames=varnames)

print("\nQ\n")
for i,j in Q.items():
    print(i,":",j)

print("\nQ_alt\n")
for i,j in Q_alt.items():
    print(i,":",j)

print("\nindex\n")
for i,j in index.items():
    print(i,":",j)

#qubo_prep(A,b,n,prec_lifst,varnames=varnames)
