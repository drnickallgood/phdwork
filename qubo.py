import numpy as np
import copy
import dimod
import random
import math
from math import log
import dimod
import pprint
import neal
import numpy.linalg as LA

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
    return v_dict, x_dict, x_dict_rev, n
            

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

#v = np.array([[1,2], [3,4]])   #2x2
v = np.array([[3,2], [3,1]])
#a = np.array([ [1,2,3], [3,4,5] ])  # 2 x 3
#b = np.array([ [1,2,3], [4,5,6], [7,8,9] ])   # 3x3


#for make qubo, V = B as our input
# Prec List won't always be like this, prob need to dynamically figure it out based
# on inputs..
prec_list = [1,0]
prec_list_str = ['null', '1', '0']


# v = our V matrix which is p x n matrix
# w = our W matrix which is p x k
# h = our H matrix which is k x n (binary)
# k = # of clusters


k = 2
A = np.zeros([1,k])
A += 1  # make matrix all 1's

v_rows = v.shape[0]
v_cols = v.shape[1]

v_dict, x_dict, x_dict_rev, n = find_vars(v,k)


#print(v_dict)
#print(x_dict)
#varnames = []

# Get x vector names/symbols
#for i,j in x_dict.items():
    #varnames.append(i)

#for key, val in v_dict.items():
   # print(key,":",val)

#for key, val in x_dict.items():
#    print(key, ":", val)

'''
(1, 1) : {'v_val': '1', 'wh': [('w11', 'h11'), ('w12', 'h21')]}
(1, 2) : {'v_val': '2', 'wh': [('w11', 'h12'), ('w12', 'h22')]}
(2, 1) : {'v_val': '3', 'wh': [('w21', 'h11'), ('w22', 'h21')]}
(2, 2) : {'v_val': '4', 'wh': [('w21', 'h12'), ('w22', 'h22')]}
'''

# v_11 - x1 - x2
# v_12 - x3 - x4
# v_21 - x5 - x6
# v_22 - x7 - x8

#print(v_dict[(1,1)['v_val'])
#print(v_dict[(1,1)]['wh'])
#print(v_dict[(1,1)]['wh'][0][0])
#print(x_dict_rev)

# So we need to go through and do one quadratic at a time
# We send only the varnames associated with the v - wh
# we loop through everything in v_dict
# We get all WH's as indexes to x_dict_rev
# We use this to get our x values to pass to varnames
# once we get everything we then call make_qubo
# We get the q_alt return and add it to q_total
#print(v_dict[(1,1)]['wh'][1])

# Go through main dictionary to get data
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
        # Get each element of V for qubo_prep
        # print(varnames)
        # Also store them as a floating point number vs a string
    #for v_key, v_val in v_dict.items():
    b = float(v_dict[key]['v_val'])
    #print(b)
    Q, Q_alt, index = qubo_prep(A,b,n,prec_list,varnames=varnames)
    # Put everything from each Q_alt dict into master Q_total
    for key, val in Q_alt.items():
        # Check if key is already here, if so add to it
        if key in Q_total:
            Q_total[key] += val
        else:
            Q_total[key] = val


#print(Q_total)
#pprint.pprint(Q_total)

# linearization
# Delta == lagaragne param
delta = 5
for x_key, x_val in x_dict.items():
    temp_h = x_val[1]
    #print(temp_h)
    for prec in prec_list_str:
        temp_x = x_key + "_" + prec
        #print(test)
        temp_w = x_val[0] + "_" + prec
        #print(temp_w)
        Q_total[temp_w, temp_h] = 2 * delta
        Q_total[temp_x, temp_w] = -4 * delta
        Q_total[temp_x, temp_h] = -4 * delta
        Q_total[temp_x, temp_x] += 6 * delta

#pprint.pprint(Q_total)

p = v.shape[0]
#n = v.shape[1]


## Need to do like we did above with the h variables and not send them in all at once
## Make as nay Q_alt2 as there are columns in H
## Each column call qubo_prep_nonneg with the correct varnames to that column
'''
col1 = 11, 21, 31...
col2 = 12, 22, 32...
'''








#pprint.pprint(sampleset.first.sample)


        
#Q,Q_alt,index = qubo_prep_nonneg(A,b,n,prec_list,varnames=varnames)

#for q in range(0,v_rows):
#    for r in range(0,v_cols):
#        print(v[q][r])
        #A = np.zeros([1,k])
        #A += 1  # make matrix all 1's
        #Q,Q_alt,index = qubo_prep_nonneg(A,b[q][r],n,prec_list,varnames=varnames)


## So here we are going to try the H penalty where we ensure H only has 1 selection
# We go through this process for H only , then we add the results to Q_Total dict
# n will be length of h colum, so this is k
# A matrix will be all A 1's length of k
# Call qubo_noneg n times
# This is really bad analysis space wise, but we can approximate, can optmize somewhat
#       x: p * k * n
#       w: c * p * k  where c = precision
#       h: k * n
#   (pn + cp + n) , k is out because it's present in all of them , c doesn't grow with inputsize
# Accuracy / Time -- Ratio.. more accurate with less time, than clasical

# I feel like we could basically just create those varnames dynamically based on number of columns
#And length of columns of course

# refresh on quantum complexity classes...committie will ask


#Lets try (1 - h11 - h21 - h31)^2
#A2 = np.array([[1,1,1,1]])

#A2 = np.zeros([1,k])
#A2 += 1

'''

NEW STUFF for H penalty

'''

prec_list2 = [0] #all variables are binary, DO NOT CHANGE VALUE
b2 = np.array([1]) # This 1 enforces only one variable to be a 1 :D
varnames2 = list()
delta2 = 1  # lagarange multiplier
Q_alt2 = {} # new dict for Q_alt but diff key names

for h_i in range(0, k):
    for h_j in range(0, n):
        varnames2.append('h'+str( (h_j+1) ) + str( (h_i+1) ))
        
    Q2, Q2_alt, index = qubo_prep_nonneg(A, b2, n, prec_list2, varnames=varnames2)
    # Multiply everything by delta 
    for key,value in Q2_alt.items():
        #Erase all the characters in the key after underscore
        #temp_key = (key[0].split('_')[0],key[1].split('_')[0])
        Q_alt2[key] = value*delta2

for key, val in Q_alt2.items():
    if key in Q_total:
        Q_total[key] += val
    else:
        Q_total[key] = val
    

pprint.pprint(Q_alt2)
print ("\n---\n")
pprint.pprint(Q_total)


'''
Current H output, 

{('h11_0', 'h11_0'): -1.0, ('h11_0', 'h21_0'): 2.0, ('h21_0', 'h21_0'): -1.0}

'''

# Stopping here for debugging
exit(1)



sampler2 = dimod.ExactSolver()
sampleset2 = sampler2.sample_qubo(Q_alt2)


sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample_qubo(Q_total, num_sweeps=99999, num_reads=40)

solution_dict = {}
solution_dict = sampleset.first.sample

## This is the verification part

W = np.zeros([p,k])
H = np.zeros([k,n])

for i in range(0,k):
    for j in range(0,n):
        temp_h = "h" + str(i+1) + str(j+1)
        H[i,j] = solution_dict[temp_h]
        
for i in range(0,p):
    for j in range(0,k):
        temp_w = "w" + str(i+1) + str(j+1)
        for sol_key, sol_val in solution_dict.items():
            if temp_w in sol_key:
                #print(temp_w, sol_key)
                temp_str = sol_key.split('_')[1]
                #print(temp_str)
                if temp_str == "null":
                    W[i,j] += -(2**(prec_list[0]+1))*sol_val
                else:
                    W[i,j] += (2**int(temp_str))*sol_val


print("V = ", v)
print("W = ", W)
print("H = ", H)
print("WH = ", np.matmul(W,H))
print("First energy: ", sampleset.first.energy)

print("Norm: ", LA.norm(v - np.matmul(W,H)))
print("Verifying best energy via Frobenius Norm: ", LA.norm(v)**2)

'''

For testing, make up a WH and get a result of V, then test
Make sure to put this in dissertation as test


V = p x n
W = p x k
H = k x n

V = 2 x 2
W = 2 x 2
H = 2 x 2


W           H      =      V
[1, 2]   [1, 0]        [3, 2]       
[2, 1]   [1, 1]        [3, 1]


linearization penalty == lagarange multiplier 
w11h11 = x1 + 2(w11h11 - 2x1(w11 + h11) + 3x1)
Penatly piece = 2(w11h11 - 2x1(w11 + h11) + 3x1)
Not messing with powers of 2
have to do this penalty for every bit that makes up w value.. 

Once the penalty is applied, we then test it by submitting to the solver, for a known
WH with a V, and if we get close, we can verify this. 



'''

