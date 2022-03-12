import numpy as np
import copy
import dimod
import random
import math
from math import log
import dimod

# v = our V matrix which is p x n matrix
# w = our W matrix which is n x k
# h = our H matrix which is k x n
# k = # of clusters

def approx(v):

    return np.floor(v)

    
    
# Equation is good, but we only will have matrix to work with
# do not make equation and parse it as a string
#
'''


    V        W        H
[v11, v12]  [w11, w12]  [h11, h12]
[v21, v22]  [w21, w22]  [h21, h22]

(v11 - (w11 * h11) + (w12 * h21) )^2 +
(v12 - (w11 * h12) + (w12 * h22) )^2 +
(v21 - (w21 * h11) + (w22 * h21) )^2 + 
(v22 - (w21 * h12) + (w22 * h22) )^2


00 00 + 01 10
00 01 + 01 11



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
'''
    V        W        H
[0, 1]    [0, 1]    [0, 1]
[2, 3]    [2, 3]    [2, 3]

'''

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
    p = v.shape[0]
    n = v.shape[1]

    w_rows = p
    w_cols = k

    h_rows = k
    h_cols = n


    #Get the V's
    for i in range(0,len(v)):
        for j in range(0,len(v)):
            # stringify what is in V at this location
            i_idx = i+1
            j_idx = j+1
            v_dict[i_idx,j_idx] = {}
            v_dict[i_idx,j_idx]['v_val'] = str(v[i][j])
            v_dict[i_idx,j_idx]['wh'] = []
            #v_str = str(v[i][j]) + "-"
            #v_list.append(v_str)

    # Build WH
    wh_cnt = 1
    for i in range(0,w_rows):
        for j in range(0,w_cols):
            i_idx = i+1
            j_idx = j+1
            for l in range(0,h_rows):   # This is the column vector selection
                #print("w" + str(i+1) + str(l+1) + "h" + str(l+1) + str(j+1))
                #x_dict['x'+str(wh_cnt)] = ("w" + str(i+1) + str(l+1) + "h" + str(l+1) + str(j+1))
                x_dict['x'+str(wh_cnt)] = ("w" + str(i+1) + str(l+1), "h" + str(l+1) + str(j+1))
                x_dict_rev[("w" + str(i+1) + str(l+1), "h" + str(l+1) + str(j+1))] = 'x'+str(wh_cnt)
                v_dict[i_idx,j_idx]['wh'].append( ("w"+str(i+1) + str(l+1), "h"+str(l+1)+str(j+1)) )
                wh_cnt += 1
               # x_dict['x'+str(i)] = "w" + str(i+1) + str(l+1) + "h" + str(l+1) + str(j+1)


    for k,v in x_dict.items():
        print(k,":", v)

    for k,v in x_dict_rev.items():
        print(k,":", v)

    for k,v in v_dict.items():
        print(k,":",v)
            

            
            
            
            

    

# Ax-b is similar to V-WH
# V = b here
# W = A
# H = x ??

def make_qubo(v,bits,n):

    Q = {}

    # Create a np array that has our listing of powers including an extra for the
    # sign qubit
    #
    # V-WH == -V + WH = -WH + V , V=WH && WH-V = 0
    powersoftwo = np.zeros(len(bits)+1)

    
    # Fill up our powersof two by taking 2^bit in our list
    for i in range(0, len(bits)+1):
        if i == 0:
            powersoftwo[i] = (2 ** bits[i]+1) * (-1) # Sign qubit
        else:
            powersoftwo[i] = (2 ** bits[i-1])
            
    
    lvar = 'l'
    pvar = 'x'
    qvar = 'q'
    #n is our column vector length
    col_cnt = 0
    power_ctr = 0
    twosymb_ctr = 0

    # We need to do a quadratic expansion that includes the unknowns
    
    
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

    Qinit = np.zeros([n,n])

    # This gives us our quadratic expansino of an input matrix
    # aborle dissertation (3.6) - w_jk = 2 * sum(A_ij * A_ik)
    # v is our constants
    
    vnew = 2*v
    
   # for i in range(0,n):
        #for j in range(i,n):
          #  Qinit[i,j] = 2 * sum(V[:,i] * V[:,j])

            

            
    # (v0 - (w0 * h0) + (w1 * h2) )^2 +
   # for i in range(0,len(a)):
       # for j in range(0,len(a)):
        #    Q[i,j] = v[i,j]
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

    return Qinit

    

    
###

#V = np.array([[2.5,7.3],[3.5,2]]

v = np.array([[1,2], [3,4]])

bits = [1,0]    # gets us range -4 to +3
k = 2

#Q = make_qubo(v,bits,n)

#print(Q)


find_vars(v, k)



             
