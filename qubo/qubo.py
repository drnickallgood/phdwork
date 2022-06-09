import numpy as np
import copy
import dimod
import random
import math
from math import log
import pprint
import neal
import numpy.linalg as LA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from datetime import datetime
import tabu
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import sys

class Qubo:
    def __init__(self, dataset, num_centers, num_samples):
        self.dataset = dataset
        self.num_centers = num_centers
        self.num_samples = num_samples 
        
    
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
        
    def convert_result(soln_dict,index):
        new_dict = {}
        for key,value in soln_dict.items():
            new_dict[index[key]] = value
        return new_dict

    def count_ones(H):
        bad_cols = 0
        for col_num in range(0, H.shape[1]):
            col_count = 0
            for row_num in range(0, H.shape[0]):
                #print("Row: ", row_num, "Col: ", col_num)
                if H[row_num][col_num] == 1:
                    col_count += 1
                
            if col_count > 1 or col_count == 0:
                print("Bad solution, multiple or no 1's in column: ", col_num)
                bad_cols += 1
                
                    
            print("Col: ", col_num, "1count: ", col_count)
        print("Total Violated Columns: ", bad_cols)
        
    def print_test():
        print("test failed successfully")
    
    # Annealer will return binary value, need to convert back to reals
    # using approximation
    # Convert real value by taking floor of float value then make binary
    def approx_real(self,r):
        return bin(math.floor(r).split('0b')[1])

    def bin_to_real(self,binstr):
        return int(binstr,2)

    # Figure out all v_ij - sum_h w_ih * h_hj
    # pass in k , # of clusters

    def find_vars(self, v,k):

        v_list = list()

        # Store our wh values and in reverse
        x_dict = {}
        x_dict_rev = {}

        # store our v values
        # This will essentially be a set of nested dictionaries

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

        return v_dict, x_dict, x_dict_rev, p, n
            

#if (__name__ == '__main__'):
#    Qubo.print_test()
