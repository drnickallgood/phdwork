import numpy as np
import copy
import dimod
import random
import math
from math import log
import dimod

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


#(6 - x1 - x2)^2
#Inorder to repurpose a code for ||Ax - b|| for individual quadratic equations for n variables,THe Dimensions for A is 1 x n, for x : n x 1 and b : 1 x 1
A = np.array([[1,1]]) 
prec_list = [1,0]
b = np.array([6])
n=2
varnames = ['x1','x2']
Q,Q_alt,index = qubo_prep(A,b,n,prec_list,varnames=varnames)


sampler = dimod.ExactSolver()
sampleset = sampler.sample_qubo(Q_alt)
print(sampleset.first.sample)


#Get the solution
soln_dict = convert_result(sampleset.first.sample,index)
#convert dict to string
soln_bin= get_bin_str(soln_dict,isising=False)
#convert string to np.array x
x = qubo_to_real(soln_bin,n,prec_list)
print(x)

Q = qubo_prep(A,b,n,prec_list)
sampler = dimod.ExactSolver()
sampleset = sampler.sample_qubo(Q)
print(sampleset.first.sample)


#Get the solution
soln_dict = sampleset.first.sample
#convert dict to string
soln_bin= get_bin_str(soln_dict,isising=False)
#convert string to np.array x
x = qubo_to_real(soln_bin,n,prec_list)
print(x)

#(2 - x1 - x2 - x3)^2 =  (x1 + x2 + x3-2)^2 :
#where x is binary
A = np.array([[1,1,1]])
prec_list = [0]
b = np.array([2])
varnames = ['x1','x2','x3']
n=3

Q,Q_alt,index = qubo_prep_nonneg(A,b,n,prec_list,varnames=varnames)

print(Q_alt)
