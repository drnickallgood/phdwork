### ORIGINAL CODE BY AJINKYA BORLE aborle1@umbc.edu ###
import numpy as np
import copy
import dimod
import random
import math
from math import log
import dimod

#Converts a ||Ax-b|| problem into a QUBO
def qubo_prep(A,b,n,bitspower):
    #Input:
    #'A' is the matrix which is a np.array(), a matrix which is m x n
    #b is the vector in the expression : Ax-b, an np.array() of length m
    #n is an integer value that is the length of the column vector x
    #bitspower is a list of powers that would be applied in a 2s complement way, 
    #eg : [2,1,0] would be applied with another bit tacked on front as : -2^(len(bitspower) + 2^(2) + 2^(1) + 2^(0)
    #Note : the implementation requires the list to be a set of decreasing and consecutive numbers, sorry!
    #Output:
    #A dictionary Qdict which is the QUBO for this problem
    
    
    n_i_ctr = 0
    i_powerctr = 0
    i_twosymb = 0
    
    powersoftwo = np.zeros(len(bitspower)+1)
    for i in range(0,len(bitspower)+1):
        if i==0:
            powersoftwo[i] = (2**(bitspower[i]+1))*(-1)
        else:
            powersoftwo[i] = 2**(bitspower[i-1])
    Qinit = np.zeros([n,n])
    Qdict = {} #Same qubo but in dictionary format
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
            if i==j:
                Qdict[i,i] = (powersoftwo[i_powerctr]**2)*(sum(A[:,n_i_ctr]**2)) - powersoftwo[i_powerctr]*sum(A[:,n_i_ctr]*bnew)
            else:
                if j%len(powersoftwo)==0 and j>0:
                    n_j_ctr = n_j_ctr + 1
                    j_powerctr = 0
                Qdict[i,j] = powersoftwo[i_powerctr]*powersoftwo[j_powerctr]*Qinit[n_i_ctr,n_j_ctr]
            
            j_powerctr = j_powerctr + 1
        i_powerctr = i_powerctr + 1
    
    return Qdict

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

#Toy problem for linear system of equations, here m == n
#x will be np.array([3,1])

#Declare A and b
# Linear system to solve
# 2x + y = 7
# x + 3y = 6

A = np.array([[2,1],[1,3]])
b = np.array([7,6])
n = 2

prec_list = [1,0] #bitspower will go from -4 to 3
Q = qubo_prep(A,b,n,prec_list) #prepare qubo

for k,v in Q.items():
    print(k)

#Use Exactsolver to solve the qubo
sampler = dimod.ExactSolver()
sampleset = sampler.sample_qubo(Q)

#Get the solution
soln_dict = sampleset.first.sample
#convert dict to string
soln_bin= get_bin_str(soln_dict,isising=False)
#convert string to np.array x
x = qubo_to_real(soln_bin,n,prec_list)

print("Solution x is : ",x)

#In the cases where Ax = b, ||Ax - b|| = 0 and also ||b||^2 = ground state energy of the qubo
print("||b||^2: ",np.linalg.norm(b)**2)
print("Energy of soln: ",sampleset.first.energy)


