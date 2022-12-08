import numpy as np
import copy
import dimod
import random
import math
from math import log
import dimod
import numpy.linalg as LA 

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
    print(str_bitspower)
    #prepare Qinit
    for i in range(0,n):
        for j in range(i,n):
            Qinit[i,j] = 2*sum(A[:,i]*A[:,j])    
    
    bnew = float(copy.deepcopy(b))
    bnew -= offset_total #we subtract offset here coz the whole bnew is gonna be subtracted from other parts
    bnew = 2*bnew
    
    
    for i in range(0,n*bits_no):
        if i%bits_no==0 and i>0:
            n_i_ctr = n_i_ctr + 1
            i_powerctr=0
        n_j_ctr = n_i_ctr
        j_powerctr = i_powerctr

        for j in range(i,n*bits_no):
            if i==j: #Linear coefficients
                Qdict[i,i] = (scale_list[n_i_ctr]**2)*(powersoftwo[i_powerctr]**2)*(sum(A[:,n_i_ctr]**2)) - scale_list[n_i_ctr]*powersoftwo[i_powerctr]*sum(A[:,n_i_ctr]*bnew)
                if varnames != None:
                    tempvar1 = varnames[n_i_ctr] + '_' + str_bitspower[i_powerctr]
                    index_dict[tempvar1] = i
                    Qdict_alt[tempvar1,tempvar1] = Qdict[i,i]
            else:
                if j%bits_no==0 and j>0:
                    n_j_ctr = n_j_ctr + 1
                    j_powerctr = 0
                
                Qdict[i,j] = scale_list[n_i_ctr]*scale_list[n_j_ctr]*powersoftwo[i_powerctr]*powersoftwo[j_powerctr]*Qinit[n_i_ctr,n_j_ctr]
                if varnames != None:
                    tempvar2 = varnames[n_j_ctr] + '_' + str_bitspower[j_powerctr]
                    Qdict_alt[tempvar1,tempvar2] = Qdict[i,j]
            
            j_powerctr = j_powerctr + 1
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


####


n=2
upper_limit = 100
lower_limit  = -100
bits_no = 3
sampler = dimod.ExactSolver()

s = (upper_limit - lower_limit)/(2**(bits_no) - 1)
scale_list = [s for i in range(0,n)]
offset_list = [lower_limit for i in range(0,n)]


#(6 - x1 - x2)^2
#Inorder to repurpose a code for ||Ax - b|| for individual quadratic equations for n variables,THe Dimensions for A is 1 x n, for x : n x 1 and b : 1 x 1
A = np.array([[1,1]]) 
prec_list = [1,0]
b = np.array([6])
n=2
varnames = ['x1','x2']
Q,Q_alt,index = qubo_prep_adaptive(A,b,n,scale_list,offset_list,bits_no,varnames=varnames)


x_cur = [0 for x in range(0,n)]
x_cur = np.array(x_cur)
itr = 0
while LA.norm(np.matmul(A,x_cur)- b) > 10**-10:
    print("scale_list: ",scale_list," offset_list: ",offset_list)
    Q,Q_alt,index = qubo_prep_adaptive(A,b,n,scale_list,offset_list,bits_no,varnames=varnames)
    #Q = qubo_prep_adaptive(A,b,n,scale_list,offset_list,bits_no)
    sampleset = dimod.ExactSolver().sample_qubo(Q_alt)


    print("A:\n", A)
    print("b:\n", b)
    print("x_cur:\n", x_cur)
    
    #Get the solution in the form of the non-labelled index (compatible with legacy code that way)
    soln_dict = convert_result(sampleset.first.sample,index)
    
    #convert solution into binary string
    binstr = get_bin_str(soln_dict,isising=False)

    binstr_vec = ['' for i in range(0,n)]
    temp_ctr = 0
    
    #from the binstr create a list with entries for each variable in n, Eg for n=2, if binstr = '011100' then binstr_vec = ['011','110']
    for i in range(0,n):
        for j in range(0,bits_no):
            binstr_vec[i]+= binstr[temp_ctr]
            temp_ctr += 1
    print("binstr_vec",binstr_vec)
    
    #convert qubo result to an np.array of floating point values
    x_cur = qubo_to_real_adaptive(binstr,n,scale_list,offset_list,bits_no)
    x_cur = np.array(x_cur)

    print("Iteration: ",itr, " x_cur: ",x_cur, " cur norm: ",LA.norm(np.matmul(A,x_cur)- b))
    
    #Here we re-adjust the scale and offset for each variable
    new_scale_list = []
    new_offset_list = []

    for i in range(0,n):
        temp_scale,temp_offset = get_scale_offset(binstr_vec[i],scale_list[i],offset_list[i],upper_limit,lower_limit,bits_no,2)
        new_scale_list.append(temp_scale)
        new_offset_list.append(temp_offset)
    
    scale_list = new_scale_list
    offset_list = new_offset_list
    itr += 1
    print("--------------------")



