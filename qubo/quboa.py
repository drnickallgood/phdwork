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
from .penalizer import Penalizer
from sklearn import metrics

class QuboA:
    #def __init__(self, v, v_dict, x_dict, x_dict_rev, prec_list, k, p, n, delta1, delta2,upper_limit, lower_limit, offset_list, scale_list, bits_no, num_sweeps, num_reads, tabu_timeout, solver, num_x):
    def __init__(self, v, v_dict, x_dict, x_dict_rev, prec_list, k, p, n, delta1, delta2,upper_limit, lower_limit, bits_no, num_sweeps, num_reads, tabu_timeout, solver):
        #self.dataset = dataset
        #self.num_centers = num_centers
        #self.num_samples = num_samples
        self.v_dict = v_dict
        self.x_dict = x_dict
        self.x_dict_rev = x_dict_rev
        self.prec_list = prec_list 
        self.Q_total = {}
        self.k = k
        self.p = p
        self.n = n
        #prec_list = [2, 1, 0]   #-8 to +7
        #self.prec_list_str = ['null']
        self.prec_list_str = []
        # Get string versions of prec_list_stirngs
        prec_strings = [self.prec_list_str.append(str(x)) for x in self.prec_list]
        self.delta1 = delta1
        self.delta2 = delta2
        self.v = v
        self.W = np.zeros([self.p, self.k])
        self.H = np.zeros([self.k, self.n])
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        #self.offset_list = offset_list 
        self.offset_list = []
        #self.scale_list = scale_list
        self.scale_list = [] 
        self.bits_no = bits_no 
        self.num_sweeps = num_sweeps
        self.num_reads = num_reads
        self.tabu_timeout = tabu_timeout
        self.solver = solver 
        self.solution_dict = {}
        #self.num_x = num_x 


        self.build_qtotal()
    
    #This is just to convert a dictionary based result into a binary string based result
    # first version relies on numerical keys 
    # The idea is that each result will change this from the solver...
    def get_bin_str(self, config,isising=True):
        #Input:
        #config is a dictionary
        #isising is True if config has -1 or +1 and False if config has 0 or 1
        #Output:
        # a binary string of 0s and 1s
        binstr = ""

        '''
        print("value:", end= " ")
        for k,v in config.items():
            print(v, end = " ")
        '''
        
        # So this goes through each key and item in binstr
        for k,v in config.items():
            #print(k, ":", v)
            if isising == True:
                if v == 1:
                    binstr += str(1)
                elif v == -1:
                    binstr += str(0)
            else:
                if v == 1:
                    binstr += str(1)
                elif v == 0:
                    binstr += str(0)
            #print("binstr so far:", binstr)
        return binstr

    #processes the binary string into a np.array vector.
    def qubo_to_real(self, binstr,n,prec_list):
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
    def qubo_prep(self,A,b,n,bitspower,varnames=None):
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
        
    def qubo_prep_nonneg(self, A,b,n,bitspower, varnames=None):
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
        
    def convert_result(self, soln_dict,index):
        new_dict = {}
        for key,value in soln_dict.items():
            new_dict[index[key]] = value
        return new_dict
    
    # Annealer will return binary value, need to convert back to reals
    # using approximation
    # Convert real value by taking floor of float value then make binary
    def approx_real(self,r):
        return bin(math.floor(r).split('0b')[1])

    def bin_to_real(self,binstr):
        return int(binstr,2)

    def build_qtotal(self):

        print("Building Q_total...")

        scale_temp = []
        offset_temp = []
        b_list = []

        for key, val in self.v_dict.items():

            # Go through individual list of tuples
            varnames = []
            for item in self.v_dict[key]['wh']:
                
                # Get our corresponding x values to WH values
                varnames.append(self.x_dict_rev[item])
                # Build a row vector of 1's for A
                #A = np.zeros([1,k])
                #A += 1
                # Get each element of V for qubo_prep
                # print(varnames)
                # Also store them as a floating point number vs a string
                #for v_key, v_val in v_dict.items():
                # Build a row vector of 1's for A
                #A = np.zeros([1,self.k])
                # A for adaptive is size n x 1
                #A = np.zeros([1,self.num_x])
                
            #Build scale list based on # of x variables in current pass
            s2 = (self.upper_limit - self.lower_limit)/(2**(self.bits_no) - 1)
            scale_list2 = [s2 for i in range(0, len(varnames))]
            offset_list2 = [self.lower_limit for i in range(0, len(varnames))]

            # extend main scale list with new items
            # Temp holding spot to keep building up all the stuff we have
            scale_temp.extend(scale_list2)
            offset_temp.extend(offset_list2)

            #A = np.zeros([1,self.k])

            A = np.zeros([1, len(scale_list2)])
            A += 1
            b = np.array(float(self.v_dict[key]['v_val']))
            #print(b)
            b_list.append(b)

            #Q, Q_alt, index = self.qubo_prep(A,b,self.k,self.prec_list,varnames=varnames)
            ## SHould be not self.k but something else , perhaps len(varnames)
            #Q, Q_alt, index = self.qubo_prep_adaptive(A, b, self.k, self.scale_list, self.offset_list, self.bits_no, varnames=varnames)
            Q, Q_alt, index = self.qubo_prep_adaptive(A, b, len(scale_list2), scale_list2, offset_list2, self.bits_no, varnames=varnames)

            #Q, Q_alt, index = self.qubo_prep_adaptive(A,b,self.num_x, self.scale_list, self.offset_list, self.bits_no, varnames=varnames)

            # Put everything from each Q_alt dict into master Q_total
            for key, val in Q_alt.items():
                # Check if key is already here, if so add to it
                if key in self.Q_total:
                    self.Q_total[key] += val
                else:
                    self.Q_total[key] = val

        ## QTOTAL IS BUILT AT THIS POINT !! ## 
        # Q_total is done, lets get the correct scale and offset list per Q_total to use later
        self.scale_list = scale_temp
        self.offset_list = offset_temp 

        penal = Penalizer(self.x_dict, self.delta1, self.delta2, self.Q_total, self.prec_list_str)
        print("Applying linearization penalties...")
        penal.linearization()
                    
        # Make as many q_alt2s as there are columns in H
        prec_list2 = [0] #all variables are binary, DO NOT CHANGE VALUE
        b2 = np.array([1]) # This 1 enforces only one variable to be a 1 :D
        
        print("Applying penalties to H...")
        for h_i in range(0, self.n):        # row
            varnames2 = []
            for h_j in range(0, self.k):    # col
                varnames2.append('h'+str( (h_j+1) ) + str( (h_i+1) ))
                #pprint.pprint(varnames2)

            A2 = np.zeros([1,self.k])
            #A = np.zeros([1,self.num_x])
            A2 += 1
            #pprint.pprint(varnames2)    
            Q2, Q2_alt, index2 = self.qubo_prep_nonneg(A2, b2, self.k, prec_list2, varnames=varnames2)
        
            penal.h_penalty(Q2_alt)

        # Q_total is built, now we can run it through the adaptive piece
        self.adaptive_form(b_list)

        
    def adaptive_form(self, b_list):
        
        w_itr = 0
        b_ctr = 0
        iter_delta = 0

        self.offset_xvar_dict = {}

        while(w_itr < 3):

            # temp scale lists
            new_scale_list = []
            new_offset_list = []

            print("iteration: ", w_itr)

            # submit qubo, data is tsored in self.solution_dict
            self.qubo_submit()

            # get binary string from solution dictionary, from quantum solver
            binstr = self.get_bin_str(self.solution_dict, isising=False)
        
            # Put binstr into vector, range is 0 to num of x_variables
            binstr_vec = ['' for i in range(0, len(self.scale_list)) ]
            temp_ctr = 0

            for i in range(0, len(self.scale_list)):
                for j in range(0,self.bits_no):
                    binstr_vec[i]+= binstr[temp_ctr]
                    temp_ctr += 1

            x_cur = self.qubo_to_real_adaptive(binstr, len(self.scale_list), self.scale_list, self.offset_list, self.bits_no)
            x_cur = np.array(x_cur)

            #print("x_cur: ", x_cur)


            #This is intended to be used for scaling/iterations
            #Problem is we need ot ensure it subtracts from correct b value
            #possibly need to store all b values.. and go that way

            A = np.zeros([1, len(self.scale_list)])
            A += 1
            
            b = b_list[0]
            print("curr norm:", LA.norm(np.matmul(A,x_cur)-b), "\n", x_cur, "\n")

            # This adjust the scale list 
            for i in range(0, len(self.scale_list)):
                temp_scale,temp_offset = self.get_scale_offset(binstr_vec[i], self.scale_list[i], self.offset_list[i],
                self.upper_limit, self.lower_limit, self.bits_no, 2)
                new_scale_list.append(temp_scale)
                new_offset_list.append(temp_offset)

            self.scale_list = new_scale_list
            self.offset_list = new_offset_list         

            w_itr += 1
      
            #print("scale_list:", self.scale_list, "\n")
            #print("offset_list:", self.offset_list, "\n")  

        # End while loop
        off_ctr = 0
        for k,v in self.x_dict_rev.items():
            self.offset_xvar_dict[v] = self.offset_list[off_ctr]
            off_ctr += 1

    # End method 



    def qubo_submit(self):
        if self.solver == "exact":
            print("Submitting to Exact Solver...")
            sampler = dimod.ExactSolver()
            sampleset = sampler.sample_qubo(self.Q_total)
        elif self.solver == "hybrid":
            print("Submitted to Hybrid Solver...")
            sampler = LeapHybridSampler(solver={'category': 'hybrid'})
            sampleset = sampler.sample_qubo(self.Q_total)
        elif self.solver == "tabu":
            print("Submitted to TABU Solver...")
            sampler = tabu.TabuSampler()
            sampleset = sampler.sample_qubo(self.Q_total, timeout=self.tabu_timeout)
        elif self.solver == "sim":
            print("Submitted to Simulated Annealer...")
            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample_qubo(self.Q_total, num_sweeps=self.num_sweeps, num_reads=self.num_reads)
        else:
            print("Invalid options for qubo submission")
            exit(1)
            
              
        # 2000q Sampler
        #sampler = EmbeddingComposite(DWaveSampler(solver='DW_2000Q_6'))
        # Advantage5.1 Pegasus
        #sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))
        # Hybrid Solver BQM
        self.sampleset = sampleset 
        self.solution_dict = sampleset.first.sample
        
    def get_solution_dict(self):
        return self.solution_dict

    def qubo_verify_adaptive(self):

        print("Creating verification W and H...\n")
        for i in range(0,self.k):
            for j in range(0,self.n):
                temp_h = "h" + str(i+1) + str(j+1)
                self.H[i,j] = self.solution_dict[temp_h]
                
        # For W we have to basically subtract the scale list from W

        for i in range(0,self.p):
            for j in range(0,self.k):
                temp_w = "w" + str(i+1) + str(j+1)
                for sol_key, sol_val in self.solution_dict.items():
                    #print("sol_key:", sol_key)
                    #print("sol_val:", sol_val)
                    if temp_w in sol_key:   
                        #print(list(self.offset_xvar_dict.keys())[list(self.offset_xvar_dict.values())]) 
                        # Loop through xvar dict, see if tempw is in part of it, if so get value
                        #print(list(self.offset_xvar_dict.keys())[0][0])
                        #print(list(self.offset_xvar_dict.values())[0])                    
                        #exit(1)
                 
                       # This is the part we need to do stuff with the scale list
                        temp_str = sol_key.split('_')[1]
                        temp_w_entry = (2**int(temp_str))*sol_val
                        temp_w_entry = self.scale_list[i] * temp_w_entry + self.offset_list[i]
                        self.W[i,j] += temp_w_entry 

        
        self.count_ones()
        print("")

    def qubo_verify(self):

        #self.W = np.zeros([self.p, self.k])
        #W = np.transpose(W)
        #self.H = np.zeros([self.k, self.n])
        #H = np.transpose(H)

        print("Creating verification W and H...\n")
        for i in range(0,self.k):
            for j in range(0,self.n):
                temp_h = "h" + str(i+1) + str(j+1)
                self.H[i,j] = self.solution_dict[temp_h]
                
        # For W we have to basically subtract the scale list from W
        for i in range(0,self.p):
            for j in range(0,self.k):
                temp_w = "w" + str(i+1) + str(j+1)
                for sol_key, sol_val in self.solution_dict.items():
                    if temp_w in sol_key:
                        #print(temp_w, sol_key)
                        temp_str = sol_key.split('_')[1]
                        #print(temp_str)
                        ## This is old not needed for adaptive
                        if temp_str == "null":
                            self.W[i,j] += -(2**(self.prec_list[0]+1))*sol_val
                        else:
                        #but a little correction, you need to do the following for every entry in W
                        #final_W_entry = scale * temp_W_entry + offset
                        # (temp_W_entry is what you have after doing the get_W_h procedure currently)
                            temp_w_entry = (2**int(temp_str))*sol_val
                            temp_w_entry = self.scale_list[i] * temp_w_entry - self.offset_list[i]
                            self.W[i,j] += temp_w_entry 
                            #self.W[i,j] += scale * temp_w_entry + offset 
                            #self.W[i,j] += (2**int(temp_str))*sol_val
                            #final_W_entry = scale * temp_W_entry + offset
        
        self.count_ones()
        print("")

                         
    def get_lagrange_params(self):
        return self.delta1, self.delta2
        
        
    def count_ones(self):
        bad_cols = 0
        for col_num in range(0, self.H.shape[1]):
            col_count = 0
            for row_num in range(0, self.H.shape[0]):
                #print("Row: ", row_num, "Col: ", col_num)
                if self.H[row_num][col_num] == 1:
                    col_count += 1
                
            if col_count > 1 or col_count == 0:
                #print("Bad solution, multiple or no 1's in column: ", col_num)
                bad_cols += 1
                
                    
            #print("Col: ", col_num, "1count: ", col_count)
        print("Total Violated Columns: ", bad_cols)
                            
        return bad_cols
        
    def cluster_analysis(self):
        '''
        This will do our full cluster analysis
        We need Inertia, Silohouette Scores, Homogeneity, Completeness, V-measure
        Scikit has all of this stuff, but we can calculate it

        Inertia = Sum of Squared errors = L2_Norm(V - WH)^2)

        :param V - Original data matrix
        :param W: - calculated W matrix
        :param H:  - Calculated H matrix
        :return:
        '''
        # We need to do some label computations here

        ## Need original blob labels for comparison


        # Inertia
        self.inertia = LA.norm(self.v - np.matmul(self.W, self.H)) ** 2

        # Silhouette


        # Homogeneity


        # Completeness


        # V-Measure


        return True


    def get_results(self):
        #print("\n--- Sampleset ---\n")
        #print(sampleset)
        #self.qubo_verify()
        print("\n--- Verification ---\n")
        delta1, delta2 = self.get_lagrange_params()
        print("delta1: ", delta1, "\ndelta2: ", delta2)
        print("Num Clusters: ", self.k, "\n")


        #print("V (transposed) = \n", v, "\n")
       # print("V Shape (transposed): " , v.shape)
        #print("\nComputed W = \n", self.W, "\n")
        #print("W Shape: ", W.shape)
        #print("\nComputed H = \n", self.H, "\n")
        #print("H Shape: ", H.shape)
        #print("\nComputed WH = \n ", np.matmul(self.W, self.H))
        #print("WH Shape: ", np.matmul(W,H).shape, "\n")
        #print("\nFirst energy: ", self.sampleset.first.energy)
        print("\nFirst energy: ", self.sampleset.first.energy)

        #print("Norm: ", LA.norm(self.v - np.matmul(self.W, self.H)))
        #print("Inertia: ", LA.norm(self.v - np.matmul(self.W, self.H))**2)
        #print("Verifying best energy via Frobenius Norm: ", LA.norm(self.v, 'fro')**2, "\n")

        #print("Number of samples: ", v.shape[1])
        #print("Running time: ", datetime.now()-start_time, "\n")
        #print("")
        #self.count_ones()
        #print("")
        #print("Given Centers: ", centers)
        #print("Gaussian Centers: ", blob_centers)
       # print("")
        #print("Solver: ", sampler.solver.name)
        #print(sampleset.info)
            
    def get_quantum_centers(self):
        '''
        Parse columns of W which are our center points
        '''
        
        #self.qubo_verify()
        centers = list()
        
        # Using this method because we ensure qubo_verify only gets called once.
        W, H = self.get_w_h()
       # print("W: ", W)
       # print("H: ", H)

        w_transpose = np.transpose(W)
        
        for row in range(0, w_transpose.shape[0]):
            coords = list()
            for col in range(0, w_transpose.shape[1]):
                coords.append(w_transpose[row][col])
            centers.append(coords)
            
        
        #print(w_transpose.shape[1])
        
        '''
        for i in range(0, self.W.shape[1]):
            #centers.append(self.W[:, i:i+1].tolist())
            print( self.W[:, i:i+1].shape )
                
            #print(self.W[:, i:i+1])
            #for j in self.W[:, i:i+1]:
             #   print(j)
            #centers.append(self.W[:, i:i+1])
            #print(self.W[:, i:i+1])
        '''

        return centers
                    
                
    def get_qtotal(self):
        return self.Q_total
        
    
    '''
        Only use this if you need to check W and H, be careful as it calls
        qubo_verify twice which messes up some math if you pair it with get_results
        so make sure to not use it or if you do comment it out first
    '''
    def get_w_h(self):
        #self.qubo_verify()
        self.qubo_verify_adaptive()
        return self.W, self.H
        
## Adaptive Stuff

    def qubo_to_real_adaptive(self, binstr,n,scale_list,offset_list,bits_no):
        powersoftwo = np.zeros(bits_no)
        
        #powersoftwo, gets us a vector of 1, 2, 4, 8, 16....
        for i in range(0,bits_no):
            powersoftwo[i] = 2**(bits_no - i-1)
            
        bin_ctr=0
        cur_real = np.zeros(n)

        for i in range(0,n):
            for j in range(0,bits_no):
                #print("scale_list[i]:", scale_list[i])
                #print("powersoftwo[j]:", powersoftwo[j])
                #print("binstr[bin_ctr]:", binstr[bin_ctr])

                cur_real[i] += scale_list[i]*powersoftwo[j]*int(binstr[bin_ctr])

                #print("curr_real[i] before offset list:", cur_real[i])
                bin_ctr += 1
            cur_real[i] += offset_list[i]
            #print("curr_real[i] after offset list:", cur_real[i])
        
        return cur_real

    # n for this is number of variables
    # A = 1 x n
    # x = n x 1 
    # b = 1 x 1
    # Example call - self.qubo_prep_adaptive(A,b,self.num_x, self.scale_list, self.offset_list, self.bits_no, varnames=varnames)
    def qubo_prep_adaptive(self, A, b, n, scale_list, offset_list, bits_no, varnames=None):

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

                    if varnames != None:
                        tempvar2 = varnames[n_j_ctr] + '_' + str_bitspower[j_powerctr]
                        Qdict_alt[tempvar1,tempvar2] = Qdict[i,j]


                j_powerctr = j_powerctr + 1

            i_powerctr = i_powerctr + 1


        if varnames != None:
            return Qdict, Qdict_alt, index_dict
        else:
            return Qdict #just return the bare bones if varnames not requested

    def get_scale_offset(self, binstr,scale,offset,upper_limit,lower_limit,bits_no,scale_factor):
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
            
