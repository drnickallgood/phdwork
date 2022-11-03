Research Notes

** Include STDEV and seed for gaussian blobs in dissertation **

** Current Status **
 * sim annealing deltas 29 / 167
 Delta1:  29
Delta2:  167



Computed Centers
[-2.0, -2.0]
[-2.0, 14.0]
[-6.0, 12.0]
[-2.0, 14.0]
[14.0, -2.0]

Running time:  4:21:16.876852


delta1  - 29
delta2 - 165 , gets close too:

[-2.0, -2.0]
[-2.0, 14.0]
[-2.0, 12.0]
[-2.0, -2.0]
[14.0, -2.0]

delta1 29, delta2 164 gets worse...

delta1 29, delta2 166 slightly better, but still fail



* Waiting on sim annealing for  50 samples for 5 clusters, seed 1725. 
* Adaptive formulation notes:
    offset_vector = np.array(offset_list)
    offset_total = np.matmul(A,offset_vector)
    bnew = (float)(b)
    bnew = bnew - offset_total   # This is a critical piece, we keep minimizing our offset
    bnew = 2*bnew

    # 1
    n=2

    upper_limit = 100
    lower_limit  = -100
    bits_no = 3
    sampler = dimod.ExactSolver()

    s = (upper_limit - lower_limit)/(2**(bits_no) - 1)
    scale_list = [s for i in range(0,n)]
    offset_list = [lower_limit for i in range(0,n)]

    # 2
    #(6 - x1 - x2)^2
    #Inorder to repurpose a code for ||Ax - b|| for individual quadratic equations #for n variables,THe Dimensions for A is 1 x n, for x : n x 1 and b : 1 x 1
    A = np.array([[1,1]]) 
    prec_list = [1,0]
    b = np.array([6])
    n=2
    varnames = ['x1','x2']
    Q,Q_alt,index = qubo_prep_adaptive(A,b,n,scale_list,offset_list,bits_no,varnames=varnames)


** TO DO **

* Kmeans exp 5 clusters, 1725, for sim annealing , document results
* add 5 cluster, 1725 seed data to charts in dissertation
* Add comparison graphs:
* Inertia comparison for all 3 processes
* Silhouette, Homog, Compl, V-measure for all 3 processes


* Expand Tabu Charts for 20, 30, 35, 40, 45, 50, 100 samples , get graphs



* Adaptive solution which allows the use of floating point in the results
    * Tabu
    * Sim Annealing
    * Hybrid BQM


* Real life data (Malware MOTIF data set)
    * \# of samples out of specific malware famalies, cluster them
    * See how close we get, MOTIF has ground truth labels or close to it...



** Tentative Schedule **

* Experiments done:  Dec 31, 2022
* Dissertation Draft:  March 15, 2023
* Dissertation Defense: April 15, 2023



** Prospective Papers **

* Quantum NMF and clustering with it
* Adaptive Quantum NMF
* Quantum clustering of malware data
* Quantum clustering of DNA data