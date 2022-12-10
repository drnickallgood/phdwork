Research Notes

** Make main2.py and qubo2.py , update imports, and convert to class/objects
  ** Try to use them, but instead of qubo_prep use qubo_prep_adaptive
    ** Prec_list for adaptive won't begin with null
    ** prec_list_str set to []
    ** make sure eto not use convert_result, this was something setup for one problem 
    ** When we get WH back first, we get integers, but then we use convert_real to make it real
      ** based on scale list


--- Ajinkya Notes below --

See section B of this paper should give you an idea:
https://arxiv.org/abs/2106.04305

Other papers
First paper that suggested an iterative approach:
https://www.nature.com/articles/s41598-019-46729-0
(associated github repo)
https://github.com/cchang5/quantum_poly_solver/blob/master/DWSolveQUBO.py

THis other work also does things iteratively
https://arxiv.org/abs/2005.02846


0. take an initial range and scale
1. build q_total, submit,
2.  get results, if results in acceptable threshold of error OR certain no of counts are over, BREAK LOOP
3. interpret new range and scale from results, GOTO step 1 

(6 - x1 - x2)

x1 = 2, 1, 0 binary
x2 = 2, 1, 0  binary

Fit this into Ax = b

# Asize = # variables 
A = (1, 1)(x1)   = 6
          (x2)  


x1 + x2 = 6
(6 - x1 -x2)^2 = 0    // Minimize

x1 x2 are binary variables! 1 or 0


scale_value * (2^2 * x1_2 + 2x1_1 + 2x1_0^0 ) + offset 
Range of variables is 2^0 - 2^2 = 7 variables

7  6  5  4  3  2  1  offset  (when 0)

Space between numbers above , controled , scaled by lambda

  scales / divides range of numbers, uses 3 binary variable for 1 real variable

  (6 - x1 - x2)^2  -- any 2 numbers that can make a -6 will be a right answer
   in thix example n = # variables
     x1 and x2 between upper and lower limnits
     bits_no = how many bits do we discretize  (power 0 power 1 power 2)
     s = scaling value (scale_value) (based on range)

     x_cur = temporary vector 

     Iteration, ||Ax -b|| norm used as iterative (LA.norm(..))
       closer it gets to 0, most ideal

     Fixed number of iterations for now... variable should govern max iteration

  resulting values = [x1, x2] -- returned in iterations 





** TO DO **

* Add NMF basics to Chapter 2 dissertation.
* Write math out for NMF solution
* Write math out for Adaptive Formulation using offsets 

* Adaptive solution which allows the use of floating point in the results
  3 cluster, seed=0

    * Tabu (30 min only)
    * Sim Annealing
    * Hybrid BQM

* K-Means test using Centroids, default kmeans as control 
    seed = 0
      Tabu
      Sim
      Hybrid BQM


* Real life data (Malware MOTIF data set)
    * \# of samples out of specific malware famalies, cluster them
    * See how close we get, MOTIF has ground truth labels or close to it...
      * Tabu
      * Sim
      * Hybrid

  * Real life data (Malware MOTIF data set) - Kmeans 
    * \# of samples out of specific malware famalies, cluster them
    * See how close we get, MOTIF has ground truth labels or close to it...
      * Tabu
      * Sim
      * Hybrid




* Expand Tabu Charts for 20, 30, 35, 40, 45, 50, 100 samples , get graphs


** Tentative Schedule **

* Experiments done:  Dec 31, 2022
* Dissertation Draft:  March 15, 2023
* Dissertation Defense: April 15, 2023


** Prospective Papers **

* Quantum NMF and clustering with it
* Adaptive Quantum NMF
* Quantum clustering of malware data
* Quantum clustering of DNA data