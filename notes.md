** Research Notes **

  ** Things seem to be working as expected, or better than thay were after breaking adaptive code out to 
  its own method
    ** Seems to stop after 10 iterations, or ther size of b_list.. this doesn't match
    ** scale_list is the length of # of x variables...
    ** offset_list is also length of # of x variables
    ** b is the num of b values...
    ** in adaptive.py , A or b don't change, but x_cur does...
    ** attempting to work around it by using just a single b value..
      ** The norm reduces but after a certain point it's very minmal
        ** for iterations, perhaps keep a delta between what changed? And if it's >= 1 iterate.. 
        

  ** Need to update the WH stuff next, basically i have to go through each entry in WH and subtract from teh computed offset
    ** I think i need to make another dictionary where the x_var is the key and the offset is the value
    ** Then i can correlate to the x_var to the respective w and h values

  ** The L2 norm does seem to decrease over time, it starts and goes up then gradually goes down
    ** using TABU only, the timeout period affects this, the longer the period , the larger the jump in norm.
    ** the other tweak is to only have a bit of the process happening, but lots more iterations



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



*** Old Notes ***

Seems to be related to having larger than bits_no variables for n.. 

## issue seems to be with varnames only being of size 3.. being fed in wrong?
## When building q_total we're only maknig 3 varnames and sending it in, with new x variable list we're sending in full variable for qubo_prep adaptive... so ti's causing indexing issues.
## Did more testing and it looks like we get the same # of x vals in the first 2 loops of v_dict.. which is expected
## perhaps we either need to loop through and do qubo_prep first , then do adaptive on the result, then add that to Q_Total?
## perhaps we need to just have the x vals that correspond from x_dict.. 
## POr maybe we just need to use self.k for qubo_prep_adaptive part only and use the x_num for later things? - NO
** using self.k i think works as a coincidence here, but really since we're iterating through it should be a certain length
   that isn't tied to k but equal in value, perhaps len(varnames)
** Then W and H will use the num_x in the correct offset lists.. 
** move scale list to build_qtotal... 


number of x variables (before discretization) is the length of those lists

  ** qubo_to_real_adaptive isn't necessary for this afterall..
  ** W modification:
      but a little correction, you need to do the following for every entry in W
    final_W_entry = scale * temp_W_entry + offset
    (temp_W_entry is what you have after doing the get_W_h procedure currently)
    ** scale_list all entries have same value
    ** offset_list all entries have different values
    ** Question now is how do we correspond offset_values to specific W entries..
      ** AJ notes **
        both offset_list and scale_list are meant for keeping offset and scale information for each x variable (in my original code). So you would need to know which x corresponds to which W and transitively which position in scale_list and offset_list does it correspond with.

its based on the number of x variables, I forget how many it relates to wrt W or its constituent dimensions

  ** binstr issues for scaling reduction ...

  ** first is related to with iterations and larger samples (such as 10), we may get a binstr like this:
  binstr: 000000000000000000001111111111111111111111111111001001111001001111001001111001001001111010011111001001111010011001111001001111101001111001001111111001001111001001111001001111001100001111001001111010011111001001111111111001001111
  ** This has a lot of 0's in front, and when we make our binstr vec we seem to constantly get the first 3 of 3 digits, so ['000', '000', '000']
  ** Possibility we may not need to do this, double checking...
  ** We might be able to just use the offset_list and subtract from W values to get center points. 
  ** Will need to find a better iteration form to iterate from


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