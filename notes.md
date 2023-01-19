* Research Notes *

** 1/18/2023 **

*** Section 4.1 need to fix TABU graphs, doesn't have samples but time and x-axis is wrong

*** Section 4 need to fix quantum annealing graphs, make them like the rest

*** Section 4 need to fix hybrid BQM graphs, make them like the rest 

*** Merge above charts into one like i've done previously...

*** Need to review any changes suggested by Dr N and others

*** Once that's done, send to committee via email






** 1/5/2023 **

*** Adaptive k-means analysis completed
*** Had to re-do the graphs do to them being wrong
*** Need to start on conclusion (Read R's thesis conclusion)
*** need to start on presentation

** 1/4/2023 **

*** Tables for adaptive motif kmeans added to dissertation
*** Need to add graphics for daptive motif kmeans
*** Need to do analysis for adaptive motif kmeans
*** Need to start on conclusion...
*** Need to work/start on presentation


** 1/3/2023 **

*** Completed Adaptive MOTIF experiements for both clustering and kmeans
*** Need to make tables for dissertation
*** Need to make graphs for dissertation
*** Need to do analysis for dissertation

** 1/2/2023 **

*** Gaussian Adaptive and Kmeans dissertation documented in dissertation and graphics uploaded
*** Adaptive Analysis and 
*** Need to do MOTIF data for adaptive and adaptive kmeans
*** Need to do MOTIF Data graphs for adaptive and adaptive kmeans
*** Need to do MOTIF analysis for adaptive and adaptive k-means
*** Need to do conclusion


** 12/31/2022 **

*** Gaussian Adaptive Exp Completed
*** Gaussain Adaptive kmeans exp completed
*** Need to make graphics/charts for adaptive gaussian exp and kmeans exp
*** Need to add data and graphics to Dissertation doc



** 12/30/2022 **

*** Working on adaptive formulation/experiements on gaussian data now. Tabu completed, working with sim annealing.
*** Might want to check out L1 norm to see what it is, and how it compares


** 12/28/2022 **

*** Motif results for original formulation completed and kmeans results completed, added to dissertation 
*** Will start working on Gaussian data for 20 - 50 samples, and then do 100, 1000, 10000 samples using 3 processes


** 12/19/2022 **

*** Created Motif results in mathematica notebook with graphs, all that data is ready to be uploaded to Ch 6 of dissertation
*** Need to Update Dissertation with data from motif default form
*** Need to start running experiments with adaptive form, 20, 30, 35, 40, 45, 50 samples
*** Need to do K-means runs with adaptive Results
*** Need to do adaptive experiements on MOTIF data
*** Need to do kmeansr uns on adaptive results for MOTIF data


** 12/18/2022 **

*** Issue seems mostly resolved, in that if i increase delta's, then H behaves correctly, but deltas seem to be much higher
than the previous formulation.

*** We now have the following parameters to work with:
  **** num_reads, num_sweeps, timeout, etc for the solver  (This is sorta new in the way we use it in conjunction with iterations)
  **** delta1, delta2 for the lagrange params
  **** NEW: Upper and lower limits for scale values to zoom in
  **** NEW: w_itr or number of iterations , seems sometimes when this is too high we get a super. In papers it's based on a tolerance
  variable which can be determined froma norm but we don't always converge on that norm.. so it's not a good idea to use this.

*** DONE -- Still need to finish up MOTIF for 35 40 45 and 50 samples for sim annealing
*** DONE -- Still need to finish up MOTIF for few samples for Hybrid
*** DONE -- Need to do K-means for Motif for Tabu, sim annealing, Hybrid
*** Need to go and update document too, put in fixes and suggestions.
*** Need to make notebook for graphs of kmeans results and such for motif



** 12/17/2022 ** 

*** So i managed to get the floating point / adaptive stuff to work, ended up also converting dictionary to w11 style
*** Had to adjust the scale value from -8 to +7 , which seems to help quite a bit instead of -100 to +100
*** Still the issue of H having violated columns, perhaps the deltas are not enough? 
*** Need to continue with rest of code and get actual cluster analysis...
*** seems to be an issue with how H is built, not only are the columns violated a lot, but
it seems that the length gets doubled per my sample size. -- need to compare while running mmain and the new stuff
  **** Need to check penalization for linearization and H, make sure it's happening, this might be why it's not working
     - Confirmed this, when the computed label size doesn't match or is greater than the blob labels, then we have violations
     - this causes issues 

*** Not sure why I didn't keep dating my logs before, but here we go in starting that.
*** Managed to get a W entry paried with a scale and offset list, getting adjusted per iteration
  **** To adjust on W's the keys are the same so this part should be easier now.
  **** In scale list for each W, value is same, so we can pick the first
  **** In offset list, we have 3 values, that correspond, so which do we pick?


** Questions for AJ
  - Issue with H and getting just 1 in a single column, and the fact when it did happen it was only in bottom
  - How to map offsets to specific W values
      -- We can ignore 'h' values in ('w11', 'h11')
      -- In build qubo, we need to build a scale list and offset list for W
      -- We then go through when building W to apply scale and offsets 
      So we basically are adding a scale list and offset list in a W-dict as the value of a w key
      -- we then will go through this and basically do the iterative pieece for the scale and offset lists in the speicifc
      w stuff..


      the w_scale and w_offset will
what you send into qubo_adaptive_prep will be a subset of those matching the correct xs
I don't think addition will work for this because what we are doing is zooming into the range of w11. (Ie making its value more precise)

Also don't think we are using offset (and scales) of x to figure out ws , think of it like we are assigning offsets of ws as offsets of x , getting back all values and adjusting the offsets of ws . Rinse and repeat

  - What about b values, limited here, not enough to go through all xvars for the norm and figure out iteration


  ** How do we get proper scale value to W? 
   ** seems if we go through we're alreayd kinda doing this by checking for specific W value in solution_dictionary
   ** seems we have multiple entries for w and when we have them, we simply add it to our entry
      ** because we're doing another loop over the items, and adding to the existing W.. so that's why this works
      ** Just need to get the w_ij format to the tuple.. that maps to offset
      ** Could further see if tempw is in the tuple of self.offset_xvar_dict..
          print(list(self.offset_xvar_dict.keys())[0][0])
          print(list(self.offset_xvar_dict.values())[0])        
      ** Need to go through the above, and basically create a dictionary that:
         ** uses the first as the key and the value is another dictionary 
            # Loop through xvar dict, see if tempw is in part of it, if so get value

  ** Also issue seems to be H has more than a single 1 in a column, OR in the past, it would work but all botom
  rows of H had a 1 and that's it, currently in all examples it violates H rules



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
    ** WH # of elements are same size as B



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