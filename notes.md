Research Notes

** Current Status **
 * IIssue resolved with indexing, needed to use 'k' instead of 'n' for everything outside of the adaptive-modified.py file.. 

 ** Current issue now is with k = 3, this seems to give us 3 values, when we really have 2 values
 per cluster, so for 3 clusters we would need 6 values
 


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