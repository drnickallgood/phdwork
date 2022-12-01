Research Notes

** Current Status **
 * Issue with indexing with varnames, took a few days but figured out where the culprit seems to be:, seems to be n_j_ctr which is then referenced by varnames[n_j_ctr]

 j_pwr_ctr 2
varnames[n_j_ctr x3
str_bitspower[j_powerctr] 0
J:  9
n_j_ctr 3
j_pwr_ctr 0



** TO DO **

* Add NMF basics to Chapter 2 dissertation.
* Write math out for NMF solution
* Write math out for Adaptive Formulation using offsets 

* Adaptive solution which allows the use of floating point in the results
    * Tabu
    * Sim Annealing
    * Hybrid BQM


* Real life data (Malware MOTIF data set)
    * \# of samples out of specific malware famalies, cluster them
    * See how close we get, MOTIF has ground truth labels or close to it...


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