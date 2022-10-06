Research Notes

** Current Status **

* Uploaded Graphs / k-means comparisons to dissertation doc


** TO DO **

* Expand Tabu Charts for 20, 30, 35, 40, 45, 50, 100 samples , get graphs

* Tabu Kmeans comparison (DONE) - Do writeup comparing to classical 

* Diff Seed value + 5 cluters + 45 samples
    * Tabu 30 min
    * Sim Annealing
    * Hybrid BQM  (longer time_limit) 
        * See if we can return time_limit from existing exp
        * sample_qubo(Q, time_limit=6)   # 6 seconds
        * sampler.properties
        * sampler.parameters


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