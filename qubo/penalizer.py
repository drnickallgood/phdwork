import numpy as np 

class Penalizer:
    def __init__(self, x_dict, delta1, delta2, Q_total, prec_list_str):
        self.x_dict = x_dict
        self.delta1 = delta1
        self.delta2 = delta2
        self.Q_total = Q_total
        self.prec_list_str = prec_list_str
        #self.Q_alt2 = {}

        
    def linearization(self):
       # print("Applying linearization penalties...\n")
        # linearization
        # Delta == lagaragne param
        #delta = 50
        for x_key, x_val in self.x_dict.items():
            temp_h = x_val[1]
            #print(temp_h)
            for prec in self.prec_list_str:
                temp_x = x_key + "_" + prec
                #print(test)
                temp_w = x_val[0] + "_" + prec
                #print(temp_w)
                self.Q_total[temp_w, temp_h] = 2 * self.delta1
                self.Q_total[temp_x, temp_w] = -4 * self.delta1
                self.Q_total[temp_x, temp_h] = -4 * self.delta1
                self.Q_total[temp_x, temp_x] += 6 * self.delta1
        
    def h_penalty(self, Q2_alt):
        #print("Applying H penalties...\n")
        Q_alt2 = {} # new dict for Q_alt but diff key names

        for key,value in Q2_alt.items():
            #Erase all the characters in the key after underscore
            temp_key = (key[0].split('_')[0],key[1].split('_')[0])
            Q_alt2[temp_key] = value * self.delta2

        #print("H Penalties: \n")
        #pprint.pprint(Q_alt2)

        #print("Adding all data to Q_total...\n")
        # Add all to Q_total for H
        for key, val in Q_alt2.items():
            if key in self.Q_total:
                self.Q_total[key] += val
            else:
                self.Q_total[key] = val
                
        
    def get_penalized_qtotal(self):
        #self.linearization()
        #self.h_penalty()
        return self.Q_total
        
        
        
        
        
        
        
        
