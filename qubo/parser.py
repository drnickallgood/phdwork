import qubo 
import numpy as np

class Parser:
    def __init__(self, v, k):
        self.v = v
        self.k = k
        #self.num_samples = num_samples 
        self.v_dict = {}
        self.x_dict = {}
        self.x_dict_rev = {}
        self.p = 0
        self.n = 0
        self.Q_total = {}
        
    def find_vars(self):

        v_list = list()

        # Store our wh values and in reverse
        #x_dict = {}
        #x_dict_rev = {}

        # store our v values
        # This will essentially be a set of nested dictionaries
        '''
           (x, y) : { 
                       v_val: { v} , 
                       wh: { (wh_ik, wh_kj) }
                    }

        '''
        # Dict for our V position and values in V
        #v_dict = {}

        # 
        wh_dict = {}
     
        # Get correct dimensions
        # V is p x n 
        self.p = self.v.shape[0]
        self.n = self.v.shape[1]

        # W is p x k
        w_rows = self.p
        w_cols = self.k

        # H is k x n
        h_rows = self.k
        h_cols = self.n

        self.w_dict = {}

        # Get W
        for i in range(0, self.p):
            for j in range(0, self.k):
                temp_w = "w" + str(i+1) + str(j+1)
                self.w_dict[i,j] = temp_w 

        #Get the V's
        # V is p x n 
        for i in range(0,self.p):
            for j in range(0, self.n):
                # stringify what is in V at this location
                i_idx = i+1
                j_idx = j+1
                self.v_dict[i_idx,j_idx] = {}
                self.v_dict[i_idx,j_idx]['v_val'] = str(self.v[i][j])
                self.v_dict[i_idx,j_idx]['wh'] = []
                #v_str = str(v[i][j]) + "-"
                #v_list.append(v_str)

        # Build WH
        # WH will be same as V , so p x n
        wh_cnt = 1
        for i in range(0,w_rows):
            for j in range(0,h_cols):
                # This is just indexing to make it match and not index from 0
                i_idx = i+1
                j_idx = j+1
                for l in range(0,w_cols):   # This is the column vector selection
                    #print("w" + str(i+1) + str(l+1) + "h" + str(l+1) + str(j+1))
                    #x_dict['x'+str(wh_cnt)] = ("w" + str(i+1) + str(l+1) + "h" + str(l+1) + str(j+1))
                    self.x_dict['x'+str(wh_cnt)] = ("w" + str(i+1) + str(l+1), "h" + str(l+1) + str(j+1))
                    self.x_dict_rev[("w" + str(i+1) + str(l+1), "h" + str(l+1) + str(j+1))] = 'x'+str(wh_cnt)
                    self.v_dict[i_idx,j_idx]['wh'].append( ("w"+str(i+1) + str(l+1), "h"+str(l+1)+str(j+1)) )
                    wh_cnt += 1
                   # x_dict['x'+str(i)] = "w" + str(i+1) + str(l+1) + "h" + str(l+1) + str(j+1)

    def parse_vdict(self):
        #print("Parsing V dictionary...\n")
        # Go through main dictionary to get data
        for key, val in self.v_dict.items():
           #print(v_dict[key]['wh'])
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
            A = np.zeros([1,self.k])
            A += 1
            b = float(self.v_dict[key]['v_val'])
            Q, Q_alt, index = qubo.QuboA.qubo_prep(A,b,self.k,prec_list,varnames=varnames)
            # Put everything from each Q_alt dict into master Q_total
            for key, val in Q_alt.items():
                # Check if key is already here, if so add to it
                if key in self.Q_total:
                    self.Q_total[key] += val
                else:
                    self.Q_total[key] = val
            
            
    def get_vars(self):
        self.find_vars()
        #self.parse_vdict()
        return self.v_dict, self.x_dict, self.x_dict_rev, self.p, self.n, self.w_dict 
        
    def get_qtotal(self):
        return self.Q_total
        
        
        
        
