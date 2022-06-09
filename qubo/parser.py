

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

    def get_vars(self):
        self.find_vars()
        return self.v_dict, self.x_dict, self.x_dict_rev, self.p, self.n
        
        
