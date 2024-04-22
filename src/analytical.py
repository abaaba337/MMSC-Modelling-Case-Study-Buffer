import numpy as np
import pandas as pd
import scipy.integrate as spi

class BufferSize() :
    
    
    ## Initialize 
    def __init__(self, size_dist_para, mu, lamb, B):
        
        # size_dist_para : tuple of float arrays, ( p , a ) specifies the parameters of the mixture exponentials for job size, 
        #                  i.e., sum( pi * ai * exp(-ai * x) ), x bytes per job
        #       mu       : float, transmission rate, bytes per sec
        #      lamb      : float, job arrival rate, jobs per sec
        #       B        : float, buffer size, bytes
        
        self.p = np.array(size_dist_para[0])
        self.a = np.array(size_dist_para[1])
        
        if not np.isclose(self.p.sum(), 1.0) :
            raise ValueError("Invalid distibution given: integral not equals to 1.")
        
        self.k = len(self.p)
        self.mu , self.lamb , self.B = mu , lamb , B
        
        # Laplace transform for the complementary cdf of the mixture exponentials, whose input should be a float
        self.LFc = np.vectorize(lambda s : (self.p / (self.a + s)).sum())
        self.x_dist = np.vectorize(lambda x : ( self.p * self.a * np.exp(-self.a * x) ).sum()) # pdf of the given mixture exponentials
        
        self.mu_in = self.LFc(0) * self.lamb  # average data arrival rate, bytes per sec
        self.decay = self.mu_in < self.mu   
        
        self.s = None                         # singularities
        self.residue_coes = None              # coefficients for residues
        
        self.p1  = None    
        self.p0  = None   
        self.av_occupancy , self.occupancyRate = None , None
        self.delta_QoS , self.epsilon_QoS , self.t_QoS = None , None , None
        
    
    # Evaluate coefficeinets for prod([s+aj])
    def multiply_polynomials(self, a):
        result = [1]
        for aj in a :
            result = np.polymul(result, [1,aj])
        return result
    
    
    ## Fit the partially-rejected case    
    def fit_PR(self, display=False):   
       
        # compute the coefficients for the polynomial, whose roots gives the singularities
        coes = self.mu * self.multiply_polynomials(self.a)
        
        for j in range(self.k) :
            a_temp = np.concatenate((self.a[:j], self.a[j+1:]))
            if len(a_temp) == 0 :
                coes[1:] -= self.lamb * self.p[j]
            else:
                coes[1:] -= self.lamb * self.p[j] * self.multiply_polynomials(a_temp)
            
        # compute the singularities
        self.s = np.real(np.roots(coes)) 
        
        # apply the residue theorem
        LFc_s = self.LFc(self.s) 
        numerator_s = self.lamb * np.array([(s + self.a).prod() for s in self.s])
        denominator_s = self.mu * np.array([(s - self.s[self.s!=s]).prod() for s in self.s])
        self.residue_coes =  LFc_s * numerator_s / denominator_s
        
        # comopute the distribution
        g = lambda y :  (self.residue_coes * np.exp(self.s*y)).sum()
        g_int  = spi.quad(g, 0, self.B)[0]
        self.p0 = 1 / ( 1 + g_int )
        self.p1 = np.vectorize(lambda y : self.p0 * g(y))
    
        # compute the quality fo service QoS
        self.av_occupancy = spi.quad(lambda y : y*self.p1(y), 0, self.B)[0]
        self.occupancyRate = self.av_occupancy / self.B
        self.delta_QoS    = self.mu * self.p1(self.B) / self.lamb
        self.epsilon_QoS  = (self.mu_in - self.mu * (1-self.p0)) / self.mu_in ###
        self.t_QoS        = self.av_occupancy / self.mu     
        
        if display:
            print('Partially Rejection')
            print('1) job arrival rate  : %.3f jobs / sec'%(self.lamb))
            print('2) average job size  : %.3f bytes / job'%(self.LFc(0)))
            print('3) data arrival rate : %.3f bytes / sec'%(self.mu_in))
            print('4) transmission rate : %.3f bytes / sec'%(self.mu))
            print('5) average occupency : %.3f / %.3f bytes, occupency rate %.3f%%'%(self.av_occupancy,self.B,self.occupancyRate*100))
            print('6) empty probability : %.3f'%(self.p0))
            print('7) delta_QoS  , job retransmission proportion  : %.3f%%'%(self.delta_QoS*100))
            print('8) epsilon_QoS, data retransmission proportion : %.3f%%'%(self.epsilon_QoS*100))
            print('9) t_QoS, average wait time : %.3f sec'%(self.t_QoS))
            
    
    
    # def simu_PR(self, T):
    #     # SET PARAMETERS FOR LOOP #
    #     elapsing   = True # while loop parameter
    #     t_elapsed  = 0    # time elapsed since t=0 (starts at 0)
    #     arrivals   = np.zeros(1)   # array for time between 2 data arrivals, starting from 0
    #     buff_array = np.zeros(1)   # array to store buffer state at every data arrival (first entry is 0 for state at t=0)

    #     # SIMULATE BUFFER WITH WHILE LOOP #
    #     while elapsing:
            
    #         t_arrival = np.random.exponential(1/self.lamb) ## sample time between data arrivals from exp dist

    #         if t_elapsed + t_arrival < T: # make sure arrivals fall within the time range t specified
    #             arrivals = np.append(arrivals, t_arrival) 
    #             t_elapsed += t_arrival # update current time

    #             p_finder = np.random.uniform() # select a random value in range [0,1]
    #             p_index  = np.searchsorted(self.p.cumsum(), p_finder, 'right') # p_index ~ self.p, use p_finder to find which exponential in mixture to sample from
    #             job_size = np.random.exponential(1/(self.a[p_index])) ## sample job size from chosen distribution

    #             buff_state = min(self.B, max(buff_array[-1] - self.mu*t_arrival, 0) + job_size) # find buffer state
    #             buff_array = np.append(buff_array, buff_state)

    #         else:
    #             elapsing = False # end loop when time elapsed exceeds final time T
    #             if  t_elapsed + t_arrival == T :
    #                 p_finder = np.random.uniform() 
    #                 p_index  = np.searchsorted(self.p.cumsum(), p_finder, 'right') 
    #                 job_size = np.random.exponential(1/(self.a[p_index])) 
    #             else :
    #                 job_size = 0
    #             final_buff_state = min(self.B, max(buff_state - self.mu*(T-t_elapsed),0) + job_size) # find buffer state at final time t=T
    #             buff_array = np.append(buff_array,final_buff_state) 
    #             arrivals = np.append(arrivals, T-t_elapsed) 
        
    #     return (buff_array, arrivals.cumsum())
    
    
    # def simu_CR(self, T):
    #     # SET PARAMETERS FOR LOOP #
    #     elapsing   = True # while loop parameter
    #     t_elapsed  = 0    # time elapsed since t=0 (starts at 0)
    #     arrivals   = np.zeros(1)   # array for time between 2 data arrivals, starting from 0
    #     buff_array = np.zeros(1)   # array to store buffer state at every data arrival (first entry is 0 for state at t=0)

    #     # SIMULATE BUFFER WITH WHILE LOOP #
    #     while elapsing:
            
    #         t_arrival = np.random.exponential(1/self.lamb) ## sample time between data arrivals from exp dist

    #         if t_elapsed + t_arrival < T: # make sure arrivals fall within the time range t specified
    #             arrivals = np.append(arrivals, t_arrival) 
    #             t_elapsed += t_arrival # update current time

    #             p_finder = np.random.uniform() # select a random value in range [0,1]
    #             p_index  = np.searchsorted(self.p.cumsum(), p_finder, 'right') # p_index ~ self.p, use p_finder to find which exponential in mixture to sample from
    #             job_size = np.random.exponential(1/(self.a[p_index])) ## sample job size from chosen distribution
                
    #             if max(buff_array[-1] - self.mu*t_arrival, 0) + job_size <= self.B:
    #                 buff_state = max(buff_array[-1] - self.mu*t_arrival, 0) + job_size
    #             else:
    #                 buff_state = max(buff_array[-1] - self.mu*t_arrival, 0)
                    
    #             buff_array = np.append(buff_array, buff_state)

    #         else:
    #             elapsing = False # end loop when time elapsed exceeds final time T
    #             if  t_elapsed + t_arrival == T :
    #                 p_finder = np.random.uniform() 
    #                 p_index  = np.searchsorted(self.p.cumsum(), p_finder, 'right') 
    #                 job_size = np.random.exponential(1/(self.a[p_index])) 
    #             else :
    #                 job_size = 0
                
    #             if max(buff_state - self.mu*(T-t_elapsed),0) + job_size <= self.B:
    #                 final_buff_state = max(buff_state - self.mu*(T-t_elapsed),0) + job_size
    #             else:
    #                 final_buff_state = max(buff_state - self.mu*(T-t_elapsed),0) 

    #             buff_array = np.append(buff_array,final_buff_state) 
    #             arrivals = np.append(arrivals, T-t_elapsed) 
        
    #     return (buff_array, arrivals.cumsum())
    
    
    # def simu_PR_multipath(self, T, n, delta_t=0.05, type='PR'):
    #     # give n simulation paths
    #     if type == 'PR' :
    #         simus = [self.simu_PR(T) for i in range(n)]
    #     elif type == 'CR' :
    #         simus = [self.simu_CR(T) for i in range(n)]
    #     buffer_states = [simu[0] for simu in simus]
    #     arrivals = [simu[1] for simu in simus]
           
    #     # modify time intervals to scale of delta_t
    #     record_time = sorted(set(np.concatenate(arrivals)))
    #     modified_record_time = np.array([0])
        
    #     for i in range(1,len(record_time)) :
    #         if record_time[i] - record_time[i-1] <= delta_t :
    #             modified_record_time = np.append(modified_record_time, record_time[i])
    #         else :
    #             node_num = round(( record_time[i] - record_time[i-1] ) / delta_t) + 1
    #             new_nodes = np.linspace(record_time[i-1], record_time[i], node_num)[1:]
    #             modified_record_time = np.concatenate((modified_record_time, new_nodes))
          
    #     # fill in missing data
    #     time_series = pd.DataFrame({'Time': modified_record_time} ).set_index('Time')
    #     for i in range(len(arrivals)):
    #         time_series.loc[arrivals[i], i] = buffer_states[i] # fill in the states already given by the simulation
         
    #     time_series = time_series.reset_index()
              
    #     for col in tqdm(time_series.columns[1:]) :
    #         missing_indices = time_series[col].index[time_series[col].isnull()] # null value indeces
    #         for id_ in missing_indices :
    #             time_series.loc[id_,col] = max(time_series.loc[id_-1,col] - self.mu * (time_series.loc[id_,'Time'] - time_series.loc[id_-1,'Time']),0)

    #     return time_series
