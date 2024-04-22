import numpy as np


class JobSizeGenerator:


    def __init__(self, p_values: np.ndarray, a_values: np.ndarray):
        self.p_values = p_values
        self.p_intervals = np.cumsum(p_values)
        self.a_values = a_values

    def f_random_size(self):
        uniform_random = np.random.uniform()
        p_index = np.searchsorted(self.p_intervals, uniform_random, 'right')
        return np.random.exponential(1/self.a_values[p_index])
    
    def complementary_cdf(self, x:float):
        return np.sum(self.p_values * np.exp(-self.a_values * x))
    
    