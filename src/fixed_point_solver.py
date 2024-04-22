import typing
import numpy as np


class FixedPointSolver:


    def __init__(self,
                 mu: float,
                 param_lambda: float,
                 buffer_size: float,
                 F_c: typing.Callable,
                 n_grid_points: int = 101,
                 initial_guess: typing.Optional[np.ndarray] = None,
                 n_iter: int = 1000,
                 ):
        self.mu = mu
        self.param_lambda = param_lambda
        self.buffer_size = buffer_size
        self.F_c = F_c
        self.n_grid_points = n_grid_points
        self.h = buffer_size / (n_grid_points - 1)
        if initial_guess is None:
            self.g = np.zeros(n_grid_points)
        else:
            self.g = initial_guess
        self.max_iter = n_iter
        self.grid = np.linspace(0, buffer_size, n_grid_points)

    def solve(self):
        for i in range(self.max_iter):
            new_g = self.g.copy()
            for j in range(self.n_grid_points - 1):
                new_g[j] = (self.param_lambda / self.mu) * (self.F_c(self.grid[j]) - self.F_c(self.buffer_size)
                                                           + self.integrate_to_index_j(j))
                
                # print("F_c contribution", self.F_c(self.grid[j]) - self.F_c(self.buffer_size))
                # print("Integral contribution", self.param_lambda * self.integrate_to_index_j(j))
            if np.linalg.norm(new_g - self.g) < 1e-6:
                break
            self.g = new_g
    
    def integrate_to_index_j(self, j: int):
        if j == 0:
            return 0
        return self.h * (0.5 * self.integrand(0, j) 
                         + sum([self.integrand(j_prime, j) for j_prime in range(1, j)])
                         + 0.5 * self.integrand(j, j))
                                                   
    def integrand(self, j_prime: int, j: int):
        return self.g[j] * (self.F_c(self.grid[j] - self.grid[j_prime]) - self.F_c(self.buffer_size - self.grid[j_prime]))
    
    def compute_probabilities(self):
        integral_g = self.h * ( np.sum( self.g[1:-1] + 0.5 * (self.g[0] + self.g[-1]) ) )
        p_0 = 1 / (1 + integral_g)
        p_1 = p_0 * self.g
        return p_0, p_1
