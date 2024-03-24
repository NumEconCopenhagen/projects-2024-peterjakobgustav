from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize, minimize_scalar

class ExchangeEconomyClass:
    def __init__(self, omega_1A=0.8, omega_2A=0.3, N=75):
        # Initialize the parameters of the model
        self.par = SimpleNamespace(alpha=1/3, beta=2/3, omega_1A=omega_1A, omega_2A=omega_2A, N=N, p2=1)

    def u_A(self, x1_A, x2_A):
        # Utility function for consumer A
        return (x1_A ** self.par.alpha) * (x2_A ** (1 - self.par.alpha))

    def u_B(self, x1_B, x2_B):
        # Utility function for consumer B using Cobb-Douglas form
        return (x1_B ** self.par.beta) * (x2_B ** (1 - self.par.beta))

    def pareto_efficient_allocations(self):
        """Compute Pareto efficient allocations."""
        # Define the set of combinations of x1_A and x2_A
        x1_A_values = np.linspace(0, 1, self.par.N) # Possible values for x1_A (N+1 ensures 75 possible values)
        x2_A_values = np.linspace(0, 1, self.par.N) # Possible values for x2_A (N+1 ensures 75 possible values)
        # Initialize lists to store valid combinations
        valid_x1_A = []
        valid_x2_A = []
        # Check each combination of x1_A and x2_A for Pareto efficiency
        for x1_A in x1_A_values:
            for x2_A in x2_A_values:
                x1_B = 1 - x1_A
                x2_B = 1 - x2_A
                # Check if the combination is valid according to the given conditions and append if true
                if self.u_A(x1_A, x2_A) >= self.u_A(self.par.omega_1A, self.par.omega_2A) and \
                    self.u_B(x1_B, x2_B) >= self.u_B(1 - self.par.omega_1A, 1 - self.par.omega_2A):
                    valid_x1_A.append(x1_A)
                    valid_x2_A.append(x2_A)

        return valid_x1_A, valid_x2_A
    
    def demand_A(self, p1):
        x1_A_star = self.par.alpha * (p1 * self.par.omega_1A + self.par.omega_2A) / p1
        x2_A_star = (1 - self.par.alpha) * (p1 * self.par.omega_1A + self.par.omega_2A) / self.par.p2
        return x1_A_star, x2_A_star


    def demand_B(self, p1):
        x1_B_star = self.par.beta * (p1 * (1 - self.par.omega_1A) + (1 - self.par.omega_2A)) / p1
        x2_B_star = (1 - self.par.beta) * (p1 * (1 - self.par.omega_1A) + (1 - self.par.omega_2A)) / self.par.p2
        return x1_B_star, x2_B_star
    
    def market_clearing_error(self, p1):
        errors = []
        for p1 in P1:
            # Calculate errors by equations giving in the question
            error1 = x1_A_star - self.par.omega_1A + x1_B_star - (1 - self.par.omega_1A)
            error2 = x2_A_star - self.par.omega_2A + x2_B_star - (1 - self.par.omega_2A)
            # Append the errors to the before empty array 'errors'
            errors.append((error1, error2))
        return errors
    
    def max_u_A(self, P1):
        """Maximize utility for consumer A given prices P1"""
        max_utility = float('-inf')
        optimal_p_1 = None
        optimal_consumption_A = None
        for p1 in P1:
            x1_B_star, x2_B_star = self.demand_B(p1)
            x1_A_star, x2_A_star = 1 - x1_B_star, 1 - x2_B_star
            # Ensure x1_A_star and x2_A_star are positive
            if x1_A_star <= 0 or x2_A_star <= 0:
                continue  # Skip this iteration if either is non-positive
            u_A = self.u_A(x1_A_star, x2_A_star)
            if u_A > max_utility:
                max_utility = u_A
                optimal_p_1 = p1
                optimal_consumption_A = (x1_A_star, x2_A_star)
        return optimal_p_1, optimal_consumption_A, max_utility
    
    def max_u_A_cont(self):
        result = minimize(lambda p1: -self.u_A(*(1 - np.array(self.demand_B(p1[0])))), x0=[1], bounds=[(0.01, None)], method='L-BFGS-B')
        if result.success:
            optimal_price = result.x[0]
            optimal_allocation_A = (1 - np.array(self.demand_B(optimal_price)))
            return optimal_price, optimal_allocation_A, -result.fun
        else:
            raise ValueError("Optimization failed to maximize consumer A's utility.")