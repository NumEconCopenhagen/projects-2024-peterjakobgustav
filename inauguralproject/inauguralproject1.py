from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize, minimize_scalar

class ExchangeEconomyClass:
    def __init__(self, w1A=0.8, w2A=0.3):
        # Constructor for the exchange economy class setting up parameters:
        # alpha: Cobb-Douglas preference parameter for consumer A
        # beta: Cobb-Douglas preference parameter for consumer B
        # w1A, w2A: Initial endowments of goods 1 and 2 for consumer A
        # p2: Price of good 2, set as the numeraire (fixed to 1)
        self.par = SimpleNamespace(alpha=1/3, beta=2/3, w1A=w1A, w2A=w2A, p2=1)

    def utility_A(self, x1A, x2A):
        # Utility function for consumer A using Cobb-Douglas form
        return (x1A ** self.par.alpha) * (x2A ** (1 - self.par.alpha))

    def utility_B(self, x1B, x2B):
        # Utility function for consumer B using Cobb-Douglas form
        return (x1B ** self.par.beta) * (x2B ** (1 - self.par.beta))

    def demand_A(self, p1):
        # Demand function for consumer A deriving from the utility maximization
        # subject to the budget constraint with prices p1 for good 1 and self.par.p2 for good 2
        income_A = self.par.w1A * p1 + self.par.w2A * self.par.p2
        x1A_star = self.par.alpha * (income_A / p1)
        x2A_star = (1 - self.par.alpha) * (income_A / self.par.p2)
        return x1A_star, x2A_star


    def demand_B(self, p1):
        # Demand function for consumer B deriving from the utility maximization
        # subject to the budget constraint with prices p1 for good 1 and self.par.p2 for good 2
        income_B = (1 - self.par.w1A) * p1 + (1 - self.par.w2A) * self.par.p2
        x1B_star = self.par.beta * (income_B / p1)
        x2B_star = (1 - self.par.beta) * (income_B / self.par.p2)
        return x1B_star, x2B_star
    
    
    
    def max_A_utility(self, P1):
        """Maximize utility for consumer A given prices P1"""
        max_utility = float('-inf')
        optimal_p_1 = None
        optimal_consumption_A = None
        for p1 in P1:
            x1_B_star, x2_B_star = self.demand_B(p1)
            x1_A_star, x2_A_star = 1 - x1_B_star, 1 - x2_B_star
            utility_A = self.utility_A(x1_A_star, x2_A_star)
            if utility_A > max_utility:
                max_utility = utility_A
                optimal_p_1 = p1
                optimal_consumption_A = (x1_A_star, x2_A_star)
        return optimal_p_1, optimal_consumption_A, max_utility
    
    def max_A_utility_cont(self):
        # Function to maximize consumer A's utility for any positive price of good 1
        # Uses a numerical solver to maximize the utility as a continuous function of price
        result = minimize(lambda p1: -self.utility_A(*(1 - np.array(self.demand_B(p1[0])))), x0=[1], bounds=[(0.01, None)], method='L-BFGS-B')
        if result.success:
            optimal_price = result.x[0]
            optimal_allocation_A = (1 - np.array(self.demand_B(optimal_price)))
            return optimal_price, optimal_allocation_A, -result.fun
        else:
            raise ValueError("Optimization failed to maximize consumer A's utility.")