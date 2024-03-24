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
    
    def market_clearing_error(self, p1_values):
        errors = []
        for p1 in p1_values:
            # Calculate allocations for consumer A and B using the given price p
            x1_A_star, x2_A_star = self.demand_A(p1)
            x1_B_star, x2_B_star = self.demand_B(p1)
            # Calculate errors by equations giving in the question
            error1 = x1_A_star - self.par.omega_1A + x1_B_star - (1 - self.par.omega_1A)
            error2 = x2_A_star - self.par.omega_2A + x2_B_star - (1 - self.par.omega_2A)
            # Append the errors to the before empty array 'errors'
            errors.append((error1, error2))
        return errors
    
    def market_clearing_price(self):
     # Solve market clearing conditions for p1 = p* with a interval of 10000
        for p1 in np.linspace(0.5, 2.5, 10000):
            # Calculate allocations for consumer A
            x1_A_star, x2_A_star = self.demand_A(p1)
            # Calculate allocations for consumer B
            x1_B_star, x2_B_star = self.demand_B(p1)
            # Check if market clears that is if x1_A_star + x1_B_star is close to 1 and the same
            # for x2_A_star + x2_B_star with the implemting of Walras' law
            if np.isclose(x1_A_star + x1_B_star, 1) and np.isclose(x2_A_star + x2_B_star, 1):
                return p1  # Return the market clearing price when found
    
    def optimize_u_A(self, p1_values):
        """Maximize utility for consumer A given prices p1_values"""
        optimal_utility = float('-inf')
        optimal_p_1 = None
        optimal_consumption_A = None
        for p1 in p1_values:
            x1_B_star, x2_B_star = self.demand_B(p1)
            x1_A_star, x2_A_star = 1 - x1_B_star, 1 - x2_B_star
            # Ensure x1_A_star and x2_A_star are positive
            if x1_A_star <= 0 or x2_A_star <= 0:
                continue  # Skip this iteration if either is non-positive
            u_A = self.u_A(x1_A_star, x2_A_star)
            if u_A > optimal_utility:
                optimal_utility = u_A
                optimal_p_1 = p1
                optimal_consumption_A = (x1_A_star, x2_A_star)
        return optimal_p_1, optimal_consumption_A, optimal_utility
       
    def optimize_u_A_continuous(self):
        """Optimizes utility for consumer A by finding the best price that maximizes utility."""
        # Define the objective function for optimization: Maximize utility of A given B's demand.
        objective = lambda price: -self.u_A(*(1 - np.array(self.demand_B(price[0]))))
    
        # Perform the optimization with initial guess and bounds for price.
        optimization_result = minimize(objective, x0=[1], bounds=[(0.01, None)])
    
        # Check if the optimization was successful and process the results.
        if optimization_result.success:
            optimal_price_continuous = optimization_result.x[0]  # Optimal price point
            optimal_allocation_A_continuous = 1 - np.array(self.demand_B(optimal_price_continuous))  # Allocation for A
            optimal_utility_continuous = -optimization_result.fun  # Maximum utility achieved at optimal price
        
            return optimal_price_continuous, optimal_allocation_A_continuous, optimal_utility_continuous
        else:
            # Show error if optimization was unsuccessful
            raise ValueError("Unable to find optimal solution for maximizing consumer A's utility.")
        





    def generate_W(self, num_elements=50):
        """Generate a set W with 50 elements of (omega_1A, omega_2A)."""
        np.random.seed(10)  # For reproducibility
        self.W = np.random.uniform(0, 1, (num_elements, 2))
        return self.W
    

    def market_clearing_price_8(self, p1_values):
        p1_values = np.linspace(0.5, 2.5, self.par.N)
        for p1 in p1_values:
            # Calculate allocations for consumer A
            x1_A_star, x2_A_star = self.demand_A(p1)
            # Calculate allocations for consumer B
            x1_B_star, x2_B_star = self.demand_B(p1)
            # Check if market clears that is if x1_A_star + x1_B_star is close to 1 and the same
            # for x2_A_star + x2_B_star with the implemting of Walras' law
            if np.isclose(x1_A_star + x1_B_star, 1) and np.isclose(x2_A_star + x2_B_star, 1):
                return p1  # Return the market clearing price when found


    def find_market_equilibrium_for_W(self):
        if not hasattr(self, 'W'):
            self.generate_W()
        
        equilibrium_prices = []
        allocations_A = []
        allocations_B = []
    
        for omega_1A, omega_2A in self.W:
            self.par.omega_1A, self.par.omega_2A = omega_1A, omega_2A
            p1_star = self.market_clearing_price_8()
        
            if p1_star is None:
                print("Market clearing price not found for omega_1A = ", omega_1A, "omega_2A = ", omega_2A)
                continue  # Skip this iteration if no market clearing price was found
        
            xA_star = self.demand_A(p1_star)
            xB_star = self.demand_B(p1_star)
        
            equilibrium_prices.append(p1_star)
            allocations_A.append(xA_star)
            allocations_B.append(xB_star)
    
        return equilibrium_prices, allocations_A, allocations_B

    