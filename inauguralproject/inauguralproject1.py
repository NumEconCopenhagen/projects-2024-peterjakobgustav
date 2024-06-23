from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
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
        x1_A_values = np.linspace(0, 1, self.par.N+1) # Possible values for x1_A 
        x2_A_values = np.linspace(0, 1, self.par.N+1) # Possible values for x2_A
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
            # Check if market clears that is if x1_A_star + x1_B_star is close to 1 and the same for x2_A_star + x2_B_star utilizing Walras' law
            if np.isclose(x1_A_star + x1_B_star, 1) and np.isclose(x2_A_star + x2_B_star, 1):
                return p1  # Return the market clearing price when found
    
    def maximize_u_A(self, p1_values):
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
       
    def maximize_u_A_continuous(self):
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
        

    def optimal_allocation_5a(self):
        # Calculate valid Pareto efficient allocations
        valid_x1_A, valid_x2_A = self.pareto_efficient_allocations()

        # Initialize variables to store the optimal allocation and utility
        optimal_allocation = None
        max_utility = -float('inf')

        # Iterate through valid combinations and find the one with maximum utility
        for i in range(len(valid_x1_A)):
            x1_A = valid_x1_A[i]
            x2_A = valid_x2_A[i]
            utility = self.u_A(x1_A, x2_A)
            if utility > max_utility:
                max_utility = utility
                optimal_allocation = (x1_A, x2_A)

        # Print the optimal allocation and its utility
        if optimal_allocation is not None:
            print(f"Optimal Allocation (x1^A, x2^A): ({optimal_allocation[0]:.4f}, {optimal_allocation[1]:.4f})")
            print(f"Maximum Utility: {max_utility:.4f}")
        else:
            print("No optimal allocation found.")
            print("Maximum Utility:", max_utility)


    def optimal_allocation_5b(self):
        # Utility functions
        def uA(x1, x2):
            return x1**self.par.alpha * x2**(1-self.par.alpha)

        def uB(x1, x2):
            return x1**self.par.beta * x2**(1-self.par.beta)

        # Parameters for consumer B
        omega_1B = 1 - self.par.omega_1A
        omega_2B = 1 - self.par.omega_2A

        # Initial utility levels for comparison
        initial_utility_A = uA(self.par.omega_1A, self.par.omega_2A)
        initial_utility_B = uB(omega_1B, omega_2B)

        # Objective function: Maximize A's utility
        def objective(x):
            xA1, xA2 = x
            return -uA(xA1, xA2)  # Negative because we use minimize

        # Constraints for Pareto improvements and total consumption
        constraints = (
            {'type': 'ineq', 'fun': lambda x: uB(1 - x[0], 1 - x[1]) - initial_utility_B},  # uB(1-xA1, 1-xA2) >= uB(ω1B, ω2B)
        )

        # Initial guess
        x0 = [self.par.omega_1A, self.par.omega_2A]

        # Bounds for xA1 and xA2
        bounds = ((0, 1), (0, 1))

        # Perform the optimization
        result = minimize(objective, x0, bounds=bounds, constraints=constraints)

        if result.success:
            optimal_xA1, optimal_xA2 = result.x
            optimal_utility = -result.fun
            print(f"Optimal Allocation: xA1 = {optimal_xA1:.4f}, xA2 = {optimal_xA2:.4f}, Maximum Utility for A: {optimal_utility:.4f}")
        else:
            print("Optimization was not successful. Please check the constraints and initial guess.")



    def optimal_allocation_6(self):
        # Utility functions
        def uA(x1, x2):
            return x1**self.par.alpha * x2**(1-self.par.alpha)

        def uB(x1, x2):
            return x1**self.par.beta * x2**(1-self.par.beta)

        def aggregate_u(x1, x2):
            return uA(x1, x2) + uB(1 - x1, 1 - x2)

        # Parameters for consumer B
        omega_1B = 1 - self.par.omega_1A
        omega_2B = 1 - self.par.omega_2A

        # Objective function: Maximize aggregate utility
        def objective(x):
            xA1, xA2 = x
            return -aggregate_u(xA1, xA2)  # Negative because we use minimize

        # Initial guess (could be the initial endowments)
        x0 = [self.par.omega_1A, self.par.omega_2A]

        # Bounds for xA1 and xA2
        bounds = ((0, 1), (0, 1))

        # Perform the optimization
        result = minimize(objective, x0, bounds=bounds)

        if result.success:
            optimal_agg_xA1, optimal_agg_xA2 = result.x
            optimal_agg_utility = -result.fun
            print(f"Optimal Allocation: xA1 = {optimal_agg_xA1:.4f}, xA2 = {optimal_agg_xA2:.4f}, Maximum Utility for A: {optimal_agg_utility:.4f}")
        else:
            print("Optimization was not successful. Please check the constraints and initial guess.")


    def generate_W(self, num_elements=50):
        """Generate a set W with 50 elements of (omega_1A, omega_2A) and return it as a list of tuples."""
        np.random.seed(10)  # For reproducibility
        self.W = np.random.uniform(0, 1, (num_elements, 2))
        # Convert to list of tuples
        list_of_tuples = [tuple(row) for row in self.W]
        return list_of_tuples

    def market_clearing_error_for_p1(self, p1, omega_1A, omega_2A):
        """Calculate the squared sum of market clearing errors for a given price p1 and endowments."""
        self.par.omega_1A = omega_1A
        self.par.omega_2A = omega_2A
        x1_A_star, x2_A_star = self.demand_A(p1)
        x1_B_star, x2_B_star = self.demand_B(p1)
        error1 = (x1_A_star + x1_B_star - 1) ** 2
        error2 = (x2_A_star + x2_B_star - 1) ** 2
        return error1 + error2

    def find_market_clearing_price(self, omega_1A, omega_2A):
        """Find the market clearing price for given endowments by minimizing market clearing errors."""
        result = minimize(self.market_clearing_error_for_p1, x0=[1], args=(omega_1A, omega_2A), bounds=[(0.01, None)])
        if result.success:
            return result.x[0]
        else:
            raise ValueError("Optimization failed to find a market clearing price.")

    def find_and_plot_equilibria(self):
        """Find equilibrium allocations for each pair in W and plot them in the Edgeworth box."""
        equilibria = []
        for omega_1A, omega_2A in self.generate_W():
            p1_star = self.find_market_clearing_price(omega_1A, omega_2A)
            x1_A_star, x2_A_star = self.demand_A(p1_star)
            # Store the equilibrium allocation
            equilibria.append((x1_A_star, x2_A_star))

        # Unpack the equilibrium allocations for plotting
        x1_A_stars, x2_A_stars = zip(*equilibria)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x1_A_stars, x2_A_stars, c='blue', label='Equilibrium Allocations')
        ax.set_xlabel('$x_{1A}^*$')
        ax.set_ylabel('$x_{2A}^*$')
        ax.set_title('Equilibrium Allocations in the Edgeworth Box')
        ax.legend()
        ax.grid(True)
        
        # Create twin axes for the right and top axes
        ax_right = ax.twinx()
        ax_top = ax.twiny()
        
        # Set labels for the right and top axes
        ax_right.set_ylabel('$x_{2B}^*$', fontsize=12)
        ax_top.set_xlabel('$x_{1B}^*$', fontsize=12)
        
        # Align the right and top axes limits with the primary axes and invert them
        ax_right.set_ylim(1, 0)
        ax_top.set_xlim(1, 0)
        
        # Displaying the plot
        plt.show()


    