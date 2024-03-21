def square(x):
    """ square numpy array
    
    Args:
    
        x (ndarray): input array
        
    Returns:
    
        y (ndarray): output array
    
    """
    
    y = x**2
    return y

    def maximize_consumer_A_utility_discrete(self, P1):
        """Maximize consumer A's utility over a discrete set of prices P1."""
        max_utility = float('-inf')
        optimal_price = None
        optimal_allocation_A = None
        for p1 in P1:
            x1B_star, x2B_star = self.demand_B(p1)
            x1A_star, x2A_star = 1 - x1B_star, 1 - x2B_star
            utility_A = self.utility_A(x1A_star, x2A_star)
            if utility_A > max_utility:
                max_utility = utility_A
                optimal_price = p1
                optimal_allocation_A = (x1A_star, x2A_star)
        return optimal_price, optimal_allocation_A, max_utility