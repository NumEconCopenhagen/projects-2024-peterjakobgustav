def square(x):
    """ square numpy array
    
    Args:
    
        x (ndarray): input array
        
    Returns:
    
        y (ndarray): output array
    
    """
    
    y = x**2
    return y

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