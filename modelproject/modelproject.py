import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

class SolowModel1:
    def __init__(self, alpha=0.3, s=0.2, delta=0.05, n=0.01, g=0.02):
        self.alpha = alpha  # Output elasticity of capital
        self.s = s  # Savings rate
        self.delta = delta  # Depreciation rate
        self.n = n  # Population growth rate
        self.g = g  # Technological progress rate
        self.k = np.linspace(0, 10, 100)  # Capital per worker values

    # Production function: Y = K^alpha * (AL)^(1-alpha)
    def production_function(self, k):
        return k**self.alpha

    # Savings function: s * Y
    def savings_function(self, k):
        return self.s * self.production_function(k)

    # Depreciation + (n + g) * k
    def depreciation_line(self, k):
        return (self.delta + self.n + self.g) * k

    def simulate(self):
        self.k_star = (self.s / (self.delta + self.n + self.g))**(1 / (1 - self.alpha))
        self.y_star = self.production_function(self.k_star)

    def plot_results(self, title):
        plt.figure(figsize=(10, 6))

        # Plot production function
        plt.plot(self.k, self.production_function(self.k), label='Production Function (Y = K^α)', color='b')

        # Plot savings function
        plt.plot(self.k, self.savings_function(self.k), label='Savings (sY)', color='g')

        # Plot depreciation line
        plt.plot(self.k, self.depreciation_line(self.k), label='Depreciation + (n + g)K', color='r')

        # Steady-state lines
        plt.axvline(x=self.k_star, linestyle='--', color='gray')
        plt.axhline(y=self.y_star, linestyle='--', color='gray')

        # Labels and title
        plt.xlabel('Capital per Worker (k)')
        plt.ylabel('Output per Worker (y) / Savings / Depreciation')
        plt.title(title)
        plt.legend()

        # Steady-state annotation
        plt.text(self.k_star, 0, f'k* = {self.k_star:.2f}', verticalalignment='bottom', horizontalalignment='right')
        plt.text(0, self.y_star, f'y* = {self.y_star:.2f}', verticalalignment='bottom', horizontalalignment='left')

        plt.grid(True)

        plt.show()


    def solve_analytically(self):
        # Define the symbols
        k, alpha, s, delta, n, g = sp.symbols('k alpha s delta n g')

        # Define the production function f(k) = k^alpha
        f_k = k**alpha

        # Define the steady-state equation s * f(k) = (delta + n + g) * k
        steady_state_eq = sp.Eq(s * f_k, (delta + n + g) * k)

        # Solve for k
        steady_state_k = sp.solve(steady_state_eq, k)
        return steady_state_k


class SolowModel:
    def __init__(self, s=0.3, delta=0.05, n=0.01, g=0.02, alpha=0.3, K0=100, L0=100, A0=1, T=60, dt=1):
        # Economic parameters initialization
        self.s = s                    # Savings rate
        self.delta = delta            # Depreciation rate
        self.n = n                    # Labor growth rate
        self.g = g                    # Technological growth rate
        self.alpha = alpha            # Output elasticity of capital
        self.K0 = K0                  # Initial capital stock
        self.L0 = L0                  # Initial labor
        self.A0 = A0                  # Initial technology level
        self.T = T                    # Number of periods
        self.dt = dt                  # Time step
        
        # Arrays to hold the time series data of the model
        self.K = np.zeros(T)
        self.L = np.zeros(T)
        self.A = np.zeros(T)
        self.Y = np.zeros(T)
        self.cor = np.zeros(T)        # Capital-Output Ratio

    def simulate(self, policy_shock_period=None, policy_shock_delta=0):
        # Reset initial conditions for each simulation
        self.K[0], self.L[0], self.A[0] = self.K0, self.L0, self.A0
        for t in range(1, self.T):
            # Apply a temporary policy shock if specified
            if policy_shock_period and t == policy_shock_period:
                original_delta = self.delta
                self.delta += policy_shock_delta  # Increase depreciation rate
            
            # Solow model dynamics
            self.Y[t-1] = self.A[t-1] * self.K[t-1]**self.alpha * self.L[t-1]**(1-self.alpha)
            self.K[t] = self.K[t-1] + (self.s * self.Y[t-1] - self.delta * self.K[t-1]) * self.dt
            self.L[t] = self.L[t-1] * np.exp(self.n * self.dt)
            self.A[t] = self.A[t-1] * np.exp(self.g * self.dt)
            self.cor[t] = self.K[t] / self.Y[t-1] if self.Y[t-1] != 0 else 0

            if policy_shock_period and t == policy_shock_period:
                self.delta = original_delta  # Revert the depreciation rate after the shock

        self.Y[self.T-1] = self.A[self.T-1] * self.K[self.T-1]**self.alpha * self.L[self.T-1]**(1-self.alpha)
        self.cor[self.T-1] = self.K[self.T-1] / self.Y[self.T-1] if self.Y[self.T-1] != 0 else 0

    def plot_results(self, title="Solow Model Simulation"):
        plt.figure(figsize=(12, 8))
        plt.plot(self.Y, label='Output')
        plt.plot(self.K, label='Capital')
        plt.plot(self.cor, label='Capital-Output Ratio', linestyle='--')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Levels')
        plt.legend()
        plt.grid(True)
        plt.show()

import sympy as sp
from IPython.display import display, Math
import numpy as np
import matplotlib.pyplot as plt

class HumanCapital:
    def __init__(self, alpha, beta, s_K, s_H, delta, n, g, A, L):
        # Initialize parameters
        self.alpha = alpha
        self.beta = beta
        self.s_K = s_K
        self.s_H = s_H
        self.delta = delta
        self.n = n
        self.g = g
        self.A = A
        self.L = L

        # Define the symbols
        self.K, self.H, self.Y, self.A_sym, self.L_sym = sp.symbols('K H Y A L')
        
        # Production function with human capital
        self.production_function = self.K**self.alpha * self.H**self.beta * (self.A_sym * self.L_sym)**(1 - self.alpha - self.beta)
        
    def solve_analytically(self):
        # Steady-state conditions for capital accumulation
        steady_state_K = sp.Eq(self.s_K * self.production_function, (self.delta + self.n + self.g) * self.K)
        steady_state_H = sp.Eq(self.s_H * self.production_function, (self.delta + self.n + self.g) * self.H)

        # Solve for K and H in terms of Y
        K_expr = sp.solve(steady_state_K, self.K)[0]
        H_expr = sp.solve(steady_state_H, self.H)[0]

        # Substitute K_expr and H_expr into the production function
        production_substituted = self.production_function.subs({self.K: K_expr, self.H: H_expr})

        # Solve for Y
        Y_solution = sp.solve(sp.Eq(self.Y, production_substituted), self.Y)[0]

        # Simplify the solution for Y
        self.Y_solution = sp.simplify(Y_solution)

        # Solve for K and H using the steady-state value of Y
        self.K_ss_solution = sp.simplify(K_expr.subs(self.Y, self.Y_solution))
        self.H_ss_solution = sp.simplify(H_expr.subs(self.Y, self.Y_solution))

    def display_solutions(self):
        display(Math(r"Y^* = " + sp.latex(self.Y_solution)))
        display(Math(r"K^* = " + sp.latex(self.K_ss_solution)))
        display(Math(r"H^* = " + sp.latex(self.H_ss_solution)))

    def simulate(self, periods=100, k0=1, h0=1):
        # Initialize arrays to store the results
        k_values = np.zeros(periods)
        h_values = np.zeros(periods)
        y_values = np.zeros(periods)
        k_values[0] = k0
        h_values[0] = h0

        # Iterate over periods to simulate the model
        for t in range(1, periods):
            y_values[t-1] = k_values[t-1]**self.alpha * h_values[t-1]**self.beta * (self.A * self.L)**(1 - self.alpha - self.beta)
            k_values[t] = self.s_K * y_values[t-1] + (1 - self.delta - self.n - self.g) * k_values[t-1]
            h_values[t] = self.s_H * y_values[t-1] + (1 - self.delta - self.n - self.g) * h_values[t-1]
        
        y_values[-1] = k_values[-1]**self.alpha * h_values[-1]**self.beta * (self.A * self.L)**(1 - self.alpha - self.beta)

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(range(periods), k_values, label='Physical Capital per Worker (K)')
        plt.plot(range(periods), h_values, label='Human Capital per Worker (H)')
        plt.plot(range(periods), y_values, label='Output per Worker (Y)')
        plt.xlabel('Periods')
        plt.ylabel('Values')
        plt.title('Simulation of the MRW Model')
        plt.legend()
        plt.grid(True)
        plt.show()
