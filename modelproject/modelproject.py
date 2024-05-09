import numpy as np
import matplotlib.pyplot as plt

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
