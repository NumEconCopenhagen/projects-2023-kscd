from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class CobbDouglasModelClass:
    def __init__(self):
        """Setup model."""

        # create namespaces
        par = self.par = SimpleNamespace()

        # Base parameters
        par.A = 1
        par.K = 20
        par.L = 50
        par.alpha = 0.3
        par.w = 1  # wage rate
        par.r = 1  # rental rate of capital

    def calculate_output(self):
        """Calculate the output of the Cobb-Douglas production function."""
        par = self.par
        Y = par.A * (par.K ** par.alpha) * (par.L ** (1 - par.alpha))
        return Y

    def calculate_cost(self):
        """Calculate the total cost of production."""
        par = self.par
        cost = par.w * par.L + par.r * par.K
        return cost

    def calculate_mpk(self):
        """Calculate the marginal product of capital (MPK)."""
        par = self.par
        MPK = par.alpha * par.A * (par.K ** (par.alpha - 1)) * (par.L ** (1 - par.alpha))
        return MPK

    def calculate_mpl(self):
        """Calculate the marginal product of labor (MPL)."""
        par = self.par
        MPL = (1 - par.alpha) * par.A * (par.K ** par.alpha) * (par.L ** (-par.alpha))
        return MPL

    def cobb_douglas_analysis(self, print_results=False):
        """Calculate the output, cost, MPK, and MPL."""
        Y = self.calculate_output()
        cost = self.calculate_cost()
        MPK = self.calculate_mpk()
        MPL = self.calculate_mpl()

        if print_results:
            print("The calculated values for the model are as follows:")
            print(f"- Output (Y): {Y:.3f}")
            print(f"- Cost: {cost:.3f}")
            print(f"- Marginal Product of Capital (MPK): {MPK:.3f}")
            print(f"- Marginal Product of Labor (MPL): {MPL:.3f}")

        return Y, cost, MPK, MPL

    def plot_constant_K_and_L(self):
        """Plot the changes in output (Y) while holding capital (K) and labor (L) constant."""
        capital_values = np.linspace(0, 100, 100)  # Array of capital values ranging from 0 to 100
        output_L_values = np.zeros_like(capital_values)
        labor_values = np.linspace(0, 100, 100)  # Array of labor values ranging from 0 to 100
        output_K_values = np.zeros_like(labor_values)

        # Calculate output with constant labor (L) and varying capital (K)
        for i, K in enumerate(capital_values):
            self.par.K = K
            output_L_values[i] = self.calculate_output()

        # Calculate output with constant capital (K) and varying labor (L)
        for i, L in enumerate(labor_values):
            self.par.L = L
            output_K_values[i] = self.calculate_output()

        # Create a line plot with two lines, one for constant labor (L) and one for constant capital (K)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(labor_values, output_K_values, color='red', label='Constant Labor (L)')
        ax.plot(capital_values, output_L_values, color='blue', label='Constant Capital (K)')
        ax.set_xlabel('Labor (L) / Capital (K)')
        ax.set_ylabel('Output (Y)')
        ax.grid()
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        ax.legend()
        ax.set_title('Output Changes with Constant Labor (L) and Capital (K)')
        plt.show()

    def _difference_from_target_cost(self, KL, target_cost):
        """Calculate the difference between the calculated cost and the target cost."""
        self.par.K, self.par.L = KL
        cost = self.calculate_cost()
        return abs(cost - target_cost)

    def optimize_KL_given_Y(self, target_Y, target_cost, print_results=False):
        """Optimize K and L based on a given output Y and target cost."""
        initial_KL = (self.par.K, self.par.L)
        result = minimize(self._difference_from_target_cost, initial_KL, args=(target_cost,), method='Nelder-Mead')
        self.par.K, self.par.L = result.x

        # Calculate the output, cost, MPK, and MPL for the optimized model
        Y_optimized, cost_optimized, MPK_optimized, MPL_optimized = self.cobb_douglas_analysis()

        if print_results:
            print("Optimal values of K and L for the desired output and cost:", result.x)
            print("")
            print("The calculated values for the optimized model are as follows:")
            print(f"- Output (Y): {Y_optimized:.3f}")
            print(f"- Cost: {cost_optimized:.3f}")
            print(f"- Marginal Product of Capital (MPK): {MPK_optimized:.3f}")
            print(f"- Marginal Product of Labor (MPL): {MPL_optimized:.3f}")

        return result.x, Y_optimized, cost_optimized, MPK_optimized, MPL_optimized

class ExtendedCobbDouglasModelClass(CobbDouglasModelClass):
    def __init__(self):
        super().__init__()

        # Extended parameters
        self.par.T = 1.1
        self.par.beta = 0.2

    def calculate_output(self):
        """Calculate the output of the extended Cobb-Douglas production function."""
        par = self.par
        Y = par.A * (par.K ** par.alpha) * (par.L ** (1 - par.alpha)) * (par.T ** par.beta)
        return Y

    def calculate_mpk(self):
        """Calculate the marginal product of capital (MPK) for the extended model."""
        par = self.par
        MPK = par.alpha * par.A * (par.K ** (par.alpha - 1)) * (par.L ** (1 - par.alpha)) * (par.T ** par.beta)
        return MPK

    def calculate_mpl(self):
        """Calculate the marginal product of labor (MPL) for the extended model."""
        par = self.par
        MPL = (1 - par.alpha) * par.A * (par.K ** par.alpha) * (par.L ** (-par.alpha)) * (par.T ** par.beta)
        return MPL

    def optimize_KL_given_Y(self, target_Y, target_cost, print_results=False):
        """Optimize K and L based on a given output Y and target cost while keeping other parameters constant."""
        initial_KL = (self.par.K, self.par.L)  # Store the initial values of K and L
        result = minimize(self._difference_from_target_cost, initial_KL, args=(target_cost,), method='Nelder-Mead')
        K_optimized, L_optimized = result.x

        # Calculate the output, cost, MPK, and MPL for the optimized model using the optimized K and L values
        self.par.K, self.par.L = K_optimized, L_optimized
        Y_optimized, cost_optimized, MPK_optimized, MPL_optimized = self.cobb_douglas_analysis()

        if print_results:
            print("Optimal values of K and L for the desired output and cost:", result.x)
            print("")
            print("The calculated values for the optimized model are as follows:")
            print(f"- Output (Y): {Y_optimized:.3f}")
            print(f"- Cost: {cost_optimized:.3f}")
            print(f"- Marginal Product of Capital (MPK): {MPK_optimized:.3f}")
            print(f"- Marginal Product of Labor (MPL): {MPL_optimized:.3f}")

        # Restore the initial values of K and L
        self.par.K, self.par.L = initial_KL

        return result.x, Y_optimized, cost_optimized, MPK_optimized, MPL_optimized
