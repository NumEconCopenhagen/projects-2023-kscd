from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class CobbDouglasModelClass:
    def __init__(self):
        """Setup model."""

        # a. create namespaces
        par = self.par = SimpleNamespace()

        # Base parameters
        par.A = 1
        par.K = 20
        par.L = 50
        par.alpha = 0.3
    
    def calculate_output(self):
        """Calculate the output of the Cobb-Douglas production function."""
        par = self.par
        Y = par.A * (par.K ** par.alpha) * (par.L ** (1 - par.alpha))
        return Y

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
        """Calculate the output, marginal product of capital, and marginal product of labor."""
        Y = self.calculate_output()
        MPK = self.calculate_mpk()
        MPL = self.calculate_mpl()

        if print_results:
            print("The calculated values for the baseline model are as follows:")
            print("- Output (Y):", Y)
            print("- Marginal Product of Capital (MPK):", MPK)
            print("- Marginal Product of Labor (MPL):", MPL)

        return Y, MPK, MPL


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
        plt.xlim(0,100)
        plt.ylim(0,100)
        ax.legend()
        ax.set_title('Output Changes with Constant Labor (L) and Capital (K)')
        plt.show()


    def plot_constant_K(self):
        """Plot the changes in output (Y) while holding capital (K) constant."""
        initial_K = self.par.K  # Store the initial value of K
        labor_values = np.linspace(0, 100, 100)  # Array of labor values ranging from 0 to 100
        output_values = np.zeros_like(labor_values)

        for i, L in enumerate(labor_values):
            self.par.L = L  # Vary the value of L
            output_values[i] = self.calculate_output()

        # Create a line plot of output as a function of labor
        plt.figure()
        plt.plot(labor_values, output_values)
        plt.xlabel('Labor (L)')
        plt.ylabel('Output (Y)')
        plt.grid()
        plt.xlim(0,100)
        plt.ylim(0,100)
        plt.title('Output Changes with Constant Capital')
        plt.show()

    def plot_constant_L(self):
        """Plot the changes in output (Y) while holding labor (L) constant."""
        initial_L = self.par.L  # Store the initial value of L
        capital_values = np.linspace(0, 100, 100)  # Array of capital values ranging from 0 to 100
        output_values = np.zeros_like(capital_values)

        for i, K in enumerate(capital_values):
            self.par.K = K  # Vary the value of K
            output_values[i] = self.calculate_output()

        # Create a line plot of output as a function of capital
        plt.figure()
        plt.plot(capital_values, output_values)
        plt.xlabel('Capital (K)')
        plt.ylabel('Output (Y)')
        plt.grid()
        plt.xlim(0,100)
        plt.ylim(0,100)
        plt.title('Output Changes with Constant Labor')
        plt.show()




    def optimize_production(self, initial_guess=None, print_results=False):
        """Maximize the output of the Cobb-Douglas Production Function subject to constraints."""

        # Define the objective function to maximize
        def objective(x):
            A = x[0]
            K = x[1]
            L = x[2]
            alpha = x[3]
            return -(-A * (K ** alpha) * (L ** (1 - alpha)))

        # Define the constraints
        def constraint1(x):
            return 100 - x[1] - x[2]  # Total amount of capital and labor used in production cannot exceed 100 units.

        def constraint2(x):
            return x[1] - 5  # Lower bound on the capital stock, which cannot be less than 5 units.

        def constraint3(x):
            return x[2] - 30  # Lower bound on the labor input, which cannot be less than 30 units.

        constraints = [{'type': 'ineq', 'fun': constraint1},
                       {'type': 'ineq', 'fun': constraint2},
                       {'type': 'ineq', 'fun': constraint3}]

        # Define the bounds on the parameters
        bounds = [(0.1, 10), (5, 50), (30, 70), (0.1, 0.9)]

        # Set the initial guess for the optimization routine
        if initial_guess is None:
            initial_guess = [self.par.A, self.par.K, self.par.L, self.par.alpha]

        # Run the optimization routine
        res = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if print_results:
            print("Optimal values:\nA = {:.2f}\nK = {:.2f}\nL = {:.2f}\nalpha = {:.2f}".format(*res.x))
            print("Output (Y): {:.2f}".format(-res.fun))

        # Update the model parameters with the optimal values
        self.par.A, self.par.K, self.par.L, self.par.alpha = res.x

        # Calculate the output, MPK, and MPL for the optimized model
        Y, MPK, MPL = self.cobb_douglas_analysis(print_results=print_results)

        return Y, MPK, MPL