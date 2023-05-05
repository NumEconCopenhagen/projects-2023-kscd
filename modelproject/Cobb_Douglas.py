from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize

class CobbDouglasModelClass:
    def __init__(self):
        """Setup model."""

        # a. create namespaces
        par = self.par = SimpleNamespace()

        # Base parameters
        par.A = 1
        par.K = 10
        par.L = 50
        par.alpha = 0.3

    def cobb_douglas_analysis(self, print_results=False):
        """Calculate the output, marginal product of capital, and marginal product of labor."""

        par = self.par

        # Calculate output
        Y = par.A * (par.K ** par.alpha) * (par.L ** (1 - par.alpha))

        # Calculate MPK
        MPK = par.alpha * par.A * (par.K ** (par.alpha - 1)) * (par.L ** (1 - par.alpha))

        # Calculate MPL
        MPL = (1 - par.alpha) * par.A * (par.K ** par.alpha) * (par.L ** (-par.alpha))

        if print_results:
            print("Output (Y):", Y)
            print("Marginal Product of Capital (MPK):", MPK)
            print("Marginal Product of Labor (MPL):", MPL)

        return Y, MPK, MPL

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