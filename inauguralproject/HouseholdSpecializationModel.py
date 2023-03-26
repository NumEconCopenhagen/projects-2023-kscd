from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = np.minimum(HM,HF)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**((par.sigma/(par.sigma-1)))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve_continuous(self,do_print=False):
        """ solve model continously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # a. Start by making guesses, this is because we need a starting point
        LM_guess = 6
        LF_guess = 6
        HM_guess = 6
        HF_guess = 6
        x_guess = [LM_guess, LF_guess, HM_guess, HF_guess]

        # Create an objective.
        # The objective is a negative utility function, thereby to maximize this function we need to optimize.minimize it
        obj = lambda x: -self.calc_utility(x[0], x[1], x[2], x[3])

        # c. We define borders, i.e. maximum and minimum values
        bounds = ((1e-8, 24-1e-8), (1e-8, 24-1e-8), (1e-8, 24-1e-8), (1e-8, 24-1e-8))

        # d. Crete result thorugh Nelder-Meld method
        result = minimize(obj, x_guess, method='Nelder-Mead', bounds=bounds)

        opt.LM = result.x[0]
        opt.LF = result.x[1]
        opt.HM = result.x[2]
        opt.HF = result.x[3]

        # e. print
        if do_print:
            for k, v in opt.__dict__.items():
                print(f"{k} = {v:6.4f}")

        return opt

    def solve_wF_vec(self, discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        # a. loop through wF_vec
        for i, wF in enumerate(par.wF_vec):
            par.wF = wF

            if discrete:
                opt = self.solve_discrete()
            else:
                opt = self.solve_continuous()

            sol.LM_vec[i] = opt.LM
            sol.HM_vec[i] = opt.HM
            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF
    
    def estimate(self):
        """ estimate alpha and sigma for each value of wF """

        par = self.par
        sol = self.sol

        # Create a list to store optimized alpha and sigma values for each wF
        optimized_alpha_sigma = []

        # Define the objective function
        def obj(x):
            par.alpha, par.sigma = x
            self.solve_wF_vec()
            self.run_regression()
            return (sol.beta0 - par.beta0_target) ** 2 + (sol.beta1 - par.beta1_target) ** 2

        # Iterate through wF values
        for wF in par.wF_vec:
            par.wF = wF

            # Initial guess and bounds
            x0 = [par.alpha, par.sigma]
            bounds = [(0.01, 1), (0.01, 2)]

            # Minimize the objective function
            res = minimize(obj, x0, bounds=bounds)

            # Store the optimized alpha and sigma values
            optimized_alpha_sigma.append(res.x)

        return optimized_alpha_sigma

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        # a. data
        x = par.wF_vec
        y = sol.HF_vec / (sol.HF_vec + sol.HM_vec)

        # b. regress y on x
        X = np.column_stack((np.ones(x.size), x))
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        sol.beta0 = beta[0]
        sol.beta1 = beta[1]

    def minimize_func(self):
        """ minimize the objective function """

        # a. define the objective function
        def obj(x):
            alpha, sigma = x
            self.par.alpha = alpha
            self.par.sigma = sigma

            self.solve_wF_vec()

            self.run_regression()

            diff_beta0 = (self.sol.beta0 - self.par.beta0_target) ** 2
            diff_beta1 = (self.sol.beta1 - self.par.beta1_target) ** 2

            return diff_beta0 + diff_beta1

        # b. initial guess for alpha and sigma
        x0 = [self.par.alpha, self.par.sigma]

        # c. bounds for alpha and sigma
        bounds = [(0.01, 0.99), (0.01, 1.99)]

        # d. minimize the objective function
        result = minimize(obj, x0, method='Nelder-Mead', bounds=bounds)

        # e. update alpha and sigma with the optimized values
        self.par.alpha, self.par.sigma = result.x