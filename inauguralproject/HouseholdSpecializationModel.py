from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy import optimize
from scipy import stats

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

    def solve_continuous(self,do_print=False): # Used copilot and chatGBT to solve this part
        """ solve model continously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
    
        # a. objective function 
        def obj(x):
            LM, HM, LF, HF = x
            return - self.calc_utility(LM, HM, LF, HF)
    
        #b. Constraints and Bounds (to minimize)
        def constraints(x):
            LM, HM, LF, HF = x
            return [24 - LM-HM, 24 -LF-HF]
    

        constraints = ({'type': 'ineq', 'fun':constraints}) 
        bounds = ((0,24), (0,24), (0,24), (0,24))

        initial_guess = [6,6,6,6]

        #c. Solver 
        solution = optimize.minimize(obj, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints, tol = 0.000000001)

        opt.LM = solution.x[0]
        opt.HM = solution.x[1]
        opt.LF = solution.x[2]
        opt.HF = solution.x[3]
        
        return opt

    def solve_wF_vec(self, discrete=False): # Used copilot and chatGBT to solve this part 
        """ solve model for vector of female wages"""

        sol = self.sol
        par = self.par

        results = np.zeros(par.wF_vec.size)

        for i, w_F in enumerate(par.wF_vec):
            par.wF = w_F
            if discrete:
                opt = self.solve_discrete()
            else:
                opt = self.solve_continuous()
            if opt is not None:
                sol.LM_vec[i], sol.HM_vec[i], sol.LF_vec[i], sol.HF_vec[i] = opt.LM, opt.HM, opt.LF, opt.HF

        sol.result = results

        sol.LM_vec[i] = opt.LM
        sol.HM_vec[i] = opt.HM
        sol.LF_vec[i] = opt.LF
        sol.HF_vec[i] = opt.HF

        return results, sol

    # Defining the regression method        
    def run_regression(self):
        """ run regression """
        
        #Setting up parameters
        par = self.par
        sol = self.sol
        
        #Running regression
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        return sol

    def estimate(self, alpha=None, sigma=None):
        """Estimate alpha and beta through regression"""
        par = self.par
        sol = self.sol

        def obj(q):
            par.alpha, par.sigma = q

            self.solve_wF_vec()
            self.run_regression()

            err =  (par.beta0_target - sol.beta0)**2 + (par.beta1_target-sol.beta1)**2
            return err


        bounds = [(0, 0.99),(0.01, 2)]
        initial_guess = (0.5, 1)

        reg_opt = optimize.minimize(obj, initial_guess, method='Nelder-Mead', bounds=bounds, tol = 0.000000001)


        alpha_hat = reg_opt.x[0]
        sigma_hat = reg_opt.x[1]

        print (f'Hereby: \n Beta0_hat ={sol.beta0:.2f} \n Beta1_hat ={sol.beta1:.2f}')

        print(f'Hereby: \n alpha = {alpha_hat:.2f} \n sigma = {sigma_hat:.2f}')

    def calc_utility_extension(self, LM, HM, LF, HF):
            """ calculate utility with savings and investment """

            par = self.par
            sol = self.sol

            # Calculate individual market incomes
            IM = par.wM * LM + par.wF * LF
            IF = par.wM * HM + par.wF * HF

            # Calculate total household income
            income = IM + IF

            # Calculate savings and investments
            savings = 0.142 * income
            investment = 0.098 * income

            # Calculate consumption of market goods
            C = (1 - 0.142 - 0.098) * income

            # Calculate home production
            if par.sigma == 1:
                H = HM ** (1 - par.alpha) * HF ** par.alpha
            elif par.sigma == 0:
                H = np.minimum(HM, HF)
            else:
                H = (
                    (1 - par.alpha) * HM ** ((par.sigma - 1) / par.sigma)
                    + par.alpha * HF ** ((par.sigma - 1) / par.sigma)
                ) ** (par.sigma / (par.sigma - 1))

            # Calculate total consumption utility
            Q = C ** par.omega * H ** (1 - par.omega)
            utility = np.fmax(Q, 1e-8) ** (1 - par.rho) / (1 - par.rho)

            # Calculate disutility of work
            epsilon_ = 1 + 1 / par.epsilon
            TM = LM + HM
            TF = LF + HF
            disutility = par.nu * (
                TM ** epsilon_ / epsilon_ + TF ** epsilon_ / epsilon_
            )

            return utility - disutility