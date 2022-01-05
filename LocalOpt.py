from scipy.optimize import minimize
from Solution import Solution

def getOptName(index = 1):
    if index == 0:
        return "Nelder-Mead"
    elif index == 1:
        return "Powell"
    elif index == 2:
        return "L-BFGS-B"
    elif index == 3:
        return "TNC"
    elif index == 4:
        return "COBYLA"
    elif index == 5:
        return "SLSQP"



class LocalOptimizer():
    def __init__(self, population, evaluator, name, bounds):
        # Population should be sorted
        population.order()
        population.truncation_percentage(0.35)
        self.pop = population
        self.old_mean = population.getMean()
        self.mean = self.old_mean
        self.evaluator = evaluator
        best = self.pop.solutions[0]
        self.number_of_params = len(best.param)
        self.optimizer_name = name
        self.uplowBounds = bounds


    def run_opt(self):
        # 'Nelder-Mead'
        res = minimize(self.evaluator, self.mean, method=self.optimizer_name, bounds=self.uplowBounds, tol=1e-6, options={"maxfev":10})
        # print("Fevals = ", res.nfev) # num function evaluations
        # print("fval= ", res.fun)
        return res.x, res.fun

    def sample_new_population(self):
        pass