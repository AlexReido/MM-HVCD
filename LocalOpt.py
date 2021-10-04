from scipy.optimize import minimize
from Solution import Solution

class LocalOptimizer():
    def __init__(self, population, evaluator):
        # Population should be sorted
        population.order()
        population.truncation_percentage(0.35)
        self.pop = population
        self.old_mean = population.getMean()
        self.mean = self.old_mean
        self.evaluator = evaluator
        best = self.pop.solutions[0]
        self.number_of_params = len(best.param)


    def run_opt(self):
        res = minimize(self.evaluator, self.mean, method='Nelder-Mead', tol=1e-6, options={"maxfev":10})
        # print("Fevals = ", res.nfev) # num function evaluations
        # print("fval= ", res.fun)
        return res.x, res.fun

    def sample_new_population(self):
        pass