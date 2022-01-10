import numpy as np
import math
class Population():

    def __init__(self, solutions):
        self.solutions = solutions
        self.size = len(solutions)
        self.number_of_params = len(solutions[0].param)

    def setObj(self, y):
        for i, s in enumerate(self.solutions):
            s.f = y[i]

    def order(self):
        """
        Sort solutions based on fitness
        :return:
        """

        self.solutions = sorted(self.solutions, key=lambda x: x.f)

    def combine(self, other_solutions, evaluator):
        self.solutions = self.solutions + other_solutions
        self.eval(evaluator)
        self.order()
        return [self.solutions.index(sol) for sol in other_solutions]

    def truncation_percentage(self, truncation_percentage):
        """Select the truncation_percentage*population_size best solutions"""
        newSize = math.ceil(truncation_percentage * self.size)
        self.solutions = self.solutions[:newSize]

    def getMean(self):
        mean = np.zeros(self.number_of_params)
        for sol in self.solutions:
            mean += sol.param
        return mean/self.size

    def eval(self, evaluator):
        # fvals = evaluator(s.param)
        for s in self.solutions:
            s.f = evaluator(s.param)

    def improvement(self, sol):
        """ if the soltution sol is better than the best in this population"""
        if self.solutions[0].f < sol.f:
            return True
        else:
            return False

    def fill_normal_univariate(self, sample_size, problem_size):
        pass