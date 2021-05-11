from pymoo.algorithms.moead import MOEAD
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize
from pymoo.problems.multi.sympart import SYMPART
from pymoo.factory import get_sampling
from pymoo.interface import sample
from Population import Population
import numpy as np
from HVC import hillvalleyclustering
from LocalOpt import LocalOptimizer
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from Archive import Archive

class Evaluator():
    def __init__(self, vector, func):
        self.vector = vector
        self.func

    def eval(self, X):
        return self.func(X) * self.vector


def initialisePopulation(vectors, problem, pop_size = 100):
    # Sample initial population
    sampling = get_sampling('real_lhs')

    # Scale to problem bounds
    X = sample(sampling, pop_size, problem.n_var)
    X *= (prob.xu - prob.xl)
    X += prob.xl
    y = problem.evaluate(X)
    P = np.zeros(len(vectors))
    evals = np.zeros(len(vectors))
    # elite archive
    E = np.zeros(len(vectors))
    for i, vector in enumerate(vectors):
        P[i] = Population(X)
        P[i].setObj(vector * y)
        evals[i] = Evaluator(vector, problem.evaluate)
        E[i] = []
    return P, evals, E



def MOEADHVC(problem):
    # Get weight vectors
    vectors = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)

    fevals = 0
    pop_size = 100
    P, evaluators, E = initialisePopulation(vectors, problem, pop_size)
    fevals += pop_size
    termination = 1000
    while fevals < termination:
        for i, vector in enumerate(vectors):
            clusters = HVC(P[i], problem.n_var, evaluators[i], prob.xu, prob.xl)
            for c in clusters:
                E[i].append(Archive(c))
                local_opt = LocalOptimizer(Population(c), evaluators[i])
                c.append(local_opt.run_opt())

#
# # generation of the local optimiser
# local_optimizers[i].estimate_sample_parameters()
#
# local_number_of_evaluations = (int) local_optimizers[i].sample_new_population(current_cluster_size)
# number_of_evaluations += local_number_of_evaluations
#
#
# local_optimizers[i].truncation_percentage()
# #local_optimizers[i]->average_fitness_history.push_back(local_optimizers[i]->pop->average_fitness())

if __name__== "__main__":
    prob = SYMPART()
    # print(prob.xl)
    # print(prob.xu)
    # print()
    # print(prob.n_obj)
    MOEADHVC(prob)

