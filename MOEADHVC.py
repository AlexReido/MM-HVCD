# Imports ===============================================================================================

# from pymoo.algorithms.moead import MOEAD
from pymoo.factory import get_problem, get_visualization, get_reference_directions, get_performance_indicator
# from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.indicators.distance_indicator import DistanceIndicator, euclidean_distance
# from pymoo.performance_indicator.distance_indicator import DistanceIndicator, euclidean_distance
from pymoo.problems.multi.sympart import SYMPART
from pymoo.factory import get_sampling
# from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.interface import sample
# from scipy import rand
import random
# from pymoo.problems.multi.omnitest import OmniTest
# import matplotlib

from Population import Population
import numpy as np
from HVC import hillvalleyclustering, HVC
from LocalOpt import LocalOptimizer
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from Archive import Archive
from Solution import Solution
from pymoo.core.population import Population as pymooPop
# from pymoo.model.population import Population as pymooPop
from pymoo.visualization.scatter import Scatter
from matplotlib import pyplot as plt
import matplotlib.cm as cm


class IGDX(DistanceIndicator):
    def __init__(self, pf, **kwargs):
        super().__init__(pf, euclidean_distance, 1, **kwargs)

def calculate_indicators(pf, ps, outvalArr, invalArr):
    igd = get_performance_indicator("igd", pf)
    igd_val = igd.do(outvalArr)
    # print("IGD", igd_val)
    igdx = IGDX(ps)
    igdx_val = igdx.do(invalArr)
    # print("IGDX", igdx_val)
    outvalArr = outvalArr[:5]
    hv = get_performance_indicator("hv", ref_point=np.array([10, 10]))
    print("out vals ===", outvalArr)
    hv_val = hv.do(outvalArr)
    # print("Hypervolume", hv_val)
    return igd_val, hv_val, igdx_val

class Evaluator():
    # global fevals
    def __init__(self, vector, func):
        self.vector = vector
        self.func = func

    def eval(self, X):
        # print("X ==========", X)
        # print("evals ==========", X.shape)

        # print("F ====", (self.func(X) * self.vector)[1])
        # Always called with t
        global fevals
        fevals += 1
        return (self.func(X) * self.vector)[1]


# simple binary tournament for a single-objective algorithm
def binary_tournament(pop, n_selected):
    n_competitors = 2

    # the result this function returns
    S = [-1] * n_selected  # , -1, dtype=int)
    sample = np.random.choice(pop, size=(n_selected, n_competitors), replace=False)
    # now do all the tournaments
    # print("SAMP:D", sample)
    for i in range(n_selected):
        if sample[i][0].f < sample[i][1].f:
            S[i] = sample[i][0]
        else:
            S[i] = sample[i][1]

    return S


def crossover(a, b):
    x  = random.random()
    return (x * (b-a)) + a

def mutation(S, prob_xu, prob_xl):
    newS = np.zeros(S.shape)
    for i, s in enumerate(S):
        newS[i] = np.random.normal(s, (0.05 * (prob_xu - prob_xl)))
        if np.any(np.less(newS[i], prob_xl)) or np.any(np.less(prob_xu, newS[i])):
            X = random.random()
            X *= (prob.xu - prob.xl)
            X += prob.xl
            newS[i] = X
    return newS




def initialisePopulation(vectors, problem, pop_size=100):
    # Sample initial population
    sampling = get_sampling('real_lhs')

    # Scale to problem bounds
    X = sample(sampling, pop_size, problem.n_var)
    X *= (prob.xu - prob.xl)
    X += prob.xl
    y = problem.evaluate(X)
    P = [0] * (len(vectors))
    evals = [0] * (len(vectors))
    # elite archive
    E = [0] * (len(vectors))
    for i, vector in enumerate(vectors):
        # print(X)
        solutions = [Solution(sol) for sol in X]
        pop = Population(solutions)
        P[i] = pop
        # P[i].setObj(vector * y)
        evaluator = Evaluator(vector, problem.evaluate)
        evals[i] = evaluator.eval
        E[i] = []
    return P, evals, E


def optimiseCluster(c, evaluator, Clusters, i, j):
    local_opt = LocalOptimizer(Population(c), evaluator)
    res = local_opt.run_opt()
    sol = Solution(res[0])
    sol.f = res[1]
    c.append(sol)
    Clusters[i].archives[j].updateArchive(sol)


def MOEADHVC(problem):
    # Get weight vectors
    vectors = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
    # crossover = SimulatedBinaryCrossover(eta=20)
    # mutation = PolynomialMutation(prob=None, eta=20)

    pop_size = 100
    P, evaluators, E = initialisePopulation(vectors, problem, pop_size)
    # fevals += pop_size
    termination = 500000
    HVCS = []
    gen = 0
    for i, vector in enumerate(vectors):
        hvc = HVC(problem.n_var, evaluators[i], problem.xu, problem.xl)
        hvc.init_clusters(P[i])
        HVCS.append(hvc)
    current_best_f = 1e28
    while fevals < termination:
        print("another gen")
        for i, vector in enumerate(vectors):
            # print(HVCS[i].population.solutions)
            pymoo = pymooPop.new()
            S1 = binary_tournament(HVCS[i].population.solutions, 5)
            S2 = binary_tournament(HVCS[i].population.solutions, 5)
            S1 = np.asarray([s.param for s in S1])
            S2 = np.asarray([s.param for s in S2])
            # print(S)
            # parents = np.random.permutation(S.size)[:crossover.n_parents]
            S = crossover(S1, S2)
            # print("crossed", S)
            # cross = crossover.do(problem, pymoo, S)
            new = mutation(S, problem.xu, problem.xl)
            new = Population([Solution(params=p) for p in new])
            HVCS[i].add_to_clusters(new)
            for j, c in enumerate(HVCS[i].clusters):
                optimiseCluster(c, evaluators[i], HVCS, i, j)
        # final_pop = []
        for i, vector in enumerate(vectors):
            for j, a in enumerate(HVCS[i].archives):
                if a.archive[0].f < current_best_f:
                    current_best_f = a.archive[0].f
                    # final_pop = final_pop + a.archive # TODO below
                elif a.archive[0].f > np.any(current_best_f + (0.2) *(problem.xu - problem.xl)):
                    # HVCS[i].archives[j] = None
                    # HVCS[i].clusters.pop(j)
                    pass
                # else: # TODO overtime analysis
                #     final_pop = final_pop + a.archive


    final_pop = []
    for i, vector in enumerate(vectors):
        for j, a in enumerate(HVCS[i].archives):
            final_pop = final_pop + a.archive
    return final_pop


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

if __name__ == "__main__":
    global fevals
    fevals = 0
    prob = SYMPART()
    # print(prob.xl)
    # print(prob.xu)
    # print()
    # print(prob.n_obj)
    final_pop = MOEADHVC(prob)
    print(final_pop)
    print("Size: ", len(final_pop))
    pf = prob._calc_pareto_front(5000)
    ps = prob._calc_pareto_set(5000)
    # Y = np.asarray([s.f for s in final_pop])
    X = np.asarray([s.param for s in final_pop])
    Y = prob.evaluate(X, return_values_of=["F"])
    cluster = [s.cluster_number for s in final_pop]
    igd_val, hv_val, igdx_val = calculate_indicators(pf, ps, Y, X)
    print("IGD", igd_val)
    print("IGDX", igdx_val)
    print("Hypervolume", hv_val)

    from matplotlib import cm, colors

    norm = colors.Normalize(vmin=0.0, vmax=20.0, clip=False)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Oranges)
    print(cluster)
    colours = [mapper.to_rgba(c) for c in cluster]
    Scatter(title="SYMPART decision", xlabel="$x_1$", ylabel="$x_2$").add(ps).add(X, color=colours).show()
    Scatter(title="SYMPART Objective", xlabel="$f_1$", ylabel="$f_2$").add(pf).add(Y, color=colours).show()
    # plt.scatter(X, Y)
    print(prob.xu)
    print(prob.xl)

