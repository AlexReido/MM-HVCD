# Imports ===============================================================================================
import warnings

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
import runStefGens
from Population import Population
import numpy as np
from HVC import HVC
from LocalOpt import LocalOptimizer
from LocalOpt import getOptName
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from Archive import Archive
from Solution import Solution
from pymoo.core.population import Population as pymooPop
# from pymoo.model.population import Population as pymooPop
from pymoo.visualization.scatter import Scatter
from matplotlib import pyplot as plt
import matplotlib.cm as cm


class IGDX(DistanceIndicator):
    def __init__(self, ps, **kwargs):
        super().__init__(ps, euclidean_distance, 1, **kwargs)


def calculate_indicators(pf, ps, outvalArr, invalArr, reference_point):
    igd = get_performance_indicator("igd", pf)
    igd_val = igd.do(outvalArr)
    # print("IGD", igd_val)
    # print("pareto set: ", ps)
    # print("invals : ", invalArr)
    igdx = IGDX(ps)
    igdx_val = igdx.do(invalArr)
    # print("IGDX", igdx_val)
    # outvalArr = outvalArr[:5]
    hv = get_performance_indicator("hv", ref_point=reference_point)
    # print("out vals ===", outvalArr)
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
        # TODO do this for a population
        # print("evals ==========", X.shape)

        # print("F ====", (self.func(X) * self.vector)[1])
        # Always called with t
        global fevals
        fevals += 1
        return np.sum(self.func(X) * self.vector)


# simple binary tournament for a single-objective algorithm
def binary_tournament(pop, n_selected, random_seed):
    if random_seed:
        np.random.seed(0)
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


def crossover(a, b, random_seed):
    if random_seed:
        np.random.seed(0)
    x = random.random()
    return (x * (b - a)) + a


def mutation(S, prob_xu, prob_xl, random_seed):
    if random_seed:
        np.random.seed(0)
    newS = np.zeros(S.shape)
    for i, s in enumerate(S):
        newS[i] = np.random.normal(s, (0.1 * (prob_xu - prob_xl)))
        if np.any(np.less(newS[i], prob_xl)) or np.any(np.less(prob_xu, newS[i])):
            X = random.random()
            X *= (prob_xu - prob_xl)
            X += prob_xl
            newS[i] = X
    return newS


def initialisePopulation(vectors, problem, pop_size=100):
    # Sample initial population
    sampling = get_sampling('real_lhs')

    # Scale to problem bounds
    X = sample(sampling, pop_size, problem.n_var)
    X *= (problem.xu - problem.xl)
    X += problem.xl
    P = [0] * (len(vectors))
    evals = [0] * (len(vectors))
    # elite archive
    E = [0] * (len(vectors))
    for i, vector in enumerate(vectors):
        # print("i: ", i, " sample X ", X)
        solutions = [Solution(sol) for sol in X]
        pop = Population(solutions)
        P[i] = pop
        # P[i].setObj(vector * y)
        evaluator = Evaluator(vector, problem.evaluate)
        evals[i] = evaluator.eval
        E[i] = []
    return P, evals, E


from scipy.optimize import Bounds


def optimiseCluster(c, evaluator, Clusters, i, j, opt_name, bounds):
    """
    This is called for each cluster in each weight vector
    :param c:
    :param evaluator:
    :param Clusters:
    :param i: The weight vector
    :param j: The cluster number
    :return:
    """
    # print("cluster ", c)
    # for s in c:
    #     print(s.cluster_number, end="")
    local_opt = LocalOptimizer(Population(c), evaluator, opt_name, bounds)
    res = local_opt.run_opt()
    sol = Solution(res[0])
    sol.f = res[1]
    c.append(sol)
    sol.cluster_number = c[0].cluster_number
    # print(i, j)
    # print(Clusters[i])
    # print(Clusters[i].archives[j])
    # try:
    Clusters[i].archives[j].updateArchive(sol)
    # except:
    #     print("EXCEPTION")
    #     print("i: ", i, " j: ", j)
    #     print("len of clusters = ", len(Clusters[i].clusters))
    #     print("len of archive = ", len(Clusters[i].archives))


def getbest100(archives):
    """
    This funtion first populates a list with 100 solutions from the archives.
    Next this list is sorted with the best solutions last in the list.
    The remaining solutions in the archives are looped over and if better than
    the current last solution they replace it before the list is resorted.
    :param archives:
    :return:
    """
    the_best = []
    for i, a in enumerate(archives):
        a_index = i
        for j, sol in enumerate(a.archive):
            sol_index = j
            the_best.append(sol)
            if (len(the_best) >= 100):
                break
        if (len(the_best) >= 100):
            break

    # print("lenth: ", len(the_best))
    the_best = sorted(the_best, key=lambda x: x.f)
    while a_index < len(archives):
        a = archives[a_index].archive
        while sol_index < len(a):
            sol = a[sol_index]
            if sol.f < the_best[-1].f:
                the_best.pop(-1)
                the_best.append(sol)
                the_best = sorted(the_best, key=lambda x: x.f)
            sol_index += 1
        # print(len(the_best))
        a_index += 1
    # print("final length: ", len(the_best))
    return the_best


def MOEADHVC(problem, opt_name, overtime: bool, random_seed: bool, reference_point):
    if random_seed:
        np.random.seed(0)
    # Get weight vectors
    vectors = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
    # crossover = SimulatedBinaryCrossover(eta=20)
    # mutation = PolynomialMutation(prob=None, eta=20)
    bounds = Bounds(problem.xl, problem.xu)
    # print(vectors)
    pop_size = 100
    generation_size = 10  # divisible by 2
    P, evaluators, E = initialisePopulation(vectors, problem, pop_size)
    # fevals += pop_size
    termination = 60000
    HVCS = []
    gen = 0
    if overtime:
        overtime_results = []
        pf = problem._calc_pareto_front(5000)
        ps = problem._calc_pareto_set(5000)
        global fevals
    for i, vector in enumerate(vectors):
        hvc = HVC(problem.n_var, evaluators[i], problem.xu, problem.xl, vector)
        hvc.init_clusters(P[i])
        HVCS.append(hvc)

    while fevals < termination:
        # print("gen", gen)
        gen += 1
        for i, hvc in enumerate(HVCS):
            S = binary_tournament(hvc.population.solutions, generation_size, random_seed)
            S = np.asarray([s.param for s in S])
            S = crossover(S[:int(generation_size / 2)], S[int(generation_size / 2):], random_seed)
            new = mutation(S, problem.xu, problem.xl, random_seed)
            new = Population([Solution(params=p) for p in new])
            hvc.add_to_clusters(new)

            for j, c in enumerate(hvc.clusters):
                optimiseCluster(c, evaluators[i], HVCS, i, j, opt_name, bounds)


        for i, hvc in enumerate(HVCS):
            for j, a in enumerate(hvc.archives):
                if a.archive[0].f < hvc.current_best_f:
                    hvc.current_best_f = a.archive[0].f
                    # final_pop = final_pop + a.archive # TODO below
                elif a.archive[0].f > np.any(hvc.current_best_f + ((0.2) * (problem.xu - problem.xl))):
                    # HVCS[i].clusters.pop(j)
                    # HVCS[i].archives.pop(j)  # [j] = None
                    pass

        if overtime:
            final_pop = []
            removed_count = 0
            for i, hvc in enumerate(HVCS):
                vector_pop = getbest100(hvc.archives)
                # plot_vector(vector_pop, problem, i)
                final_pop = final_pop + vector_pop
            igd_val, hv_val, igdx_val = interpret_results(problem, " ", final_pop, reference_point, False)
            n_evals = fevals
            overtime_results.append([igd_val, hv_val, igdx_val, n_evals])
            # else: # TODO overtime analysis

    final_pop = []
    removed_count = 0
    for i, hvc in enumerate(HVCS):
        vector_pop = getbest100(hvc.archives)
        # plot_vector(vector_pop, problem, i)
        final_pop = final_pop + vector_pop

    # print("Final length [ALL] : ", len(final_pop))

    for sol in final_pop:
        if sol.cluster_number == -1:
            print(sol.cluster_number)
            raise Exception("BAD CLUSTER NUMBER")
    if overtime:
        return overtime_results, final_pop
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

def interpret_results(problem, problem_name, population, reference_point, draw_graph: bool):
    """
    This interprets the results outputting a graph as well as retuning the indicator performance values.
    :param problem: The pymoo problem object
    :param problem_name: String the name of the problem.
    :param population: List of solutions from final population
    :param reference_point: Used to calculate the hypervolume.
    :return: The inter generational distance and hypervolume values.
    """
    pf = problem._calc_pareto_front(5000)
    ps = problem._calc_pareto_set(5000)
    # Y = np.asarray([s.f for s in final_pop])
    X = np.asarray([s.param for s in population])
    Y = problem.evaluate(X, return_values_of=["F"], reference_point=reference_point)
    cluster = [s.cluster_number for s in population]

    igd_val, hv_val, igdx_val = calculate_indicators(pf, ps, Y, X, reference_point)  # , igdx_val

    # print("IGD", igd_val)
    # # print("IGDX", igdx_val)
    # print("Hypervolume", hv_val)
    if draw_graph:
        plotResults.standard_plots(problem, problem_name, X, Y, ps, pf, cluster)
    return igd_val, hv_val, igdx_val


from LocalOpt import LocalOptimizer, getOptName
import GetProblem
import plotResults


def run_MOEADHVC(random_seed: bool, problems: list, local_opts: list, overtime: bool, graph: bool, repetitions: int):
    if random_seed:
        np.random.seed(0)
    hv = []
    igd = []
    print("Algorithm run for ", repetitions, " repetitions.")
    # list[ (problem_name, list[ (opt_name, list[ rep1_hv, rep2_hv, ... ]), (opt2_name, list[hv_vals]) ]), (problem2_name, list[],]
    for i, problem in enumerate(problems):
        # hv.append((problem, []))
        prob, reference_point = GetProblem.getProblem(problem)
        for j, local_opt in enumerate(local_opts):
            # hv[i][1].append((local_opt))
            print("Running MOEADHVC on the problem ", problem, " with the local opt ", local_opt, " to ", eval)
            # print("Number of variables: ", prob.n_var)
            # print("Number of objectives: ", prob.n_obj)
            hv_vals = []
            igd_vals = []
            igdx_vals = []
            for i in range(repetitions):
                global fevals
                fevals = 0
                # print(prob.xl)
                # print(prob.xu)
                # print()
                # print(prob.n_obj)
                if overtime:
                    overtime_res, final_pop = MOEADHVC(prob, local_opt, overtime, random_seed, reference_point)
                    #  igd_val, hv_val , igdx_val
                    overtime_res = np.asarray(overtime_res)
                    title = "Overtime analysis on " + problem + " with " + local_opt + " using "
                    plotResults.plot_overtime("Hyper Volume", overtime_res[:, 1], overtime_res[:, 3], (title + "Hyper Volume"))
                    plotResults.plot_overtime("IGD", overtime_res[:, 0], overtime_res[:, 3], (title + "IGD"))
                    plotResults.plot_overtime("IGDX", overtime_res[:, 2], overtime_res[:, 3], (title + "IGDX"))
                else:
                    final_pop = MOEADHVC(prob, local_opt, overtime, random_seed, reference_point)
                igd, hv, igdx = interpret_results(prob, problem, final_pop, reference_point, graph)
                hv_vals.append(hv)
                igd_vals.append(igd)
                igdx_vals.append(igdx)

            ave = sum(igd_vals) / len(igd_vals)
            igd_vals.sort()
            median = igd_vals[int((repetitions - 1) / 2)]
            print("igd - Average== ", ave, " Median== ", median)

            ave = sum(hv_vals) / len(hv_vals)
            hv_vals.sort()
            median = hv_vals[int((repetitions - 1) / 2)]
            print("hv  - Average== ", ave, " Median== ", median)

            ave = sum(igdx_vals) / len(igdx_vals)
            igdx_vals.sort()
            median = igdx_vals[int((repetitions - 1) / 2)]
            print("igdx - Average== ", ave, " Median== ", median)

            print()
            print()


if __name__ == "__main__":
    opt_name = ["Nelder-Mead"]# , "Powell", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]
    # opt_name = [getOptName(0)]
    global fevals
    fevals = 0
    # "OmniTest"  "SYMPART"  "DTLZ1"
    problems = ["SYMPART", "OmniTest"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_MOEADHVC(False, problems, opt_name, False, True, 1)

    # problem_name = "SYMPART"
    # problem, reference_point = GetProblem.getProblem(problem_name)
    # number_of_runs = 31  # 31
    #
    # # runthebenchmark(problem_name, 20, number_of_runs)
    # overtime_res, final_pop = MOEADHVC(problem, opt_name, True, True, reference_point)
    # #  igd_val, hv_val , igdx_val
    # overtime_res = np.asarray(overtime_res)
    # moeadhvc_evals =  overtime_res[:, 3]
    # moeadhvc_hv = overtime_res[:, 1]
    # moeadhvc_IGD = overtime_res[:, 0]
    # moeadhvc_IGDX = overtime_res[:, 2]
    #
    # title = "Overtime analysis on " + problem_name #+ " with " + local_opt + " using "
    # # plotResults.plot_overtime("Hyper Volume", overtime_res[:, 1], overtime_res[:, 3], (title + "Hyper Volume"))
    # # plotResults.plot_overtime("IGD", overtime_res[:, 0], overtime_res[:, 3], (title + "IGD"))
    # # plotResults.plot_overtime("IGDX", overtime_res[:, 2], overtime_res[:, 3], (title + "IGDX"))
    #
    # #
    # hv_reps, igd_reps, igdx_reps, evals = runStefGens.calculateresults(problem_name, number_of_runs, problem, reference_point, name_append="1", eval_limit=60000)
    #
    # plt.plot(evals, hv_reps, label="benchmark")
    # plt.plot(moeadhvc_evals, moeadhvc_hv, label="moeadhvc")
    # plt.title(title + "using hypervolume")
    # plt.xlabel("Number of evaluations")
    # plt.ylabel("Hyper volume")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(evals, igd_reps, label="benchmark")
    # plt.plot(moeadhvc_evals, moeadhvc_IGD, label="moeadhvc")
    # plt.title(title + "using IGD")
    # plt.xlabel("Number of evaluations")
    # plt.ylabel("IGD")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(evals, igdx_reps, label="benchmark")
    # plt.plot(moeadhvc_evals, moeadhvc_IGDX, label="moeadhvc")
    # plt.title(title + "using IGDX")
    # plt.xlabel("Number of evaluations")
    # plt.ylabel("IGDX")
    # plt.legend()
    # plt.show()

