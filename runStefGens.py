import os
import subprocess

import numpy as np
from pymoo.factory import get_performance_indicator
from pymoo.problems.multi.sympart import SYMPART
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.optimize import minimize
from pymoo.model.result import Result


def getnumberofGenerations(fpath, run):
    my_path = os.path.abspath(os.path.dirname(__file__))  # generationsstatistics_SYMPART1_1.txt
    # "teststatistics_SYMPART1_1.txt"
    print(my_path)
    path = os.path.join(my_path, "./res/SSUF3/SSUF3" + str(run) + "statistics_SSUF3_123.txt")
    f = open(path)
    stats = list(f)
    if len(stats) == 2:
        stats = stats[1]
        start = stats.find("[") + 1
        end = stats.find("]")
        substring = stats[start:end]
        subgen = substring.split()[1]
        print("Number of generations= ", subgen)
        subgen = int(subgen)
        return subgen
    else:
        raise RuntimeError("len of stats file not 2")
        # TODO add for stats files without comment line


def read_results(filepath):
    my_path = os.path.abspath(os.path.dirname(__file__))
    fpath = "./res/SSUF3/" + filepath
    path = os.path.join(my_path,
                        fpath)
    f = open(path)
    results = list(f)
    print("READING: ", filepath)
    invals = []
    outvals = []
    ranks = []
    for l in results:
        # print(l)
        vals = list(map(float, l.split()))
        inval = vals[:2]
        # print(vals)
        # print("Inputs= ", inval)
        outval = vals[2:4]
        # print("Outputs= ", outval)
        rank = vals[-1]
        # print("rank= ", rank)
        # print()
        # out = problem.evaluate(inval, return_values_of=["F"])
        # print("input: ", inval, " outputs: ", out)
        invals.append(inval)
        outvals.append(outval)
        ranks.append(rank)
    return invals, outvals, ranks


from pymoo.performance_indicator.distance_indicator import DistanceIndicator, euclidean_distance, modified_distance


class IGDX(DistanceIndicator):
    def __init__(self, pf, **kwargs):
        super().__init__(pf, euclidean_distance, 1, **kwargs)


def calculate_indicators(pf, ps, outvalArr, invalArr):
    igd = get_performance_indicator("igd", pf)
    igd_val = igd.calc(outvalArr)
    # print("IGD", igd_val)
    igdx = IGDX(ps)
    igdx_val = igdx.calc(invalArr)
    # print("IGDX", igdx_val)
    hv = get_performance_indicator("hv", pf)
    hv_val = hv.calc(outvalArr)
    # print("Hypervolume", hv_val)
    return igd_val, hv_val, igdx_val


# res.F = np.asarray(np.append(res.F, outvalArr, axis=0))
# res.X = np.asarray(np.append(res.X, invalArr, axis=0))
#
# ndsIndex = find_non_dominated(res.F)
# res.X = np.asarray([res.X[i] for i in ndsIndex])
# res.F = np.asarray([res.F[i] for i in ndsIndex])
def addToArchive(objective, decision, outvalArr, invalArr):
    objective = objective.tolist()
    decision = decision.tolist()
    # print(objective)
    for i, solution in enumerate(outvalArr):
        # print(solution)
        # print(objective)
        # print(solution.tolist())
        sol = solution.tolist()
        if not sol in objective:
            # print("hvjvhv")
            objective.append(sol)
            decision.append(invalArr[i])  # ], axis=0)

    F = np.asarray(objective)
    X = np.asarray(decision)
    # print(len(F))
    ndsIndex = find_non_dominated(F)
    # print(ndsIndex)
    X = np.asarray([X[i] for i in ndsIndex])
    F = np.asarray([F[i] for i in ndsIndex])
    return F, X


if __name__ == '__main__':
    # Benchmark Settings
    number_of_runs = 31
    max_fevals = 1000000
    problems_list = [10, 12, 13, 14, 15, 16]
    problem_index = str(11)
    PROBLEM_NAME = "TwoOnOne2"
    problem_variables = [2, 2, 2, 2, 2, 2]
    number_of_parameters = str(2)
    lopt = 0
    popsize = str(-1);
    # standard parameter settings
    lower_user_range = str(-20)
    upper_user_range = str(20)
    maximum_number_of_populations = str(1)
    elitist_archive_size_target = str(1000)
    appr_size_target = str(100)
    maximum_number_of_evaluations = str(max_fevals)
    vtr = str(1e-8)
    maximum_number_of_seconds = str(40000)
    number_of_subgenerations_per_population_factor = str(2)
    random_seed_str = str(123)
    #
    # problem = SYMPART()
    # pf = problem._calc_pareto_front(5000)
    # ps = problem._calc_pareto_set(5000)

    # write_directory = "./res/SSUF/"
    src_exe = "../mohvea/MOHillVallEA/MOHillVallEA/mo-hillvallea.app"
    hv_reps = []
    igd_reps = []
    igdx_reps = []
    for rep in range(number_of_runs):
        write_directory = "./res/TwoOnOne2/TwoOnOne2" + str(rep)
        # RUNNING STEFS from here
        processes = set()
        #
        # opt_proc = (subprocess.Popen(
        #     [src_exe, "-v"]))
        opt_proc = (subprocess.Popen(
            [src_exe, "-w", "-s", problem_index, number_of_parameters, str(lopt), lower_user_range,
             upper_user_range, maximum_number_of_populations, popsize, elitist_archive_size_target, appr_size_target,
             maximum_number_of_evaluations, vtr, maximum_number_of_seconds,
             number_of_subgenerations_per_population_factor,
             random_seed_str, write_directory], shell=False))
        print(opt_proc)
        # out, err = opt_proc.communicate()
        # errcode = opt_proc.returncode
        # print("OUT ",out)
        # print("ERR ", errcode)

        # RUNNING STEFS until here

    #
    #     n_gen = getnumberofGenerations("write_directory", rep)-1
    #     # n_gen = 100 # TODO change to above
    #     print(n_gen, " generations")
    #     IGD_vals = []
    #     IGDX_vals = []
    #     HV_vals = []
    #     generations = list(range(n_gen-1))
    #     res = Result()
    #     res.problem = problem
    #     res.F = np.asarray([[20, 20]])
    #     res.X = np.asarray([[0, 0]])
    #     for gen in generations:
    #         fname = "SSUF" + str(rep) + "approximation_set_generation_SSUF1_123_" + str(gen).zfill(5) + ".txt"
    #         invals, outvals, ranks = read_results(fname)
    #         outvalArr = np.asarray(outvals)
    #         invalArr = np.asarray(invals)
    #
    #
    #
    #         res.F, res.X = addToArchive(res.F, res.X, outvalArr, invalArr)
    #         # print(len(res.F))
    #         igd_val, hv_val, igdx_val = calculate_indicators(pf, ps, res.F, res.X)
    #         IGD_vals.append(igd_val)
    #         IGDX_vals.append(igdx_val)
    #         HV_vals.append(hv_val)
    #     # TODO include final value
    #     generations.append(n_gen)
    #     fname = "SSUF" + str(rep) + "approximation_set_generation_SSUF1_123_" + "final" + ".txt"
    #     invals, outvals, ranks = read_results(fname)
    #     outvalArr = np.asarray(outvals)
    #     invalArr = np.asarray(invals)
    #
    #     # res.F = np.asarray(np.append(res.F, outvalArr, axis=0))
    #     # res.X = np.asarray(np.append(res.X, invalArr, axis=0))
    #     # ndsIndex = find_non_dominated(res.F)
    #     # res.X = np.asarray([res.X[i] for i in ndsIndex])
    #     # res.F = np.asarray([res.F[i] for i in ndsIndex])
    #     res.F, res.X = addToArchive(res.F, res.X, outvalArr, invalArr)
    #
    #     igd_val, hv_val, igdx_val = calculate_indicators(pf, ps, res.F, res.X)
    #     IGD_vals.append(igd_val)
    #     IGDX_vals.append(igdx_val)
    #     HV_vals.append(hv_val)
    #
    #     igd_reps.append(IGD_vals)
    #     igdx_reps.append(IGDX_vals)
    #     hv_reps.append(HV_vals)
    #
    # igd_reps = np.mean(igd_reps, axis=0)
    # igdx_reps = np.mean(igdx_reps, axis=0)
    # hv_reps = np.mean(hv_reps, axis=0)
    # print("Hv: ",hv_reps)
    # print("igd: ", igd_reps)
    # print("igdx: ", igdx_reps)
    # plt.plot(generations, hv_reps, label="Hypervolume")
    # plt.title("HyperVolume")
    # plt.xlabel("Generation")
    # plt.ylabel("Hypervolume")
    # plt.show()
    # plt.plot(generations, igdx_reps, label="IGDX")
    # plt.title("IGDX")
    # plt.xlabel("Generation")
    # plt.ylabel("IGDX value")
    # plt.show()
    # plt.plot(generations, igd_reps, label="IGD")
    # plt.title("IGD")
    # plt.xlabel("Generation")
    # plt.ylabel("IGD value")
    # plt.show()
