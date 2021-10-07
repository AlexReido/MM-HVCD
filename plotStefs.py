import sys
import math
from pymoo.problems.multi.sympart import SYMPART, SYMPARTRotated
# from pymoo.algorithms.genetic_algorithm import gen
import os.path
from pymoo.factory import get_performance_indicator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getnumberofGenerations(fpath):
    my_path = os.path.abspath(os.path.dirname(__file__))  # generationsstatistics_SYMPART1_1.txt
    "teststatistics_SYMPART1_1.txt"
    path = os.path.join(my_path, "../mohvea/MOHillVallEA/MOHillVallEA/results/")
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





def readresults(filepath):
    my_path = os.path.abspath(os.path.dirname(__file__))
    fpath = "../mohvea/MOHillVallEA/MOHillVallEA/" + filepath
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
        out = problem.evaluate(inval, return_values_of=["F"])
        # print("input: ", inval, " outputs: ", out)
        invals.append(inval)
        outvals.append(outval)
        ranks.append(rank)
    return invals, outvals, ranks


def readPset(filepath):
    my_path = os.path.abspath(os.path.dirname(__file__))
    fpath = "../mohvea/MOHillVallEA/MOHillVallEA/" + filepath
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
        outval = vals[2:]
        # print("Outputs= ", outval)
        # rank = vals[-1]
        # print("rank= ", rank)
        # print()
        # out = problem.evaluate(inval, return_values_of=["F"])
        # print("input: ", inval, " outputs: ", out)
        invals.append(inval)
        outvals.append(outval)
        # ranks.append(rank)
    return invals, outvals  # , ranks


def plotPareto(fx, ranks):
    data = pd.DataFrame({"X Value": fx[:, 0], "Y Value": fx[:, 1], "Category": ranks})

    groups = data.groupby("Category")
    for name, group in groups:
        plt.plot(group["X Value"], group["Y Value"], marker="o", linestyle="", label=name)
        plt.vlines(group["X Value"], group["Y Value"], 6)
        plt.hlines(group["Y Value"], group["X Value"], 6)
    plt.legend()
    plt.show()


from pymoo.performance_indicator.distance_indicator import DistanceIndicator, euclidean_distance,modified_distance


class IGDX(DistanceIndicator):

    def __init__(self, pf, **kwargs):
        super().__init__(pf, euclidean_distance, 1, **kwargs)

class IGDXPlus(DistanceIndicator):

    def __init__(self, pf, **kwargs):
        super().__init__(pf, modified_distance, 1, **kwargs)


def calculate_indicators(pf, ps):
    igd = get_performance_indicator("igd", pf)
    print("IGD", igd.calc(outvalArr))
    igdx = IGDX(ps)
    print("IGDX", igdx.calc(invalArr))
    igdx_plus = IGDXPlus(ps)
    print("IGDX+", igdx_plus.calc(invalArr))
    hv = get_performance_indicator("hv", pf)
    print("Hypervolume", hv.calc(outvalArr))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # command used mo-hillvallea.app -v -w -s 12 2 0 -20 20 1 10 15 10 30000 0 10 2 1 .\\results\\test

    problem = SYMPART()

    # generationsapproximation_set_generation_SYMPART1_1_final.txt
    invals, outvals, ranks = readresults("results/testapproximation_set_generation_SYMPART1_1_final.txt")

    try:
        plotPareto(np.array(outvals), ranks)
    except Exception as err:
        print(err)
    invals, outvals, ranks = readresults("results/paretoSetapproximation_set_generation_SYMPART1_1_final.txt")


    # Plot the pareto front
    try:
        plotPareto(np.array(outvals), ranks)
    except Exception as err:
        print(err)
    #
    # invals, outvals, ranks = readresults("results/testpopulation_00000_generation_00066_SYMPART1_1.txt")

    # Plot the pareto front
    # try:
    #     plotPareto(np.array(outvals), ranks)
    # except Exception as err:
    #     print(err)


    pf = problem._calc_pareto_front(5000)
    ps = problem._calc_pareto_set(5000)
    print("Pareto front:")
    # print(pf[:10])
    # # print("invals")
    outvalArr = np.asarray(outvals)
    invalArr = np.asarray(invals)
    # # print(invals)
    #

    # igd = get_performance_indicator("igd", pf)
    calculate_indicators(pf, ps)
    # print("IGD", igd.calc(outvalArr))
    # igdx = IGDX(ps)
    # print("IGDX", igdx.calc(invalArr))
    # igdx_plus = IGDXPlus(ps)
    # print("IGDX+", igdx_plus.calc(invalArr))
    # print()
    print()
    print("Calculate the indicator values using c++ pareto set")
    psvals, pfvals = readPset("PSET.txt")
    cpf = np.asarray(pfvals)
    cps = np.asarray(psvals)
    calculate_indicators(cpf, cps)

    # print(invalArr[:10])



    # subgen = getnumberofGenerations("teststatistics_SYMPART1_1.txt")
    # s = (subgen - 5)
    # f = (subgen-1)
    # for i in range(s, f):
    #     n = str(i).zfill(5)
    #     fpath = "../mohvea/MOHillVallEA/MOHillVallEA/results/testapproximation_set_generation_SYMPART1_1_" + n + ".txt"
    #     path = os.path.join(my_path,fpath)
    #     f = open(path)
    #     results = list(f)
    #     invals = []
    #     for l in results:
    #         # print(l)
    #         vals = list(map(float, l.split()))
    #         inval = vals[:2]
    #         # print(vals)
    #         out = problem.evaluate(inval, return_values_of=["F"])
    #         print("input: ", inval, " outputs: ", out)
    #         invals.append(inval)
    #
    #     igd = get_performance_indicator("igd", pf)
    #     print("IGD", igd.calc(invals))
    #     igd_plus = get_performance_indicator("igd+", pf)
    #     print("IGD+", igd_plus.calc(invals))

    #
    # print()
    # path = os.path.join(my_path,"../mohvea/MOHillVallEA/MOHillVallEA/generationspareto_set_SYMPART1_1.txt")
    # f = open(path)
    # results = list(f)
    # print("generationspareto_set_SYMPART1_1")
    # print(len(results))

    # set = problem._calc_pareto_set(10)
    # X = [[0,0],[1,2], [-1,10]]
    #
    # out = {}

    #
    # print(out)
