import sys

from LocalOpt import LocalOptimizer, getOptName
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import HVC
from Solution import Solution
from Population import Population
from HVC import HVC
from scipy.optimize import Bounds

def evaluate(x):
    """:returns this function creates a multi modal test problem"""
    # five is the standard
    NUMBER_OF_NICHES = 5
    # assert (x[0] >= 0) and (x[0] <= 1), ("x = " + str(x[0]))
    if (x[0] < 0) or (x[0] > 1):
        print("x out of bounds ", str(x))
        return sys.maxsize
    return 1.1 - math.exp(-2 * x[0]) * math.sin(NUMBER_OF_NICHES * x[0] * math.pi) ** 2

def plotBasic():
    x = np.linspace(0, 1, 300)
    y = [evaluate([z]) for z in x]
    plt.plot(x, y)

def test_hillvalleytest():
    plotBasic()
    for i in range(5):
        x1 = random.random()
        x2 = random.random()
        sol1 = Solution(x1)
        sol2 = Solution(x2)
        sol1.f = evaluate(sol1.param)
        sol2.f = evaluate(sol2.param)
        same_val = HVC.hillvalleytest(evaluate, sol1, sol2, 5)
        if same_val:
            colour = "green"
        else:
            colour = "red"
        plt.plot([sol1.param, sol2.param], [sol1.f, sol2.f], color=colour)
    plt.title("Testing the hillvalleytest")
    plt.show()

def plotLocalPops(clusters):
    plotBasic()
    for i, c in enumerate(clusters):
        # print("Cluster " + str(i)+ ":")
        cluster_indexes = []
        x = []
        f = []
        for s in c:
            cluster_indexes.append(i)
            x.append(s.param[0])
            f.append(s.f)

        plt.scatter(x,f, cmap=plt.get_cmap("tab20"), label="Cluster: " +str(i))

    plt.show()

def test_localOpt(index = 0, bounds = Bounds([0], [1])):
    np.random.seed(0)
    plotBasic()
    sols = []
    for i in range(30):
        s = Solution([random.random()])
        s.f = evaluate(s.param)
        sols.append(s)
    pop = Population(sols)
    hvc = HVC(1, evaluate, [1], [0], [1])
    hvc.init_clusters(pop)
    # clusters = HVC.hillvalleyclustering(pop, 1, evaluate, [1], [0])
    for c in hvc.clusters:
        name = getOptName(index)
        print("testing optimiser: ", name)
        local_opt = LocalOptimizer(Population(c), evaluate, name, bounds)
        res = local_opt.run_opt()
        print(res)
        plt.scatter(res[0][0], res[1], cmap=plt.get_cmap("tab20"), label="Cluster: " + str(i))
    plt.title(("Testing the local optimiser: " + name))
    # plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.show()



if __name__ == "__main__":
    bounds = Bounds([0], [1])
    for i in range(0, 6):
        test_localOpt(i, bounds)
