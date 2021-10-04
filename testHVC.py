import random

import matplotlib.pyplot as plt
import math
import numpy as np
import HVC
from Solution import Solution
from Population import Population


def evaluate(x):
    """:returns this function creates a multi modal test problem"""
    # five is the standard
    NUMBER_OF_NICHES = 5
    assert (x[0] >= 0) and (x[0] <= 1)
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

def test_hvc(parameter_upper_limits, parameter_lower_limits):
    sols = []
    for i in range(30):
        s = Solution([random.random()])
        s.f = evaluate(s.param)
        sols.append(s)
    pop = Population(sols)
    clusters = hillvalleyclustering(pop, 1, evaluate, [1], [0])
    plotBasic()
    cluster_indexes = []
    x = []
    f = []
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

    plt.title("Testing Hill Valley Clustering")
    # plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.show()

#Original function version
def hillvalleyclustering(population, numberofparameters, evaluator, parameter_upper_limits, parameter_lower_limits):
    # parameter_upper_limits = [1] # TODO add
    # parameter_lower_limits = [0]
    scaled_search_volume = 1
    for p in range(numberofparameters):
        scaled_search_volume *= pow(parameter_upper_limits[p] - parameter_lower_limits[p], 1 / numberofparameters)
    average_edge_length = scaled_search_volume * pow(population.size, -1.0 / numberofparameters)

    test_points = []
    cluster_index_of_test_points = []
    clustering_max_number_of_neighbours = numberofparameters + 1
    cluster_index = [-1] * population.size
    cluster_index[0] = 0
    number_of_clusters = 1
    # Generate nearest better tree
    population.order()
    edges = []
    for i in range(population.size):
        dist = np.zeros(i)
        nearest_dist = 1e308
        furthest_dist = 0
        nearest_better = 0
        furthest_better = 0
        for j in range(i - 1):
            dist[j] = population.solutions[i].param_distance(population.solutions[j])
            if dist[j] < nearest_dist:
                nearest_dist = dist[j]
                nearest_better = j
            if dist[j] > furthest_dist:
                furthest_dist = dist[j]
                furthest_better = j

        test_points_for_curr_sol = []
        edge_added: bool = False
        # line 1073 in hillvallea.cpp
        does_not_belong_to = [-1] * clustering_max_number_of_neighbours
        for j in range(min(i, clustering_max_number_of_neighbours)):
            if (j > 0):
                old_nearest_better = nearest_better
                nearest_better = furthest_better

                for k in range(i - 1): # TODO check orginingal k < i
                    if (dist[k] > dist[old_nearest_better] and dist[k] < dist[nearest_better]):
                        nearest_better = k

            skip_neighbour = False
            for k in range(len(does_not_belong_to)):
                if (does_not_belong_to[k] == cluster_index[nearest_better]):
                    skip_neighbour = True
                    break

            if skip_neighbour:
                continue


            force_accept = False
            max_number_of_trial_solutions = 1 + (int(dist[nearest_better] / average_edge_length))
            new_test_points = []
            # print("doing trials : ", max_number_of_trial_solutions)

            if i > (0.5 * population.size) and max_number_of_trial_solutions == 1:
                force_accept = True

            if force_accept or HVC.hillvalleytest(evaluator, population.solutions[i], population.solutions[nearest_better], max_number_of_trial_solutions, new_test_points):
                cluster_index[i] = cluster_index[nearest_better]
                edge_added = True

                for t in new_test_points:
                    test_points.append(t)
                    cluster_index_of_test_points.append(cluster_index[nearest_better])
                break
            else:
                does_not_belong_to[j] = cluster_index[nearest_better]

                if len(new_test_points) > 0:
                    for k in range(len(new_test_points)-1):
                        test_points_for_curr_sol.append(new_test_points[k])

        if not edge_added:
            cluster_index[i] = number_of_clusters
            number_of_clusters += 1

            for t in test_points_for_curr_sol:
                test_points.append(t)
                cluster_index_of_test_points.append(cluster_index[i])

    # Generate clusters
    # population for each cluster
    candidate_clusters = [[] for i in range(number_of_clusters)]
    cluster_active = [True] * number_of_clusters
    # print("solutions: ", population.solutions)
    # print("cluster index: ", cluster_index)

    for i in range(len(cluster_index)):
        candidate_clusters[cluster_index[i]].append(population.solutions[i])
        population.solutions[i].cluster_number = int(cluster_index[i])

        if len(candidate_clusters[cluster_index[i]]) == 1 and population.solutions[i].elite:
            cluster_active[cluster_index[i]] = False

    # TODO test points here
    for i in range(len(test_points)):
        candidate_clusters[cluster_index_of_test_points[i]].append(test_points[i])

    clusters = []
    for i in range(len(candidate_clusters)):
        if (cluster_active[i]):
            clusters.append(candidate_clusters[i])
    clusters = [x for x in clusters if x != []]
    # print("final clusters: ", clusters)
    return clusters

if __name__ == "__main__":
    # plotBasic()
    # plt.show()
    # Testing
    # test_hillvalleytest()

    test_hvc(parameter_upper_limits = [1], parameter_lower_limits = [0])
