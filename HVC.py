import Solution
import numpy as np


def hillvalleytest(evaluator, sol1: Solution, sol2: Solution, max_trials: int) -> bool:
    trialSols = []
    for i in range(max_trials):
        trialSols.append(Solution())
    hillvalleytest(evaluator, sol1, sol2, max_trials, trialSols)


def hillvalleytest(evaluator, sol1: Solution, sol2: Solution, max_trials: int, trial_sols: list(Solution)) -> bool:
    """
    The hvt checks if worse solution between them a peak is found therefore in differing clusters
    if in same cluster return true else false
    :param evaluator:
    :param problem:
    :param sol1:
    :param sol2:
    :param max_trials:
    :return:
    """
    if (sol1.elite == True and sol2.elite == True):
        return False
    nevals = 0
    # print(f)
    val1 = sol1.param
    val2 = sol2.param
    print(val1)
    print(val2)
    for k in range((max_trials)):
        # TODO multi dimension parameters
        testval = val1[0] + ((k + 1.0) / (max_trials + 1.0)) * (val2[0] - val1[0])
        testvals.append(testval)
    print(testvals)
    nevals += len(testvals)
    for t in testvals:
        # TODO columnwise?
        f = evaluator([t])
        # TODO evals ++
        # If worse solution between them a peak is found therefore in differing clusters
        if f > max(sol1.f, sol2.f):
            return False

    return True


class Edge:
    def __init__(self, start, end, length):
        self.start = start
        self.end = end
        self.length = length


def hillvalleyclustering(solutions, numberofparameters, evaluator):
    parameter_upper_limits = [1]  # TODO add
    parameter_lower_limits = [0]
    scaled_search_volume = 1
    for p in range(numberofparameters):
        scaled_search_volume *= pow(parameter_upper_limits[p] - parameter_lower_limits[p], 1 / numberofparameters)
    average_edge_length = scaled_search_volume * pow(len(solutions), -1.0 / numberofparameters)

    test_points = []
    cluster_index_of_test_points = []
    clustering_max_number_of_neighbours = numberofparameters + 1
    cluster_index = [-1] * len(solutions)
    cluster_index[0] = 0
    number_of_clusters = 1
    # Generate nearest better tree
    solutions = sorted(solutions, key=lambda x: x.f)
    edges = []
    for i in range(len(solutions)):
        dist = np.zeros(i)
        nearest_dist = 1e308
        furthest_dist = 0
        nearest_better = 0
        furthest_better = 0
        for j in range(i - 1):
            dist[j] = solutions[i].param_distance(solutions[j])
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

                for k in range(i - 1):  # TODO check orginingal k < i
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
            print("doing trials : ", max_number_of_trial_solutions)
            if (i > (0.5 * len(solutions)) and max_number_of_trial_solutions == 1):
                force_accept = True

            if (force_accept or hillvalleytest(evaluator, solutions[i], solutions[nearest_better],
                                               max_number_of_trial_solutions)):
                cluster_index[i] = cluster_index[nearest_better]
                edge_added = True
                break
                # TODO add test vals to cluster if accepted
            else:
                does_not_belong_to[j] = cluster_index[nearest_better]

        if not edge_added:
            cluster_index[i] = number_of_clusters
            number_of_clusters += 1

    # Generate clusters
    # population for each cluster
    candidate_clusters = [[] for i in range(number_of_clusters)]
    cluster_active = [True] * number_of_clusters
    print("solutions: ", solutions)
    print("cluster index: ", cluster_index)
    print("candidate clusters: ", candidate_clusters)
    for i in range(len(cluster_index)):
        candidate_clusters[cluster_index[i]].append(solutions[i])
        solutions[i].cluster_number = int(cluster_index[i])

        if len(candidate_clusters[cluster_index[i]]) == 1 and solutions[i].elite:
            cluster_active[cluster_index[i]] = False

    # TODO test points here
    clusters = []
    for i in range(len(candidate_clusters)):
        if (cluster_active[i]):
            clusters.append(candidate_clusters[i])
    return clusters
