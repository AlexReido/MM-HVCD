from Archive import Archive
from Solution import Solution
import numpy as np
from Population import Population

# def hillvalleytest(evaluator, sol1: Solution, sol2: Solution, max_trials: int) -> bool:
# trialSols = []
# # for i in range(max_trials):
# # trialSols.append(Solution())
# hillvalleytest(evaluator, sol1, sol2, max_trials, trialSols)


def hillvalleytest(evaluator, sol1: Solution, sol2: Solution, max_trials: int, trial_sols) -> bool:
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
    # print(val1)
    # print(val2)
    for k in range(max_trials):
        # TODO multi dimension parameters
        testval = val1 + ((k + 1.0) / (max_trials + 1.0)) * (val2 - val1)
        t = Solution(testval)
        trial_sols.append(t)

        nevals += 1
        # TODO columnwise?
        t.f = evaluator(t.param)
        # TODO evals ++
        # If worse solution between them a peak is found therefore in differing clusters
        if t.f > max(sol1.f, sol2.f):
            # TODO if greater than any other between them?
            return False

    return True


class Edge:
    def __init__(self, start, end, length):
        self.start = start
        self.end = end
        self.length = length


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

            if force_accept or hillvalleytest(evaluator, population.solutions[i], population.solutions[nearest_better], max_number_of_trial_solutions, new_test_points):
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


class HVC():

    def __init__(self, numberofparameters, evaluator, parameter_upper_limits, parameter_lower_limits):
        self.parameter_upper_limits = parameter_upper_limits
        self.parameter_lower_limits = parameter_lower_limits
        self.number_of_parameters = numberofparameters
        self.evaluator = evaluator
        self.clusters = []
        self.archives = []

    def init_clusters(self, population):
        scaled_search_volume = 1
        for p in range(self.number_of_parameters):
            scaled_search_volume *= pow(self.parameter_upper_limits[p] - self.parameter_lower_limits[p],
                                        1 / self.number_of_parameters)
        self.average_edge_length = scaled_search_volume * pow(population.size, -1.0 / self.number_of_parameters)
        self.population = population
        test_points = []
        cluster_index_of_test_points = []
        clustering_max_number_of_neighbours = self.number_of_parameters + 1
        self.cluster_index = [-1] * population.size
        # first solution (best) goes in first cluster
        self.cluster_index[0] = 0
        self.number_of_clusters = 1
        # Generate nearest better tree
        self.population.eval(self.evaluator)
        self.population.order()
        # for every solution
        for i in range(self.population.size):
            dist = np.zeros(i)
            nearest_dist = 1e308
            furthest_dist = 0
            nearest_better = 0
            furthest_better = 0
            # loop through all solutions before solution i therefore solution j is always better than i
            for j in range(i - 1):
                # TODO could calculte all distances in matrix before
                dist[j] = self.population.solutions[i].param_distance(self.population.solutions[j])
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
                    if (does_not_belong_to[k] == self.cluster_index[nearest_better]):
                        skip_neighbour = True
                        break

                if skip_neighbour:
                    continue

                force_accept = False
                max_number_of_trial_solutions = 1 + (int(dist[nearest_better] / self.average_edge_length))
                new_test_points = []
                # print("doing trials : ", max_number_of_trial_solutions)

                if i > (0.5 * self.population.size) and max_number_of_trial_solutions == 1:
                    force_accept = True

                if force_accept or hillvalleytest(self.evaluator, self.population.solutions[i],
                                                  self.population.solutions[nearest_better], max_number_of_trial_solutions,
                                                  new_test_points):
                    self.cluster_index[i] = self.cluster_index[nearest_better]
                    edge_added = True

                    for t in new_test_points:
                        test_points.append(t)
                        cluster_index_of_test_points.append(self.cluster_index[nearest_better])
                    break
                else:
                    does_not_belong_to[j] = self.cluster_index[nearest_better]

                    if len(new_test_points) > 0:
                        for k in range(len(new_test_points) - 1):
                            test_points_for_curr_sol.append(new_test_points[k])

            if not edge_added:
                self.cluster_index[i] = self.number_of_clusters
                self.number_of_clusters += 1

                for t in test_points_for_curr_sol:
                    test_points.append(t)
                    cluster_index_of_test_points.append(self.cluster_index[i])

        # Generate clusters
        # population for each cluster
        candidate_clusters = [[] for i in range(self.number_of_clusters)]
        cluster_active = [True] * self.number_of_clusters
        # print("solutions: ", self.population.solutions)
        # print("cluster index: ", self.cluster_index)

        for i in range(len(self.cluster_index)):
            candidate_clusters[self.cluster_index[i]].append(self.population.solutions[i])
            self.population.solutions[i].cluster_number = int(self.cluster_index[i])

            if len(candidate_clusters[self.cluster_index[i]]) == 1 and self.population.solutions[i].elite:
                cluster_active[self.cluster_index[i]] = False

        # TODO test points here
        for i in range(len(test_points)):
            candidate_clusters[cluster_index_of_test_points[i]].append(test_points[i])

        clusters = []
        for i in range(len(candidate_clusters)):
            if (cluster_active[i]):
                clusters.append(candidate_clusters[i])
        clusters = [x for x in clusters if x != []]
        for i, c in enumerate(clusters):
            for sol in c:
                sol.cluster_number = i
        # print("final clusters: ", clusters)
        self.clusters = clusters
        self.archives = [Archive(cluster, self.evaluator) for cluster in clusters]
        for i, sol in enumerate(self.population.solutions):
            # print(self.population.solutions[i].cluster_number)
            if self.population.solutions[i].cluster_number is None:
                raise Exception()
        return clusters

    # ========================================================================================

    def add_to_clusters(self, newPop: Population):
        scaled_search_volume = 1
        for p in range(self.number_of_parameters):
            scaled_search_volume *= pow(self.parameter_upper_limits[p] - self.parameter_lower_limits[p],
                                        1 / self.number_of_parameters)
        # print("sols ", newPop.solutions)
        # print("self", self.population.solutions)
        newPop_indexes = self.population.combine(newPop.solutions, self.evaluator)
        self.average_edge_length = scaled_search_volume * pow(self.population.size, -1.0 / self.number_of_parameters)
        self.cluster_index = []
        test_points = []
        cluster_index_of_test_points = []
        clustering_max_number_of_neighbours = self.number_of_parameters + 1
        new_cluster_index = [-1] * newPop.size
        old_cluster_count = self.number_of_clusters
        old_archives = self.archives.copy()
        # new_cluster_index[0] = 0

        # Generate nearest better tree
        # combine new pop by looping thoru worse solutions with i as each new pop when placed in with old pop

        edges = []
        for index in range(newPop.size):
            i = newPop_indexes[index]
            dist = np.zeros(i)
            nearest_dist = 1e308
            furthest_dist = 0
            nearest_better = 0
            furthest_better = 0
            for j in range(i-1):
                dist[j] = self.population.solutions[i].param_distance(self.population.solutions[j])
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
                    # if self.population.solutions[nearest_better].cluster_number != None:
                    if (does_not_belong_to[k] == self.population.solutions[nearest_better].cluster_number):
                        skip_neighbour = True
                        break

                if skip_neighbour:
                    continue

                force_accept = False
                max_number_of_trial_solutions = 1 + (int(dist[nearest_better] / self.average_edge_length))
                new_test_points = []
                # print("doing trials : ", max_number_of_trial_solutions)

                if i > (0.5 * self.population.size) and max_number_of_trial_solutions == 1:
                    force_accept = True

                if force_accept or hillvalleytest(self.evaluator, self.population.solutions[i],
                                                  self.population.solutions[nearest_better], max_number_of_trial_solutions,
                                                  new_test_points):
                    self.population.solutions[i].cluster_number = self.population.solutions[nearest_better].cluster_number
                    edge_added = True

                    for t in new_test_points:
                        test_points.append(t)
                        cluster_index_of_test_points.append(self.population.solutions[nearest_better].cluster_number)
                    break
                else:
                    does_not_belong_to[j] = self.population.solutions[nearest_better].cluster_number

                    if len(new_test_points) > 0:
                        for k in range(len(new_test_points) - 1):
                            test_points_for_curr_sol.append(new_test_points[k])

            if not edge_added:
                self.population.solutions[i].cluster_number = self.number_of_clusters
                self.number_of_clusters += 1

                for t in test_points_for_curr_sol:
                    test_points.append(t)
                    cluster_index_of_test_points.append(self.population.solutions[i].cluster_number)

        # Generate clusters
        # population for each cluster
        candidate_clusters = [[] for i in range(self.number_of_clusters)]
        cluster_active = [True] * self.number_of_clusters
        for i, sol in enumerate(self.population.solutions):
            # print(self.population.solutions[i].cluster_number)
            candidate_clusters[self.population.solutions[i].cluster_number].append(self.population.solutions[i])

        # print("solutions: ", self.population.solutions)
        # print("cluster index: ", new_cluster_index)
        # self.cluster_index = self.cluster_index + new_cluster_index
        # for i in range(len(self.cluster_index)):
        #     candidate_clusters[new_cluster_index].append(self.population.solutions[i])
        #     self.population.solutions[i].cluster_number = int(self.cluster_index[i])
        #

        #     if len(candidate_clusters[self.cluster_index[i]]) == 1 and self.population.solutions[i].elite:
        #         cluster_active[self.cluster_index[i]] = False

        # TODO test points here
        last_index =len(self.population.solutions)
        num_clusters = 1
        for i in range(len(test_points)):
            candidate_clusters[cluster_index_of_test_points[i]].append(test_points[i])
            test_points[i].cluster_number = cluster_index_of_test_points[i]
            self.population.solutions.append(test_points[i])
            if test_points[i].cluster_number > num_clusters:
                num_clusters = test_points[i].cluster_number
            # if test_points[i].cluster_number >= len(old_archives):
            #     self.archives.append(Archive([test_points[i]], self.evaluator))
            #     old_cluster_count += 1
            # else:
            #     self.archives[test_points[i].cluster_number].updateArchive(test_points[i])


        clusters = []
        for i in range(len(candidate_clusters)):
            if cluster_active[i]:
                clusters.append(candidate_clusters[i])
        clusters = [x for x in clusters if x != []]
        # print("final clusters: ", clusters)
        self.clusters = clusters
        newSols = [self.population.solutions[index] for index in newPop_indexes]
        for sol in newSols:
            if sol.cluster_number >= num_clusters:
                num_clusters = sol.cluster_number
        # print("num clusteres ", num_clusters)
        while len(self.archives) < num_clusters+1:
            self.archives.append(None)


        # print("old count", old_cluster_count)
        # print(self.archives)


        for i, sol in enumerate(test_points):
            # print("c num ", sol.cluster_number)
            if self.archives[sol.cluster_number] is None:
                self.archives[sol.cluster_number] = Archive([sol], self.evaluator)
            else:
                self.archives[sol.cluster_number].updateArchive(sol)

        for i, sol in enumerate(newSols):
            # print("new num ", sol.cluster_number)
            if self.archives[sol.cluster_number] is None:
                self.archives[sol.cluster_number] = Archive([sol], self.evaluator)
            else:
                self.archives[sol.cluster_number].updateArchive(sol)
        self.archives = [a for a in self.archives if a is not None]
        # print(len(clusters))
        # print(clusters)
        # print(self.archives)

            # if sol.cluster_number >= len(old_archives):
            #     self.archives.append(Archive([sol], self.evaluator))
            #     old_cluster_count += 1

            # print("old arcive", old_archives)
            # self.archives[sol.cluster_number].updateArchive(sol)
        return clusters