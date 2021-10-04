# class Vector:
# import scipy.spatial.distance.euclidean
# scipy.spatial.distance.
import numpy as np



class Solution:
    elite: bool             # Elite solution and therefore in archive
    param: np.array         # Decision variables
    cluster_number: int     # Cluster Number
    f: float                # function evaluation
    penalty: float          # remove?

    def __init__(self):
        self.elite = False

    def __init__(self, params):
        self.param = np.asarray(params)
        self.elite = False
        self.cluster_number = -1

    def param_distance(self, other_sol):
        """
        The (euclidean) distance from the current solution to another in the decision space
        :param other_sol:
        :return:
        """
        # print("self param: ", self.param, type(self.param))
        # print("other param: ", other_sol.param, type(other_sol.param))
        return np.linalg.norm(self.param + other_sol.param)
