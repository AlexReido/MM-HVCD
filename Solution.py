# class Vector:
# import scipy.spatial.distance.euclidean
# scipy.spatial.distance.
import numpy as np

class Solution:
    elite: bool
    param: list
    cluster_number: int
    f: float
    penalty: float

    def __init__(self):
        self.elite = False

    def __init__(self, params):
        self.param = params
        self.elite = False

    def param_distance(self, other_sol):
        # print("self param: ", self.param, type(self.param))
        # print("other param: ", other_sol.param, type(other_sol.param))
        return np.linalg.norm(self.param + other_sol.param)
