from pymoo.problems.multi.sympart import SYMPART
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.factory import get_problem
import numpy as np


def getProblem(problem_name):
    """
    Takes the problem name as a string and returns the problem instantiated along with the reference point for the
    given problem.

    The reference point should be the same length as the number of dimensions of the given problem.
    :param problem_name:
    :return: Problem instance, reference point
    """
    if problem_name == "SYMPART":
        reference_point = np.array([10, 10])
        return SYMPART(), reference_point
    elif problem_name == "OmniTest":
        reference_point = np.array([10, 10])
        return OmniTest(), reference_point
    elif problem_name == "DTLZ1":
        reference_point = np.array([10, 10, 10])
        return get_problem("dtlz1"), reference_point
    elif problem_name == "DTLZ2":
        reference_point = np.array([10, 10, 10])
        return get_problem("dtlz2"), reference_point
    elif problem_name == "DTLZ2":
        reference_point = np.array([10, 10, 10])
        return get_problem("dtlz2"), reference_point
    elif problem_name == "DTLZ3":
        reference_point = np.array([10, 10, 10])
        return get_problem("dtlz3"), reference_point
    elif problem_name == "TwoOnOne2":
        reference_point = np.array([10, 10, 10])
        return get_problem(""), reference_point
    elif problem_name == "SSUF":
        pass




