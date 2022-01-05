from pymoo.problems.multi.sympart import SYMPART, SYMPARTRotated
from pymoo.core.problem import Problem
from pymoo.factory import get_problem, anp


class Solution:
    def __init__(self, D):
        self.n_variables = len(D)
        self.D = D

    def set_F(self, F):
        self.n_objectives = len(F)
        self.F = F

class SimpleProblem(Problem):
    def __init__(self, **kwargs):
        super().__init__(n_var=1, n_obj=1, elementwise_evaluation=True, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        if x > 3 and x < 5:
            out["F"] = x + 100
        else:
            out["F"] = x.sum()


class MFProblem(Problem):
    def __init__(self, **kwargs):
        super().__init__(n_var=1, n_obj=2, elementwise_evaluation=True, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        if x>3 and x < 5:
            out["F"] = anp.column_stack([10, 0.5/x])
        else:
            out["F"] = anp.column_stack([x, 0.5/x])


def hillvalleytest(problem, sol1: Solution, sol2: Solution, max_trials: int) -> bool:
    """
    The hvt checks if worse solution between them a peak is found therefore in differing clusters
    if in same cluster return true else false
    :param problem:
    :param sol1:
    :param sol2:
    :param max_trials:
    :return:
    """
    nevals = 0
    testvals = []
    # print(f)
    val1 = sol1.F
    val2 = sol2.F
    print(val1)
    print(val2)
    testvals = []
    for k in range((max_trials)):
        testval = val1 + ((k + 1.0) / (max_trials + 1.0)) * (val2 - val1)
        testvals.append(testval)
    print(testvals)
    nevals += len(testvals)
    for t in testvals:
        f = problem.evaluate(t, return_values_of=["F"])
        # If worse solution between them a peak is found therefore in differing clusters
        if f > max(val1, val2):
            return False

    return True


if __name__ == '__main__':
    problem = SimpleProblem()
    a = Solution([1])
    b = Solution([2])


    mfp = MFProblem()
    a.set_F(mfp.evaluate(a.D, return_values_of=["F"]))
    b.set_F(mfp.evaluate(b.D, return_values_of=["F"]))
    print(a.F)
    print(b.F)
    print(hillvalleytest(problem, a, b, 5))

    a = Solution([3])
    b = Solution([6])

    mfp = MFProblem()
    a.set_F(mfp.evaluate(a.D, return_values_of=["F"]))
    b.set_F(mfp.evaluate(b.D, return_values_of=["F"]))
    print(a.F)
    print(b.F)
    print(hillvalleytest(problem, a, b, 5))
    # print(mfp.evaluate([3], return_values_of=["F"]))


