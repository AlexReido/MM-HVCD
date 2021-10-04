import Solution
# from pymoo.algorithms.nsga2 import NSGA2
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from Population import Population


class Archive():

    def __init__(self, solutions, evaluator):
        self.archive = []
        for i, sol in enumerate(solutions):
            elite = True
            for j, otherSol in enumerate(solutions[i:]):
                #relation = self.getrelation(sol, otherSol)
                if sol.f > otherSol.f:
                    elite = False
                    break
            sol.elite = elite

        self.archive = [sol for sol in solutions if sol.elite]

    def updateArchive(self, solution):
        remove = []
        insert = {}
        elite = False
        for i, elite in enumerate(self.archive):
            if solution.f == elite.f:
                elite = True
            elif solution.f < elite.f:
                elite = True
                remove.append(i)

        if elite:
            self.archive.append(solution)
        self.archive = [val for i, val in enumerate(self.archive) if not i in remove]

    def getrelation(self, this_sol: Solution, other_sol: Solution):
        """    This is used for non-dominated sorting and returns the dominance relation between objectives
         return returns 1 if objective dominates, -1 if objective is dominated and 0 if objectives are indifferent
        """
        val = 0
        for i in range(len(this_sol.objectives)):
            if this_sol.objectives[i] < other_sol.objectives[i]:
                if val == -1:
                    return 0
                val = 1
            elif this_sol.objectives[i] > other_sol.objectives[i]:
                if val == 1:
                    return 0
                val = -1
            else:
                return 0
        return val
