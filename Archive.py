import Solution
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from Population import Population


class Archive():

    def __init__(self, pop: Population):
        self.archive = []
        for i, sol in enumerate(pop.solutions):
            elite = True
            for j, otherSol in enumerate(pop.solutions[i:]):
                relash = self.getrelation(sol, otherSol)
                if relash == -1:
                    elite = False
                    break
            sol.elite = elite

        self.archive = [sol for sol in pop.solutions if sol.elite]

    def updateArchive(self, solutions):
        remove = {}
        insert = {}
        for i, elite in enumerate(self.archive):
            for j, sol in enumerate(solutions):
                relash = self.getrelation(sol, elite)
                if relash == 1:
                    remove.add(i)
                    insert.add(j)
                elif relash == -1:
                    break
            insert.add(j)

        self.archive = [val for i, val in self.archive if not i in remove]
        for i in insert:
            self.archive.append(solutions[i])

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
