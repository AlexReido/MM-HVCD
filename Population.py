

class Population():

    def __init__(self, solutions):
        self.solutions = solutions
        self.size = len(solutions)

    def truncation_percentage(self, population, truncation_percentage):
        """Select the truncation_percentage*population_size best solutions"""
        newSize = int(truncation_percentage * len(population))
        return population[:newSize]


