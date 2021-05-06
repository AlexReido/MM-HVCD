from Solution import *

class Amalgam_Uni():
    def __init__(self, population):
        # Population should be sorted
        self.pop = sorted(population, key=lambda x: x.f)
        self.multiplier = 1.0
        self.no_improvement_stretch = 0
        self.old_mean = popMean(self.pop)
        self.mean = self.old_mean
        best = self.pop[0]

    def estimate_sample_parameters(self):
        self.old_mean = self.mean
        if self.multiplier < 1:
            # focus on best solution
            mean = self.pop[0].param
        else:
            self.mean = popMean(self.pop)



    def covariance_univariate(self, mean, covariance):
        # use maximum likelihood estimate
        problemSize = len(self.pop[0].param)
        covariance = np.zeros((problemSize, problemSize))
        for i in range(problemSize):
            for k in range(len(self.pop)):
                covariance[i][i] += (self.pop[k].param - mean[i]) ** 2
            covariance[i][i] /= len(self.pop)
        return covariance





    def sample_new_population(self, current_cluster_size):
        number_of -_sample= fill_normal_univariate(current_cluster_size)

        ams_direction = self.mean - self.old_mean
        number_of_ams_sols = 0.5 * current_cluster_size * selection_fraction
        apply_ams_to_pop(number_of_ams_sols, delta_ams * self.multiplier, ams_direction)

        # evaluate pop and sort on fitness

