from Solution import *

class Amalgam_Uni():
    def __init__(self, population, evaluator):
        # Population should be sorted
        population.order()
        self.pop = population
        self.multiplier = 1.0
        self.no_improvement_stretch = 0
        self.old_mean = population.getMean()
        self.mean = self.old_mean
        self.evaluator = evaluator
        best = self.pop.solutions[0]
        self.no_improvement_stretch = 0
        self.number_of_params = len(best.param)
        self.maximum_no_improvement_stretch = (int)(self.number_of_params + 25)

    def estimate_sample_parameters(self):
        self.old_mean = self.mean
        if self.multiplier < 1:
            # focus on best solution
            mean = self.pop.solutions[0].param
        else:
            self.mean = self.pop.getMean()
        # if the population size is too small,
        # estimate a univariate covariance matrix
        if (pop->size() == 1):
            covariance.setIdentity(mean.size(), mean.size())
            covariance.multiply(init_univariate_bandwidth * 0.01)
        else:
            self.pop.covariance_univariate(mean, covariance)
        # Cholesky decomposition
        choleskyDecomposition_univariate(covariance, cholesky)

        # apply the multiplier
        cholesky.multiply(sqrt(multiplier))

        // invert
        the
        cholesky
        decomposition
        int
        n = (int)
        covariance.rows();
        inverse_cholesky.setRaw(matrixLowerTriangularInverse(cholesky.toArray(), n), n, n);

    def covariance_univariate(self, mean, covariance):
        # use maximum likelihood estimate
        problemSize = len(self.pop[0].param)
        covariance = np.zeros((problemSize, problemSize))
        for i in range(problemSize):
            for k in range(self.pop.size):
                covariance[i][i] += (self.pop.solutions[k].param - mean[i]) ** 2
            covariance[i][i] /= self.pop.size
        return covariance



    def sample_new_population(self, current_cluster_size):
        """ current cluster size is sample size """
        number_of_samples = fill_normal_univariate(current_cluster_size)

        ams_direction = self.mean - self.old_mean
        number_of_ams_sols = 0.5 * current_cluster_size * selection_fraction
        apply_ams_to_pop(number_of_ams_sols, delta_ams * self.multiplier, ams_direction)

        # evaluate pop and sort on fitness
        self.pop.eval(self.evaluator)
        self.pop.order

        improvement = self.pop.improvement(self.best)
        sdr = getSDR(inverse_cholesky)
        sample_success_ratio = (current_cluster_size-1) / number_of_samples
        update_distribution_multiplier(improvement, sample_success_ratio, sdr);
        best = self.pop.solutions[0]

        # number_of_generations += 1

        # return number_of_evaluations

    def getSDR(self, inverse_cholensky):
        # find improvements over the best.
        average_params = [0] * self.number_of_parameters
        i = 0
        while (i < self.pop.size) and (self.pop.solutions[i].f < self.best.f):
            # TODO pairwise add
            average_params += self.pop.solutions[i].param
            i += 1

        if (i == 0):
            return 0.0

        average_params /= i
        # TODO pairwise
        diff = average_params - self.mean

        return inverse_chol.lowerProduct(diff).infinitynorm();


    def updateDistributionMultiplier(self, improvement, sample_success_ratio, sdr):
        # default variables;
        st_dev_ratio_threshold = 1
        sample_succes_ratio_threshold = 0.10
        distribution_multiplier_decrease = 0.9

        # if > 90 % of the samples is out of bounds, multiplier * 0.5
        if (sample_success_ratio < sample_succes_ratio_threshold):
            self.multiplier *= 0.5

        if (improvement):
            self.no_improvement_stretch = 0;

            if (self.multiplier < 1.0):
                self.multiplier = 1.0;

            if (sdr > st_dev_ratio_threshold):
                self.multiplier /= distribution_multiplier_decrease

        else:
            if (self.multiplier <= 1.0):
                self.no_improvement_stretch += 1

            if (self.multiplier > 1.0 or self.no_improvement_stretch >= self.maximum_no_improvement_stretch):
                self.multiplier *= distribution_multiplier_decrease;

            if (self.multiplier < 1.0 and self.no_improvement_stretch < self.maximum_no_improvement_stretch):
                self.multiplier = 1.0;

