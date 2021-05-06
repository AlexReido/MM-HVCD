from pymoo.algorithms.moead import MOEAD
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize
from pymoo.problems.multi.sympart import SYMPART


# problem = SYMPART()
#
#
# algorithm = MOEAD(
#     get_reference_directions("das-dennis", 2, n_partitions=12),
#     n_neighbors=15,
#     decomposition="pbi",
#     prob_neighbor_mating=0.7,
#     seed=1
# )
#
#
# res = minimize(problem, algorithm, termination=('n_gen', 200))

ref = get_reference_directions("das-dennis", 2, n_partitions=12)
print(ref)
get_visualization("scatter").add(ref).show()


# generation of the local optimiser
local_optimizers[i].estimate_sample_parameters()

local_number_of_evaluations = (int) local_optimizers[i].sample_new_population(current_cluster_size)
number_of_evaluations += local_number_of_evaluations


local_optimizers[i].truncation_percentage()
#local_optimizers[i]->average_fitness_history.push_back(local_optimizers[i]->pop->average_fitness())

