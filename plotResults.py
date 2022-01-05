import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.problem import Problem
from mpl_toolkits import mplot3d # for three d plots
from pymoo.factory import get_visualization

def plot_vector(solutions, problem, i, gen_zero=False):
    X = np.asarray([s.param for s in solutions])
    Y = problem.evaluate(X, return_values_of=["F"])
    cluster = [s.cluster_number for s in solutions]
    from matplotlib import cm, colors

    norm = colors.Normalize(vmin=0.0, vmax=10.0, clip=False)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Oranges)
    # print(cluster)
    colours = [mapper.to_rgba(c) for c in cluster]

    pf = problem._calc_pareto_front(5000)
    ps = problem._calc_pareto_set(5000)

    fig = plt.figure(0)
    fig.clear()
    plt.scatter(ps[:, 0], ps[:, 1], marker="x", alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], color="red")
    # print("X = ", ps)
    if gen_zero:
        plt.title("Gen 0, Vector " + str(i))
    else:
        plt.title("after init_clusters Vector " + str(i))
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    plt.close(fig)

def plot_decision_space(three_dimension: bool, X, ps, colours, problem_name):
    fig = plt.figure(0)
    fig.clear()
    plt.scatter(ps[:, 0], ps[:, 1], marker="x", alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], color=colours)
    # print("X = ", ps)

    plt.title(problem_name + " Decision space")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    plt.close(fig)

from pymoo.factory import get_visualization

def plot_objective_space(three_dimension: bool, Y, pf, colours, problem_name):
    fig = plt.figure(0)
    fig.clear()

    if three_dimension:
        get_visualization("scatter", angle=(45, 45)).add(Y).show()
        print("plotting 3d ")
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(Y[:, 0], Y[:, 1], Y[:, 2], color=colours)
        # ax.view_init(-60, 44)
        # plt.show()
        # ax.view_init(-123, 44)
    else:
        plt.scatter(Y[:, 0], Y[:, 1], color=colours)
    # print(scater_val)
    # plt.legend()
    # plt.title(problem_name + " Objective space")
    # plt.xlabel("$f_1$")
    # plt.ylabel("$f_2$")
    plt.show()
    # for sol in final_pop:
    #     plt.scatter(sol.param, prob.evaluate(np.asarray([sol.param]), return_values_of=["F"]), color= mapper.to_rgba(sol.cluster_number), label=sol.cluster_number)
    # plt.legend()
    # plt.show()


def standard_plots(problem: Problem, problem_name: str, X: np.array, Y: np.array, ps, pf, cluster):
    from matplotlib import cm, colors

    norm = colors.Normalize(vmin=0.0, vmax=10.0, clip=False)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Oranges)
    # print(cluster)
    colours = [mapper.to_rgba(c) for c in cluster]
    if problem.n_var < 4:
        if problem.n_var == 3:
            three_dim = True
        else:
            three_dim = False

        plot_decision_space(three_dim, X, ps, colours, problem_name)

    else:
        print("Too many variables to plot decision space; n_var = ", problem.n_var)

    if problem.n_obj < 4:
        if problem.n_obj == 3:
            three_dim = True
            get_visualization("scatter", angle=(45, 45)).add(Y).show()
        else:
            three_dim = False
            get_visualization("scatter").add(Y).show()

            # plot_objective_space(three_dim, Y, pf, colours, problem_name)
    else:
        print("Too many variables to plot objective space; n_var = ", problem.n_obj)

    # print(problem.xu)
    # print(problem.xl)