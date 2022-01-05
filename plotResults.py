import matplotlib.pyplot as plt
import numpy as np




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