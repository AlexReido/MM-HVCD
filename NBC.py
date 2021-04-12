# Nearest better clustering started writing by accident


class Edge:
    def __init__(self, start, end, length):
        self.start = start
        self.end = end
        self.length = length

def hillvalleyclustering(solutions: list(Solution)):

    # Generate nearest better tree
    solutions = sorted(solutions, key=lambda x: x.f)
    edges = []
    for i in range(len(solutions)):
        dist = np.zeros(i-1)
        nearest_dist = 1e308
        for j in range(i-1):
            dist[j] = solutions[i].param_distance(solutions[j])
            if dist[j] < nearest_dist:
                nearest_dist = dist[j]
                nearest_better = j
        # edge is a tuple with from, to and edge length
        edges.append(Edge(solutions[i], solutions[j], nearest_dist))

    # Generate clusters
    mean_edge_len = np.mean([e.length for e in edges])