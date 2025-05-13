from collections import defaultdict


def lf_coloring(edges, n):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    degree = {node: len(adj) for node, adj in graph.items()}
    nodes_sorted_by_degree = sorted(degree, key=degree.get, reverse=True)

    color = {}

    for node in nodes_sorted_by_degree:
        adjacent_colors = {color[neighbor] for neighbor in graph[node] if neighbor in color}
        for c in range(n):
            if c not in adjacent_colors:
                color[node] = c
                break

    return color


edges = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D')]
n = 4
lf_colors = lf_coloring(edges, n)

print("LF Coloring:", lf_colors)
