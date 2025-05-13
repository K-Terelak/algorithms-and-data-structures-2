import heapq
from collections import defaultdict


def sl_coloring(edges, n):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    degree = {node: len(adj) for node, adj in graph.items()}

    heap = [(len(adj), node) for node, adj in graph.items()]
    heapq.heapify(heap)

    elimination_order = []
    while heap:
        _, node = heapq.heappop(heap)
        elimination_order.append(node)
        for neighbor in graph[node]:
            if neighbor in degree:
                degree[neighbor] -= 1
                heap = [(degree[n], n) for n in degree if n != node]
                heapq.heapify(heap)
        del degree[node]

    color = {}
    for node in reversed(elimination_order):
        available_colors = {color[neighbor] for neighbor in graph[node] if neighbor in color}
        for c in range(n):
            if c not in available_colors:
                color[node] = c
                break

    return color


edges = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D')]
n = 4
sl_colors = sl_coloring(edges, n)

print("SL Coloring:", sl_colors)
