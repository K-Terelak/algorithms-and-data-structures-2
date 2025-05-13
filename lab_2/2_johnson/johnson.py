import heapq
from collections import defaultdict


def johnson(edges):
    graph = defaultdict(list)

    for x, y, z in edges:
        graph[x].append((y, z))

    new_node = len(graph)
    for node in graph:
        graph[node].append((new_node, 0))
    graph[new_node] = []

    h_values = bellman_ford(graph, new_node)
    if h_values is None:
        return None

    del graph[new_node]
    for node in graph:
        graph[node] = [edge for edge in graph[node] if edge[0] != new_node]

    for node in graph:
        for i, edge in enumerate(graph[node]):
            graph[node][i] = (edge[0], edge[1] + h_values[node] - h_values[edge[0]])

    shortest_paths = []
    for node in graph:
        shortest_paths.append(dijkstra(graph, node, h_values))

    return shortest_paths


def bellman_ford(graph, source):
    dist = {node: float('inf') for node in graph}
    dist[source] = 0

    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                if dist[node] != float('inf') and dist[node] + weight < dist[neighbor]:
                    dist[neighbor] = dist[node] + weight

    for node in graph:
        for neighbor, weight in graph[node]:
            if dist[node] != float('inf') and dist[node] + weight < dist[neighbor]:
                return None

    return dist


def dijkstra(graph, source, h):
    dist = {node: float('inf') for node in graph}
    dist[source] = 0
    queue = [(0, source)]

    while queue:
        current_dist, current_node = heapq.heappop(queue)

        if current_dist > dist[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_dist + weight + h[current_node] - h[neighbor]
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))

    return dist


edges = [(0, 1, 4), (1, 2, -2), (2, 0, 1), (1, 3, 2), (3, 4, -1), (4, 1, 1)]
result = johnson(edges)
print(result)
