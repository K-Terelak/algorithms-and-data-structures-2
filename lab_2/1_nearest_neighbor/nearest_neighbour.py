from collections import defaultdict


def nearest_neighbor(edges, start_node):
    graph = defaultdict(list)

    for x, y, z in edges:
        graph[x].append([y, z])
        graph[y].append([x, z])

    for x in graph.keys():
        graph[x] = sorted(graph[x], key=lambda l: l[1])

    visited = set()
    visited.add(start_node)

    def dfs(node, path):

        if node == start_node and len(path) == len(graph):
            return path

        for nei, _ in graph[node]:

            if nei not in visited:
                visited.add(nei)
                path.append(nei)
                dfs(nei, path)

        return path

    return dfs(start_node, [start_node]) + [start_node]


edges = [(1, 2, 10), (2, 3, 20), (3, 4, 30), (4, 1, 40), (1, 3, 15), (2, 4, 25)]
start_node = 1

result = nearest_neighbor(edges, start_node)
print(f"Nearest Neighbour path is {result}")
