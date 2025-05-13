from collections import defaultdict


def spinal_tree(edges, n):
    mst = []
    res = 0

    graph = defaultdict(list)

    for x, y, z in edges:
        graph[x].append([y, z])
        graph[y].append([x, z])

    for x in graph.keys():
        graph[x] = sorted(graph[x], key=lambda l: l[1])

    visited = set()

    def dfs(node):
        nonlocal res

        if node not in visited:
            visited.add(node)
            for nei, wei in graph[node]:
                mst.append((node, nei, wei))
                res += wei
                visited.add(nei)
                dfs(nei)
                break

    for i in range(n):
        dfs(i)

    return mst, res


edges = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]
n = 4

mst, total_weight = spinal_tree(edges, n)

print("Minimum Spanning Tree (Spinal Tree):")
for edge in mst:
    print(edge)
print("Total Weight of Spinal Tree:", total_weight)
