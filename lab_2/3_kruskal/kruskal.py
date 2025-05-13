from typing import List


def kruskal(edges: List[str], n: int):
    par = [i for i in range(n)]
    rank = [1 for i in range(n)]

    def find(n1):
        res = n1
        while par[res] != res:
            par[res] = par[par[res]]
            res = par[res]

        return res

    def union(n1, n2):

        p1, p2 = find(n1), find(n2)

        if p1 == p2:
            return False

        if rank[p2] > rank[p1]:
            par[p1] = p2
            rank[p2] += rank[p1]
        else:
            par[p2] = p1
            rank[p1] += rank[p2]

        return True

    mst = []
    res = 0

    for x, y, z in sorted(edges, key=lambda l: l[2]):
        if union(x, y):
            mst.append((x, y, z))
            res += z
            if len(mst) == n - 1:
                break

    return mst, res


edges = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]
n = 4

mst, total_weight = kruskal(edges, n)

print("Minimum Spanning Tree:")
for edge in mst:
    print(edge)
print("Total Weight of MST:", total_weight)
