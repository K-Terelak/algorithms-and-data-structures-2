def d_hondt(G, M):
    P = len(G)
    T = [0] * P
    W = [G[i] / (T[i] + 1) for i in range(P)]

    for _ in range(M):
        max_val = -1
        Ind = -1
        for j in range(P):
            if W[j] > max_val:
                max_val = W[j]
                Ind = j
        T[Ind] += 1
        W[Ind] = G[Ind] / (T[Ind] + 1)

    return T


votes = [742, 251, 987, 649]
seats = 6
print(d_hondt(votes, seats))
