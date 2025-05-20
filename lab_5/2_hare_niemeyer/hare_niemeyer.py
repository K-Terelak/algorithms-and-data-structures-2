def hare_niemeyer(G, M):
    P = len(G)
    S = sum(G)
    T = [0] * P
    W = [(G[i] * M) / S for i in range(P)]

    for _ in range(M):
        max_val = -1
        Ind = -1
        for j in range(P):
            if W[j] > max_val:
                max_val = W[j]
                Ind = j
        T[Ind] += 1

        W[Ind] = (G[Ind] * M) / S - T[Ind]

    return T


votes = [742, 251, 987, 649]
seats = 6
print(hare_niemeyer(votes, seats))
