import numpy as np
from functools import reduce
from FlowNetwork import FlowNetwork
import math

class Market:
    def __init__(self, e, u):
        if len(e) != u.shape[0]:
            raise IndexError("Dimensions Do Not Match!")
        self.e = e
        self.u = u
        self.m, self.n = u.shape
        self.source = 0
        self.goods = list(range(1, 1 + self.n))
        self.buyers = list(range(1 + self.n, 1 + self.n + self.m))
        self.sink = 1 + self.n + self.m

    def equalityGraph(self, prices, m = None):
        m = self.e if m is None else m
        match_capacity = reduce(lambda c, e: c|{e: float("inf")}, self.matches(prices), {})
        good_capacity = reduce(lambda c, j: c|{(self.source, self.goods[j]): prices[j]}, range(self.n), {})
        buyer_capacity = reduce(lambda c, i: c | {(self.buyers[i], self.sink): m[i]}, range(self.m), {})
        capacity = good_capacity|match_capacity|buyer_capacity
        return FlowNetwork(self.sink + 1, capacity)

    def matches(self, prices, addAlpha = False):
        beta = np.array(list(map(lambda ui: ui / prices, self.u)))
        alpha = np.array(list(map(lambda i: max(beta[i]), range(self.m))))
        matchEdges = reduce(lambda s1, s2: s1 | s2, map(self.addGoods(alpha, beta), range(self.m)))
        return (matchEdges, alpha) if addAlpha else matchEdges

    def addGoods(self, alpha, beta):
        def f(i):
            alpha_i, beta_i = (alpha[i], beta[i])
            addGood = lambda gs, j_x: gs|{(self.goods[j_x[0]], self.buyers[i])} if j_x[1] == alpha_i else gs
            return reduce(addGood, zip(range(self.n), beta_i), set())
        return f

    def initialPrices(self):
        def go(prices):
            priceAdjuster = self.adjustedPrice(prices)
            def go2(js):
                if len(js) == 0:
                    return None
                else:
                    j = js[0]
                    newPrice = priceAdjuster(j)
                    return go2(js[1:]) if newPrice is None else self.updatedVector(prices, j, newPrice)
            adjusted = go2(list(range(self.n)))
            return prices if adjusted is None else go(adjusted)
        return go(1/self.n * np.ones(self.n))

    def updatedVector(self, vec, index, newVal):
        return np.vectorize(lambda j: newVal if j == index else vec[j])(np.arange(len(vec)))

    def adjustedPrice(self, prices):
        matchEdges, alpha = self.matches(prices, addAlpha = True)
        def f(j):
            good = self.goods[j]
            for match in matchEdges:
                if good == match[0]:
                    return None
            return max(self.u[:, j]/alpha)
        return f

    def minDecrease(self, m):
        findMin = lambda x, mi: mi if 0 < mi < x else x
        return reduce(findMin, m, float("inf"))

    def adjustedDemand(self, p, m):
        def go(b):
            c_s, c_t = (sum(p), sum(b))
            if c_s == c_t:
                return b
            else:
                delta = (c_t - c_s)
                d = np.vectorize(lambda bi: int(bi > 0))(b)
                r = min(delta / sum(d), self.minDecrease(b))
                return go(b - r * d)
        return go(m)

    def adjustNetwork(self, N, m):
        adjustCapacity = lambda c, i: c|{(self.buyers[i], self.sink): m[i]}
        newCapacity = reduce(adjustCapacity, range(self.m), N.c)
        return FlowNetwork(N.n, newCapacity)

    def inducedNetworks(self, N, S):
        def direct(c1_c2, e):
            (x, y) = e
            if x in S:
                if y in S:
                    return (c1_c2[0]|{e:N.c[e]}, c1_c2[1])
                elif x == self.source:
                    return (c1_c2[0], c1_c2[1]|{e: N.c[e]})
                elif y == self.sink:
                    return (c1_c2[0]|{e:N.c[e]}, c1_c2[1])
                else:
                    return c1_c2
            else:
                if y not in S:
                    return (c1_c2[0], c1_c2[1] | {e: N.c[e]})
                else:
                    return c1_c2
        (c1, c2) = reduce(direct, N.c.keys(), ({}, {}))
        return (FlowNetwork(N.n, c1), FlowNetwork(N.n, c2))


    def balancedFlow(self, prices):
        N = self.equalityGraph(prices)
        m = self.adjustedDemand(prices, self.e)
        N_adj = self.adjustNetwork(N, m)
        f, S = N_adj.fordFulkerson(mincut=True)
        if S == {self.source}:
            return f




