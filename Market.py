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
        self.buyer_capacity = reduce(lambda c, i: c | {(self.buyers[i], self.sink): self.e[i]}, range(self.m), {})

    def equalityGraph(self, prices):
        match_capacity = reduce(lambda c, e: c|{e: float("inf")}, self.matches(prices), {})
        good_capacity = reduce(lambda c, j: c|{(self.source, self.goods[j]): prices[j]}, range(self.n), {})
        capacity = good_capacity|match_capacity|self.buyer_capacity
        return FlowNetwork(self.source + 1, capacity)

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
        findMin = lambda x, mi: mi if mi > 1 and (x is None or mi < x) else x
        return reduce(findMin, m, None) - 1

    def adjustedDemand(self, p, m):
        def go(b):
            c_s, c_t = (sum(p), sum(b))
            if c_s == c_t:
                return b
            else:
                delta = (c_t - c_s)
                d = self.canDecrease(b, delta)
                r = min(delta // sum(d), self.minDecrease(b))
                return go(b - r * d)
        return go(m)

    def canDecrease(self, m, delta):
        dec = lambda v_count, mi: (v_count[0] + [int(mi >= 2)], v_count[1] + int(mi >= 2)) \
            if v_count[1] < delta else (v_count[0] + [0], delta)
        return np.array(reduce(dec, m, ([], 0))[0])



