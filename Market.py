import numpy as np
from functools import reduce
from FlowNetwork import FlowNetwork

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
        beta = np.array(list(map(lambda ui: ui / prices, self.u)))
        alpha = np.array(list(map(lambda i: max(beta[i]), range(self.m))))
        matches = reduce(lambda s1, s2: s1|s2, map(self.addGoods(alpha, beta), range(self.m)))
        match_capacity = reduce(lambda c, e: c|{e: float("inf")}, matches, {})
        good_capacity = reduce(lambda c, j: c|{(self.source, self.goods[j]): prices[j]}, range(self.n), {})
        capacity = good_capacity|match_capacity|self.buyer_capacity
        return FlowNetwork(self.source + 1, capacity)

    def addGoods(self, alpha, beta):
        def f(i):
            alpha_i, beta_i = (alpha[i], beta[i])
            addGood = lambda gs, j_x: gs|{(self.goods[j_x[0]], self.buyers[i])} if j_x[1] == alpha_i else gs
            return reduce(addGood, zip(range(self.n), beta_i), set())
        return f


