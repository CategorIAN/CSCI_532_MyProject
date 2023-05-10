import numpy as np
from functools import reduce
from FlowNetwork import FlowNetwork
import math
from tail_recursive import tail_recursive as tail
from EqualityGraph import EqualityGraph
from PathSet import PathSet
from ResidualNetwork import ResidualNetwork

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
        self.V = {self.source, self.sink}|set(self.goods)|set(self.buyers)

    def addGoods(self, alpha, beta):
        def f(i):
            alpha_i, beta_i = (alpha[i], beta[i])
            def addGood(gs, j_x):
                if abs(j_x[1] - alpha_i) < 0.001:
                    return gs|{(self.goods[j_x[0]], self.buyers[i])}
                else:
                    return gs
            return reduce(addGood, zip(range(self.n), beta_i), set())
        return f

    def matches(self, prices, addAlpha = False):
        beta = np.round(np.array(list(map(lambda ui: ui / prices, self.u))))
        alpha = np.round(np.array(list(map(lambda i: max(beta[i]), range(self.m)))))
        matchEdges = reduce(lambda s1, s2: s1 | s2, map(self.addGoods(alpha, beta), range(self.m)))
        return (matchEdges, alpha) if addAlpha else matchEdges

    def equalityGraph(self, prices, m = None):
        m = self.e if m is None else m
        matches = self.matches(prices)
        match_c = reduce(lambda c, e: c | {e: float("inf")}, matches, {})
        good_c = reduce(lambda c, j: c | {(self.source, self.goods[j]): prices[j]}, range(self.n), {})
        buyer_c = reduce(lambda c, i: c | {(self.buyers[i], self.sink): m[i]}, range(self.m), {})
        capacity = good_c | match_c | buyer_c
        return EqualityGraph(capacity=capacity, source=self.source, sink=self.sink, V = self.V,
                             prices = prices, goods = self.goods, buyers = self.buyers, matches = matches)

    def myPrices(self, initial = None):
        initial = 1/self.n * np.ones(self.n) if initial is None else initial
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
        return go(initial)

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
        return FlowNetwork(newCapacity, N.source, N.sink, N.V)

    def inducedNetworks(self, N, S):
        def direct(c1_c2, e):
            (x, y) = e
            if x in S:
                if y in S.union({self.sink}):
                    return (c1_c2[0]|{e:N.c[e]}, c1_c2[1]|{e:0})
                elif x == self.source:
                    return (c1_c2[0]|{e:0}, c1_c2[1]|{e: N.c[e]})
                else:
                    return (c1_c2[0]|{e:0}, c1_c2[1]|{e: 0})
            else:
                if y not in S:
                    return (c1_c2[0], c1_c2[1] | {e: N.c[e]})
                else:
                    return (c1_c2[0]|{e:0}, c1_c2[1]|{e: 0})
        (c1, c2) = reduce(direct, N.c.keys(), ({}, {}))
        T = N.V.difference(S)
        return (EqualityGraph(c1, N.source, N.sink, S|{N.sink}, N.prices, N.goods, N.buyers),
                EqualityGraph(c2, N.source, N.sink, T|{N.source}, N.prices, N.goods, N.buyers))

    def inducedNetworks2(self, N, S):
        def direct(c1_c2, e):
            (x, y) = e
            if x in S:
                if y in S.union({self.sink}):
                    return (c1_c2[0]|{e:N.c[e]}, c1_c2[1])  #(x1, x2)
                elif x == self.source:
                    return (c1_c2[0], c1_c2[1]|{e: N.c[e]})  #(s, y)
                else:
                    return (c1_c2[0], c1_c2[1])       #(x, y) does not work
            else:
                if y not in S:
                    return (c1_c2[0], c1_c2[1] | {e: N.c[e]})  #(y1, y2)
                else:
                    return (c1_c2[0], c1_c2[1])           #(y, x) does not work
        (c1, c2) = reduce(direct, N.c.keys(), ({}, {}))
        T = N.V.difference(S)
        return (EqualityGraph(c1, N.source, N.sink, S|{N.sink}, N.prices, S.intersection(N.goods),
                              S.intersection(N.buyers), lneighbors = N.lneighbors, rneighbors = N.rneighbors),
                EqualityGraph(c2, N.source, N.sink, T|{N.source}, N.prices, T.intersection(N.goods),
                              T.intersection(N.buyers), lneighbors = N.lneighbors, rneighbors = N.rneighbors))

    def balancedFlow(self, prices, network = None, showrecurse = False):
        network = self.equalityGraph(prices) if network is None else network
        def go(N, recurse):
            m = self.adjustedDemand(prices, self.e)
            N_adj = self.adjustNetwork(N, m)
            f, S = N_adj.fordFulkerson(mincut=True)
            if S == {self.source} or N.V.difference(S) == {self.sink}:
                return (f, recurse) if showrecurse else f
            else:
                #print("S: {}".format(S))
                N1, N2 = self.inducedNetworks(N, S)
                return (go(N1, True)[0]|go(N2, True)[0], True) if showrecurse else go(N1, True)|go(N2, True)
        return go(network, False)

    def surplus(self, flow):
        return np.vectorize(lambda i: self.e[i] - flow[(self.buyers[i], self.sink)], otypes=[np.float64])(range(self.m))

    def maxSurplus(self, flow):
        return max(self.surplus(flow))

    def isEquilibrium(self, flow):
        return self.maxSurplus(flow) == 0

    def I(self, surplus, delta):
        appendI = lambda I, i: I|{self.buyers[i]} if surplus[i] == delta else I
        return reduce(appendI, range(self.m), set())

    def cover(self, collection, A):
        return reduce(lambda B, i: B | collection[i], A , set())

    def J(self, eqG, I):
        return self.cover(eqG.lneighbors, I)

    def K(self, eqG, J):
        S = self.cover(eqG.rneighbors, J).difference(self.cover(eqG.rneighbors, set(self.goods).difference(J)))
        return S

    def newPrice(self, prices, J):
        def f(x):
            inc = np.vectorize(lambda g: x if g in J else 1)(self.goods)
            newPrices = inc * prices
            return newPrices
        return f

    def newMatches(self, eqG, prices):
        matches = self.matches(prices).intersection(eqG.c.keys())
        return (matches.difference(eqG.matches), eqG.matches.difference(matches))

    def resNeighbors(self, resN, I):
        X = reduce(lambda ps, i: ps + resN.neighbor_edges[i], I, PathSet([]))
        (_, S) = resN.branch(*resN.branch(X, {self.source, self.sink}))
        return S.intersection(self.buyers)

    def tightSet(self, prices, eqG, J):
        def go(N, A, B):
            supply = reduce(lambda s, g: s + prices[g - 1], A, 0)
            demand = reduce(lambda d, b: d + self.e[b - 1 - self.n], B, 0)
            x = demand / supply
            print("A, B: {}, {}".format(A, B))
            print("demand, supply: {}, {}".format(demand, supply))
            p = self.newPrice(prices, J)(x)
            (m1, m2) = self.newMatches(N, p)
            newN = N.swap(m1, m2, p)
            f, S = newN.fordFulkerson(mincut=True)
            if S == {self.source} or N.V.difference(S) == {self.sink}:
                return (x, A)
            elif len(S.intersection(self.goods)) == len(A):
                return (x, S.intersection(self.goods))
            else:
                X = S.intersection(self.goods)
                Y = self.K(N, X)
                print("X, Y: {}, {}".format(X, Y))
                N1, _ = self.inducedNetworks2(N, X.union(Y))
                return go(N1, X, Y)
        return go(eqG, self.goods, self.buyers)

    def mainAlg(self):
        prices = self.myPrices()
        N = self.equalityGraph(prices)
        def phase(prices, tight):
            print("###############")
            print("prices: {}".format(prices))
            N = self.equalityGraph(prices)
            f = self.balancedFlow(prices, N)
            totalSupply = reduce(lambda s, g: s + f[(self.source, g)], self.goods, 0)
            print("supply: {}".format(totalSupply))
            surplus = self.surplus(f)
            delta = max(surplus)
            print("delta: {}".format(delta))
            if delta > 0:
                def step(N, I):
                    print("------------------------------------------")
                    J = self.J(N, I)
                    print("N: {}".format(N.c.keys()))
                    print("J: {}".format(J))
                    K = self.K(N, J)
                    newPrice = self.newPrice(prices, J)
                    y, A = self.tightSet(prices, N, J)
                    if y == 0:
                        raise ValueError("y should not be 0")
                    def increment(x):
                        if x >= y:
                            print("Event 1")
                            print("Tight Set: {}".format(A))
                            return phase(newPrice(y))
                        else:
                            p = newPrice(x)
                            (m1, m2) = self.newMatches(N, p)
                            if len(m1) > 0:
                                newN = N.swap(m1, m2, p)
                                b = next(iter(m1))[1]
                                if b in I:
                                    print("Event 2")
                                    (f, recursed) = self.balancedFlow(p, newN, True)
                                    if recursed:
                                        resN = ResidualNetwork(newN, f)
                                        X = self.resNeighbors(resN, I)
                                        return step(newN, I.union(X))
                                    else:
                                        print("Final Price Total: {}".format(sum(p)))
                                        return f
                                elif b in K:
                                    print("Event 3")
                                    return step(newN, I)
                            else:
                                return increment(x + 1)
                    return increment(2)
                return step(N, self.I(surplus, delta))
            else:
                return f
        return phase(prices, set())

    def verify(self, flow):
        totalSupply = reduce(lambda s, g: s + flow[(self.source, g)], self.goods, 0)
        #totalDemand = reduce(lambda d, b: d + flow[(b, self.sink)], self.buyers, 0)
        totalDemand = sum(self.e)
        print("Supply: {}".format(totalSupply))
        print("Demand: {}".format(totalDemand))
        return totalSupply == totalDemand












