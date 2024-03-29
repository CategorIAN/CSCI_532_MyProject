from FlowNetwork import FlowNetwork
from functools import reduce
import numpy as np
import pandas as pd

class EqualityGraph (FlowNetwork):
    def __init__(self, capacity, source, sink, V, prices, goods, buyers, matches = None, addMatches = None, delMatches = None,
                 lneighbors = None, rneighbors = None):
        super().__init__(capacity, source, sink, V)
        appendMatch = lambda E, e: E|{e} if capacity[e] == float("inf") else E
        self.matches = reduce(appendMatch, capacity.keys(), set()) if matches is None else matches
        self.prices = prices
        self.goods = set(goods)
        self.buyers = set(buyers)
        self.lneighbors, self.rneighbors = self.neighbors(addMatches, delMatches, lneighbors, rneighbors)
        frozen = set()
        for g in self.goods:
            if len(self.rneighbors[g]) == 0:
                del self.c[(self.source, g)]
                frozen.add(g)
        self.frozen = frozen
        self.df = self.frame()

    def frame(self, flow = None):
        df = pd.DataFrame(index = sorted(list(self.c.keys())), columns = ["Capacity"])
        for e in df.index:
            df.at[e, "Capacity"] = self.c[e]
        if flow is not None:
            df.columns = df.columns
            df.loc[:, ["Flow"]] = df.index.map(lambda e: flow[e])
            df["Surplus"] = df["Capacity"] - df["Flow"]
        return df

    def __str__(self):
        return str(self.df)

    def __repr__(self):
        return str(self.df)

    def neighbors(self, addMatches = None, delMatches = None, lneighbors = None, rneighbors = None):
        addmatches = self.matches if addMatches is None else addMatches
        delmatches = set() if delMatches is None else delMatches
        lneighbors = dict(list(map(lambda b: (b, set()), self.buyers))) if lneighbors is None else lneighbors
        rneighbors = dict(list(map(lambda g: (g, set()), self.goods))) if rneighbors is None else rneighbors
        def addneighbor(ln_rn, e):
            (g, b) = e
            gs, bs = (ln_rn[0][b]|{g}, ln_rn[1][g]|{b})
            return (ln_rn[0]|{b: gs}, ln_rn[1]|{g: bs})
        def delneighbor(ln_rn, e):
            (g, b) = e
            gs, bs = (ln_rn[0][b].difference({g}), ln_rn[1][g].difference({b}))
            return (ln_rn[0] | {b: gs}, ln_rn[1] | {g: bs})

        (ln1, rn1) = reduce(addneighbor, addmatches, (lneighbors, rneighbors))
        (ln2, rn2) = reduce(delneighbor, delmatches, (ln1, rn1))
        return ln2, rn2

    def swap(self, addMatches, delMatches, newPrices):
        matches = (self.matches|addMatches).difference(delMatches)
        reduced = dict([(e, self.c[e]) for e in set(self.c.keys()).difference(delMatches)])
        added = dict([(e, float("inf")) for e in addMatches])
        changedP = dict([((0, g), newPrices[g-1]) for g in self.goods])
        return EqualityGraph(reduced|added|changedP, self.source, self.sink, self.V, newPrices, self.goods, self.buyers,
                             matches, addMatches, delMatches, self.lneighbors, self.rneighbors)










