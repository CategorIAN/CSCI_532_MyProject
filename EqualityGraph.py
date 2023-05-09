from FlowNetwork import FlowNetwork
from functools import reduce
import numpy as np

class EqualityGraph (FlowNetwork):
    def __init__(self, capacity, source, sink, V, matches, prices, addMatches = None, delMatches = None,
                 lneighbors = None, rneighbors = None):
        super().__init__(capacity, source, sink, V)
        self.matches = matches
        self.prices = prices
        self.lneighbors, self.rneighbors = self.neighbors(addMatches, delMatches, lneighbors, rneighbors)


    def neighbors(self, addMatches = None, delMatches = None, lneighbors = None, rneighbors = None):
        addmatches = self.matches if addMatches is None else addMatches
        delmatches = set() if delMatches is None else delMatches
        goods, buyers = zip(*self.matches)
        lneighbors = dict(list(map(lambda b: (b, set()), buyers))) if lneighbors is None else lneighbors
        rneighbors = dict(list(map(lambda g: (g, set()), goods))) if rneighbors is None else rneighbors
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
        changedP = dict([((0, j+1), newPrices[j]) for j in range(len(self.prices))])
        return EqualityGraph(reduced|added|changedP, self.source, self.sink, self.V, matches, newPrices, addMatches,
                             delMatches, self.lneighbors, self.rneighbors)










