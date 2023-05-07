from FlowNetwork import FlowNetwork
from functools import reduce
import numpy as np

class EqualityGraph (FlowNetwork):
    def __init__(self, capacity, source, sink, V, matches, prices):
        super().__init__(capacity, source, sink, V)
        self.matches = matches
        self.prices = prices
        self.lneighbors, self.rneighbors = self.neighbors()

    def neighbors(self):
        goods, buyers = zip(*self.matches)
        lneighbors = dict(list(map(lambda b: (b, set()), buyers)))
        rneighbors = dict(list(map(lambda g: (g, set()), goods)))
        def addneighbor(ln_rn, e):
            (g, b) = e
            gs, bs = (ln_rn[0][b]|{g}, ln_rn[1][g]|{b})
            return (ln_rn[0]|{b: gs}, ln_rn[1]|{g: bs})
        return reduce(addneighbor, self.matches, (lneighbors, rneighbors))





