import numpy as np
from functools import reduce

class Market:
    def __init__(self, e, u):
        if len(e) != u.shape[0]:
            raise IndexError("Dimensions Do Not Match!")
        self.e = e
        self.u = u
        self.m, self.n = u.shape

    def equalityGraph(self, prices):
        p_inv = np.vectorize(lambda p: 1/p)(prices)
        a = np.array(list(map(lambda ui: ui * p_inv, self.u)))
        alpha = np.array(list(map(lambda i: max(a[i]), range(self.m))))
        match = np.array(list(map(lambda i: a[i] == alpha[i], range(self.m))))


    def addGoods(self, i, vec, condition):
        addGood = lambda gs, j_b: gs|{(j_b[0], i)} if condition(j_b[1]) else gs
        return reduce(addGood, zip(range(len(vec)), vec), set())


