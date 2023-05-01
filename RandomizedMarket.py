from Market import Market
from random import random
import numpy as np
from math import ceil

class RandomizedMarket(Market):
    def __init__(self, m, n):
        e = np.vectorize(lambda xi: int(50*random())+1)(np.zeros(m))
        u = np.vectorize(lambda xi: int(50*random())+1)(np.zeros((m, n)))
        print("e: {}".format(e))
        print("u: {}".format(u))
        super().__init__(e, u)
