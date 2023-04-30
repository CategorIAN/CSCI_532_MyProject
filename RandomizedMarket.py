from Market import Market
from random import random
import numpy as np
from math import ceil

class RandomizedMarket(Market):
    def __init__(self, m, n):
        e = np.vectorize(lambda xi: ceil(50*random()))(np.zeros(m))
        u = np.vectorize(lambda xi: ceil(50*random()))(np.zeros((m, n)))
        super().__init__(e, u)
