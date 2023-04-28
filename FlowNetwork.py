from ResidualNetwork import ResidualNetwork
from tail_recursive import tail_recursive
import time
from functools import reduce
import pandas as pd

class FlowNetwork:
    def __init__(self, n, capacity):
        self.n = n
        self.source = 0
        self.sink = n - 1
        self.intNodes = set(range(1, n - 1))
        self.c = capacity

    def __str__(self):
        return "Network" + str(self.c)

    def initFlow(self):
        return dict(map(lambda e: (e, 0), self.c.keys()))

    def fordFulkerson(self, EdKarp, count):
        @tail_recursive
        def go(flow, i):
            resNetwork = ResidualNetwork(self, flow)
            resPath = resNetwork.augmentingPathBFS() if EdKarp else resNetwork.augmentingPathDFS()
            print("++++++++++++++++++++++++++++++")
            print("ResPath: {}".format(resPath))
            print("++++++++++++++++++++++++++++++")
            if resPath is None:
                return (flow, i) if count else flow
            else:
                return go.tail_call(resNetwork.augmentFlow(resPath, flow), i + 1)
        return go(self.initFlow(), 0)

    def appendtime(self, function):
        def f(*args, **kwargs):
            start = time.time()
            result = function(*args, **kwargs)
            duration = time.time() - start
            return result + (duration,) if type(result) is tuple else (result, duration)
        return f

    def flowSize(self, f):
        addFlow = lambda s, e: s + int(e[0] == 0) * f[e]
        return reduce(addFlow, f.keys(), 0)

    def toCSV(self, file = "FNetwork"):
        pd.Index(self.c.keys(), name =
        ('Tail', 'Head')).to_series(name='Capacity').map(lambda e: self.c[e]).to_csv("{}.csv".format(file))






