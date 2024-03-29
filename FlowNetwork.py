from ResidualNetwork import ResidualNetwork
from tail_recursive import tail_recursive
import time
from functools import reduce
import pandas as pd
from ResPath import ResPath

class FlowNetwork:
    def __init__(self, capacity, source = None, sink = None, V = None):
        self.source = source if source is not None else min(V)
        self.sink = sink if sink is not None else max(V)
        self.intNodes = set(range(source + 1, sink)) if V is None else V.difference({source, sink})
        self.V = {self.source, self.sink}|self.intNodes if V is None else V
        self.c = capacity

    def __str__(self):
        return "Network" + str(sorted(list(self.c.items())))

    def __repr__(self):
        return "Network" + str(sorted(list(self.c.items())))

    def initFlow(self):
        return dict(map(lambda e: (e, 0), self.c.keys()))

    def fordFulkerson(self, EdKarp = True, mincut = False, count = False):
        @tail_recursive
        def go(flow, i):
            resNetwork = ResidualNetwork(self, flow)
            #print(EdKarp)
            resPath = resNetwork.augmentingPathBFS(mincut) if EdKarp else resNetwork.augmentingPathDFS(mincut)
            #print("++++++++++++++++++++++++++++++")
            #print("ResPath: {}".format(resPath))
            #print("++++++++++++++++++++++++++++++")
            if type(resPath) is not ResPath:
                return ((flow, resPath, i) if mincut else (flow, i)) if count else ((flow, resPath) if mincut else flow)
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






