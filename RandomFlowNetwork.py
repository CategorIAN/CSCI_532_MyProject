from FlowNetwork import FlowNetwork
import random
import math

class RandomFlowNetwork(FlowNetwork):
    def __init__(self, n):
        self.n = n
        self.source = 0
        self.sink = n - 1
        self.intNodes = set(range(1, n - 1))
        numEdges = random.randint(n // 2, n * n - 2 * n + 2)
        incident_edges = set([self.incidentEdge(True)(self.source)] +
                             list(map(self.incidentEdge(), self.intNodes)) + [self.incidentEdge(False)(self.sink)])
        more_edges = set([self.edge() for i in range(numEdges - len(incident_edges))])
        edges = incident_edges.union(more_edges)
        capacity = dict(map(lambda e: (e, random.randint(0, 5)), edges))
        super().__init__(n, capacity)

    def incidentEdge(self, head = None):
        head = random.choice([True, False]) if head is None else head
        def f(u):
            if head:
                v = random.choice(tuple(self.intNodes.union({self.sink}).difference({u})))
                return (u, v)
            else:
                v = random.choice(tuple(self.intNodes.union({self.source}).difference({u})))
                return (v, u)
        return f

    def edge(self):
        u = random.choice(tuple(self.intNodes.union({self.source, self.sink})))
        head = True if u == self.source else False if u == self.sink else None
        return self.incidentEdge(head)(u)


