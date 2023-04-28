from ResPath import ResPath
from PathSet import PathSet
from functools import reduce
from math import pow
from tail_recursive import tail_recursive as tail

class ResidualNetwork:
    def __init__(self, fNetwork, f):
        self.fNetwork = fNetwork
        self.f = f
        res_forward = PathSet(list(map(lambda e: ResPath(e, (0,), fNetwork.c[e] - f[e], fNetwork.sink), fNetwork.c.keys())))
        res_backward = PathSet(list(map(lambda e: ResPath(self.direct(e, True), (1,), f[e], fNetwork.sink), fNetwork.c.keys())))
        condition = lambda p: p.res != 0
        combiner = lambda p1, p2: p1 if p1.res >= p2.res else p2
        self.edges = self.groupBy(lambda p: p.path, combiner, self.filter(condition, res_forward + res_backward))
        #print("Residual Edges:")
        #print(self.edges)
        add_neighbor = lambda d, edge: d|{edge.path[0]: d.get(edge.path[0], PathSet([])) + PathSet([edge])}
        self.neighbor_edges = reduce(add_neighbor, self.edges, dict([(v, PathSet([])) for v in range(fNetwork.n)]))

    def direct(self, edge, flip):
        return (edge[1], edge[0]) if flip else edge

    def filter(self, condition, paths):
        paths_filtered = [[p] if condition(p) else [] for p in paths]
        return reduce(lambda l1, l2: l1 + l2, paths_filtered, [])

    def groupBy(self, key, combiner, paths):
        add_ResPath = lambda d, path: d | {key(path): d.get(key(path), []) + [path]}
        resPathByPath = reduce(add_ResPath, paths, {})
        return [reduce(combiner, resPathList) for resPathList in resPathByPath.values()]

    def neighbor_paths(self, path, searched = set()):
        onlyNew = lambda p: (p.tail not in searched) and (p.head not in searched)
        return PathSet([path]) * PathSet(self.filter(onlyNew, self.neighbor_edges[path.head]))

    def branch(self, pathset, searched):
        def findNew(ps_s, path):
            newPaths = self.neighbor_paths(path, ps_s[1])
            return (ps_s[0] + newPaths, ps_s[1].union({path.head}))
        return reduce(findNew, pathset, (PathSet([]), searched))

    def augmentingPathBFS(self):
        def go(pathset, searched):
            if len(pathset.paths) == 0:
                return None
            else:
                finished = pathset.finishedPath()
                if finished is not None:
                    return finished
                else:
                    return go(*self.branch(pathset, searched))
        return go(self.neighbor_edges[0], {0})

    def augmentingPathDFS(self):
        @tail
        def go(pathlist, searched):
            #print("%%%%%%%%%%")
            #print("Paths: {}".format(PathSet(pathlist, sort = False)))
            if len(pathlist) == 0:
                return None
            else:
                path = pathlist[0]
                if path.isDone():
                    return path
                else:
                    newPaths = self.neighbor_paths(path, searched).paths
                    return go.tail_call(newPaths + pathlist[1:], searched.union({path.head}))
        return go(self.neighbor_edges[0].paths, {0})

    def augmentFlow(self, resPath, flow = None):
        flow = self.f if flow is None else flow
        if resPath is None:
            return flow
        else:
            signedPath = zip(resPath.path[1:], resPath.sign)
            augment = lambda f, e, s: f|{self.direct(e, bool(s)): f[self.direct(e, bool(s))] + int(pow(-1, s)) * resPath.res}
            augmentThrough = lambda flow_u, v_s: (augment(flow_u[0], (flow_u[1], v_s[0]), v_s[1]), v_s[0])
            return reduce(augmentThrough, signedPath, (flow, 0))[0]




