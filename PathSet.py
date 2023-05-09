from itertools import product
from functools import reduce
from tail_recursive import tail_recursive as tail

class PathSet:
    def __init__(self, paths, sort = True):
        filtered = self.filter(lambda p: p is not None, paths)
        self.paths = sorted(filtered) if sort else filtered

    def __str__(self):
        return "PathSet:" + "\n" + "\n".join(str(p) for p in self.paths)

    def __repr__(self):
        return "PathSet:" + "\n" + "\n".join(str(p) for p in self.paths)

    def __hash__(self):
        return self.paths

    def __iter__(self):
        return iter(self.paths)

    def __mul__(self, other):
        return PathSet(list(map(lambda paths: paths[0] * paths[1], product(self.paths, other.paths))))

    def __add__(self, other):
        return PathSet(self.paths + other.paths, sort = False)

    def filter(self, condition, paths = None):
        paths = self.paths if paths is None else paths
        paths_filtered = [[p] if condition(p) else [] for p in paths]
        return reduce(lambda l1, l2: l1 + l2, paths_filtered, [])

    def finishedPath(self):
        go = tail(lambda paths: None if len(paths) == 0 else paths[0] if paths[0].isDone() else go.tail_call(paths[1:]))
        return go(self.paths)

    def isDone(self):
        return len(self.paths) == 0 or reduce(lambda b1, b2: b1 or b2.isDone(), self.paths, False)


