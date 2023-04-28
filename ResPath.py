
class ResPath:
    def __init__(self, path, sign, res, sink):
        self.path = path
        self.sign = sign
        self.res = res
        self.sink = sink
        self.tail = self.path[0]
        self.head = self.path[-1]
        self.traversed = set(self.path[:-1])

    def __str__(self):
        return "Path{{{}, {}}}".format(self.path, self.res)

    def __repr__(self):
        return "Path{{{},{}}}".format(self.path, self.res)

    def __hash__(self):
        return (self.path, self.sign, self.res, self.sink)

    def __lt__(self, other):
        def compare(p, q):
            return (len(p) != 0) and ((p[0] < q[0]) or (p[0] == q[0] and compare(p[1:], q[1:])))
        return len(self.path) < len(other.path) or (len(self.path) == len(other.path) and compare(self.path, other.path))

    def __eq__(self, other):
        def compare(p, q):
            return p[0] == q[0] and compare(p[1:], q[1:])
        return len(self.path) == len(other.path) and compare(self.path, other.path)

    def __mul__(self, other):
        if self.isDone():
            return self
        elif self.head == other.tail and other.head not in self.path:
            path = self.path + other.path[1:]
            sign = self.sign + other.sign
            res = min(self.res, other.res)
            return ResPath(path, sign, res, self.sink)

    def isDone(self):
        return self.head == self.sink






