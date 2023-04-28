from FlowNetwork import FlowNetwork
import pandas as pd
from functools import reduce


class CSVFlowNetwork(FlowNetwork):
    def __init__(self, file):
        df = pd.read_csv(file, index_col=(0, 1))
        appendCapacity = lambda cm, e: (cm[0]|{e: df.at[e, 'Capacity']}, max(cm[1], e[1]))
        (capacity, m) = reduce(appendCapacity, df.index, ({}, 0))
        super().__init__(m + 1, capacity)

