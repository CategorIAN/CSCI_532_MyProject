from RandomFlowNetwork import RandomFlowNetwork
from ResidualNetwork import ResidualNetwork
from ResPath import ResPath
from CSVFlowNetwork import CSVFlowNetwork
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Market import Market
from RandomizedMarket import RandomizedMarket


def analysis(first, last, step, graph = False):
    def f(i):
        print("----------------------------------------------------------------------------------------")
        print("Size is {}".format(i))
        G = RandomFlowNetwork(i)
        print("G: {}".format(G))
        print("================")
        print("DFS")
        print("================")
        DFS = G.appendtime(G.fordFulkerson)(EdKarp = False, count = True)
        flow = DFS[0]
        print("Flow: {}".format(flow))
        print("Flow Size: {}".format(G.flowSize(flow)))
        print("================")
        print("BFS")
        print("================")
        BFS = G.appendtime(G.fordFulkerson)(EdKarp = True, count = True)
        flow = BFS[0]
        print("Flow: {}".format(flow))
        print("Flow Size: {}".format(G.flowSize(flow)))
        print("##########")
        #print(DFS[1:] + BFS[1:])
        return DFS[1:] + BFS[1:]
    df = pd.DataFrame(index = range(first, last + 1, step), columns = ['DFS_Count', 'DFS_Time', 'BFS_Count', 'BFS_Time'])
    for i in df.index:
        df.loc[i, :] = f(i)
    if graph:
        createChart(df = df)
    df.to_csv("Analysis_from_{}_to_{}.csv".format(first, last))


def createChart(df = None, file = None):
    df = pd.read_csv(file, index_col = 0) if df is None else df
    plt.figure(1)
    ax = plt.subplot(2, 1, 1)
    ax.title.set_text('Number of Augments vs Size')
    plt.plot(df.index, df['DFS_Count'], **{'color': 'blue', 'marker': 'o'}, label='DFS')
    plt.legend()
    plt.plot(df.index, df['BFS_Count'], **{'color': 'red', 'marker': 'o'}, label = 'BFS')
    plt.legend()

    ax = plt.subplot(2, 1, 2)
    ax.title.set_text('Time (s) vs Size')
    plt.plot(df.index, df['DFS_Time'], **{'color': 'blue', 'marker': 'o'}, label='DFS')
    plt.legend()
    plt.plot(df.index, df['BFS_Time'], **{'color': 'red', 'marker': 'o'}, label='BFS')
    plt.legend()

    plt.show()


def f(i):
    if i == 1:
        G = RandomFlowNetwork(10)
        print("G: {}".format(G))
        F = ResidualNetwork(G, G.initFlow())
        p = F.augmentingPathBFS()
        print("Path Type: {}".format(type(p) is ResPath))
        print("BFS Path: {}".format(p))
    if i == 2:
        e = np.array([4, 4])
        u = np.array([[4, 3, 8], [4, 5, 10]])
        M = Market(e, u)
        prices = M.initialPrices()
        f = M.balancedFlow(prices)
        print(f)
    if i == 3:
        e = np.array([4, 4])
        u = np.array([[4, 3, 8], [4, 5, 10]])
        M = Market(e, u)
        p = np.array([1, 1, 2])
        N = M.equalityGraph(p)
        print(N)
        print(M.inducedNetworks(N, {0, 1, 2, 4}))
    if i == 4:
        M = RandomizedMarket(2, 3)
        prices = M.initialPrices()
        f = M.balancedFlow(prices)
        print(f)

if __name__ == '__main__':
    f(4)


