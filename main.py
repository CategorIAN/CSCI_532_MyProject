from RandomFlowNetwork import RandomFlowNetwork
from ResidualNetwork import ResidualNetwork
from ResPath import ResPath
from CSVFlowNetwork import CSVFlowNetwork
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Market import Market


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
        print("DFS Path: {}".format(F.augmentingPathDFS()))
        print("BFS Path: {}".format(F.augmentingPathBFS()))
    if i == 2:
        analysis(first = 5, last = 40, step = 5, graph = True)
    if i == 3:
        G = RandomFlowNetwork(10)
        G.toCSV()
        H = CSVFlowNetwork('FNetwork.csv')
        print(H)
        print(H.n)
    if i == 4:
        e = np.array([4, 4])
        u = np.array([[4, 3, 8], [4, 5, 10]])
        M = Market(e, u)
        prices = M.initialPrices()
        N = M.equalityGraph(prices)
        m = M.adjustedDemand(prices, e)
        N_adj = M.adjustNetwork(N, m)
        #print(N_adj.sink)
        #print("N:")
        #print(N)
        #print("N'")
        print(N_adj)
        f = N_adj.fordFulkerson()
        print(f)
    if i == 5:
        e = np.array([12, 8])
        u = np.array([[4, 3, 8], [4, 5, 10]])
        M = Market(e, u)
        p = np.array([1, 1, 1])
        print(M.adjustedDemand(p, e))

if __name__ == '__main__':
    f(4)


