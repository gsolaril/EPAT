'''
     Minimum Spanning Tree testing file.
     Brief: Unit test of "min_span_tree.py" file and its class definition.
            Connectivity check based in laplacian matrix eigendecomposition.
     Author: "Gaston Solari Loudet"
     Version: 1.0
     To be reviewed by: Hudson & Thames (c)
     
     sources: [1] "Wikipedia: Algebraic connectivity"
                  (https://en.wikipedia.org/wiki/Algebraic_connectivity)
              [2] "Official NumPy documentation: 'Linear Algebra' module"
                  (https://docs.scipy.org/doc/numpy/reference/routines.
                   linalg.html)
'''

import numpy as np
import datetime as dt
import yfinance as yf
from min_span_tree import min_span_tree


def laplacian(dataFrame):
    '''
    Function: computes the laplacian matrix of the graph dataframe.
        -> Inputs: - Dataframe holding the ordered pairs or triples of the
                     form: "(start node, end node, [edge weight])"
        -> Output: - Numpy array for symmetrical laplacian matrix.
    '''
    
    # Retrieve linked node pairs.
    node_cols = dataFrame.columns[:2]
    edges = np.array(dataFrame[node_cols], dtype = "int")
    
    # Get matrix of adjacency (linkages).
    n_nodes = 1 + edges.max()
    matrix_adj = np.zeros((n_nodes, n_nodes))
    matrix_adj[edges[:, 0], edges[:, 1]] = 1  # Set linked.
    matrix_adj[edges[:, 1], edges[:, 0]] = 1  # Symmetrical.
    
    # Get matrix of degrees (amount of edges for each node).
    _, n_edges = np.unique(edges, return_counts = True)
    matrix_deg = np.diag(n_edges)
    
    # Get laplacian matrix as difference between both.
    return(matrix_deg - matrix_adj)

if (__name__ == "__main__"):
    '''
    To be executed when this file is run as standalone.
    '''

    # Import list of shares's name strings from GitHub:
    import pandas as pd
    url = "https://raw.githubusercontent.com/gsolaril/" \
        + "HudsonThames_SoldierOfFortune/master/oil.csv"
    quotes = [i for i in pd.read_csv(url, header = None)[0]]
    
    '''
    # Import list of shares's name strings from .csv file:
    import csv
    with open("oil.csv", encoding = 'utf-8-sig') as csvList:
        quotes = [i[0] for i in csv.reader(csvList)]
    '''
    
    # Market data retrieved from last year to date.
    tf = dt.date.today()
    td = dt.timedelta(days = 365)
    t0 = tf - td
    
    # Download history data for tickers from Yahoo!.
    shares = yf.download(quotes, start = t0, end = tf)
    # Leave aside all columns except from close prices.
    shares = shares["Close"].round(decimals = 4)
    # Leave aside all shares with no available data.
    shares.dropna(axis = 1, inplace = True, how = "all")
    # Fill weekend values with last valid close price.
    shares.fillna(axis = 0, inplace = True, method = "ffill")
    print("\nDataFrame size: %d x %d" % shares.shape)
    print("\nImported shares' last rows:")
    print(shares.tail(10))
    
    # Create MST instance.
    MST = min_span_tree(shares)
    print(MST)
    
    # Laplacian matrix calculation.
    matrix_lap = laplacian(MST.Edges)
    # Array of laplacian matrix eigenvalues.
    matrix_eig = np.linalg.eigvalsh(matrix_lap)
    # Algebraic connectivity as its second-smallest eigenvalue.
    matrix_con = sorted(matrix_eig)[1]
    print("\nAlgebraic connectivity: %.4f" % matrix_con)
    # Check global connectivity.
    if (round(matrix_con*1e4)/1e4 > 0):
        print("Tree is connected. Has no loose nodes.")
    else:
        print("Tree is not connected. Has at least one loose node.")