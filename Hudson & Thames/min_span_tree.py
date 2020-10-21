'''
     Minimum Spanning Tree testing file.
     Brief: Implementation of Kruskal's procedure for maximizing correlation
            degree between financial instrument price history data.
     Author: "Gaston Solari Loudet"
     Version: 1.0
     To be reviewed by: Hudson & Thames (c)
     
     sources: [1] "A Review of Two Decades of Correlations, Hierarchies,
                  Networks and Clustering in Financial Markets" by Gautier,
                  Marti; Nielsen, Frank; Binkowski, Mikołaj; Donnat, Philippe.
                  (https://arxiv.org/pdf/1703.00485.pdf), pages 1-2.
              [2] "Wikipedia: Minimum Spanning Tree"
                  (https://en.wikipedia.org/wiki/Minimum_spanning_tree)
              [3] "Official NetworkX library documentation"
                  (https://networkx.github.io/)
'''

import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import networkx as nx
import matplotlib.pyplot as pt


class min_span_tree:

    def __init__(self, dataFrame):
        '''
        Constructor: show MST data when invoked.
            -> Inputs: - Dataframe. Each one of its "n" columns should be the
                         market price history for one chosen instrument.
            -> Output: "min_span_tree" class instance with the following
                        attributes:
                - "Labels": dictionary storing the alphabetically ordered
                            instrument designation labels.
                - "Dists":  dataframe based in distance matrix "Dij", shaped
                            as ordered triples of the form: "(i, j, Dij)".
                - "Edges":  dataframe holding the minimum spanning tree edges
                            computed by applying Kruskal's algorithm over the
                            "Dists" dataframe.
                - "Graph":  NetworkX graph instance based in "Labels" as nodes
                            and "Edges" as edges.
                - "Handle": Handle towards NetworkX graphical object.
        '''
        
        # Retrieve column names from input.
        n_nodes = len(dataFrame.columns)
        self.Labels = {node : dataFrame.columns[node]
                        for node in range(n_nodes)}
        # Get correlation distance matrix.
        Matrix = min_span_tree.corr_distance(dataFrame)
        
        # Array assembly of node pairs and their distances.
        Matrix = np.triu(Matrix)
        nPairs = np.triu_indices(n_nodes)
        Dists = (nPairs[0], nPairs[1], Matrix[nPairs])
        Dists = np.stack(Dists, axis = 1)
        # Keep only rows which link two distinct nodes.
        Dists = Dists[Dists[:, 0] != Dists[:, 1]]
        
        # Convert "D{i, j}" numpy array into DataFrame.
        colabels = ["i", "j", "Dij"]
        self.Dists = pd.DataFrame(Dists, columns = colabels)
        # Store dataframe with nodes as ints in class.
        self.Dists["i"] = self.Dists["i"].astype("int")
        self.Dists["j"] = self.Dists["j"].astype("int")
        # Run Kruskal's algorithm
        self.Edges = min_span_tree.kruskal(self.Dists)
        
        # Build Graph instance from NetworkX Library
        self.Graph = nx.Graph()
        self.Graph.add_nodes_from(range(n_nodes))
        self.Graph.add_weighted_edges_from(np.array(self.Edges))
        
        
    def __str__(self):
        '''
        Dunder method: show MST data when invoked.
            -> Inputs: [None]
            -> Output: - String with edge and node number depiction.
        '''
    
        print("\nMST links:")
        n_edges = len(self.Edges)
        n_nodes = len(self.Labels)
        # Exhibit MST data but with tickers instead of nodes.
        print(self.Edges.replace(self.Labels))
        return("\nEdges: %d | Nodes: %d" % (n_edges, n_nodes))
        
        
    def plot(self):
        '''
        Class method: Creation and plot of graph diagram.
        Execution for more than 120 nodes is not recommended.
            -> Inputs: [None]
            -> Output: [None]
        '''
    
        # Font size, Node size, Position.
        fs = 40//(len(self.Labels)**(1/3))
        ns = 1200//(len(self.Labels)**(1/2))
        ps = nx.spring_layout(self.Graph)
        ''' "ps" to be changed in the future: can cause overlapping. '''
        
        # Plot in console and print labels for further reference.
        self.Handle = nx.draw(self.Graph, pos = ps, node_size = ns)
        nx.draw_networkx_labels(self.Graph, pos = ps, font_size = fs)
        pt.show()
        print("\n %s" % self.Labels)
        return None
    

    def corr_distance(dataFrame):
        '''
        Static method: Calculation of correlation distance matrix.
            -> Inputs: - Dataframe holding closing prices of "n"
                         instruments for the same time period.
            -> Output: - Symmetrical "n x n" numpy array with zeros
                         in diagonal.
        '''
    
        # Find out log returns and clear rows with "NaNs".
        logReturns = np.log(dataFrame) \
                   - np.log(dataFrame.shift(1))
        logReturns.dropna(how = "all", inplace = True)

        # Calculate correlation matrices.
        matrix_corr = np.array(logReturns.corr())
        matrix_dist = np.sqrt(2*(1 - matrix_corr))
        
        return matrix_dist   # Return distance numpy array.
    
    def kruskal(dataFrame):
        '''
        Static method: Computation of minimum spanning tree edge list
        by means of Kruskal's algorithm procedure as described in source [1]
        at the beginning of this document.
            -> Inputs: - Dataframe holding ordered triples of the form
                         "(start node, end node, edge weight)". For this
                         case, weight equals correlation distance.
            -> Output: - Dataframe keeping only the minimum spanning tree
                         edges from the input dataframe.
        '''
    
        # Let it work for other stuff (last column != "Dij").
        W = dataFrame.columns[-1]
        # Sort in ascending order (largest "W" on bottom).
        Weights = dataFrame.sort_values(W)
        
        # Where only the minimum distance edges will be stored.
        Edges = pd.DataFrame(None, columns = Weights.columns)
        # Will hold sets of nodes already connected somehow.
        subTrees = []
        
        # Try to continuously add edges by growing distances.
        for row in Weights.index:
            # Retrieve nodes of edge candidate.
            node_i = int(Weights["i"][row])
            node_j = int(Weights["j"][row])
            # Lists. Will hold in which subTrees are nodes in.
            node_i_in = None ; node_j_in = None
            # Flag. Will be true when new edge can be appended.
            add_edge = False
            # Check which subTree does each node already link to.
            for s in range(len(subTrees)):
                if (node_i in subTrees[s]): node_i_in = s
                if (node_j in subTrees[s]): node_j_in = s
                
            # Case 1: nodes totally new to MST.
            if (node_i_in == node_j_in == None):
                add_edge = True
                # Add a new subtree set with said nodes.
                subTrees.append(set([node_i, node_j]))
            if (node_i_in != node_j_in):
                add_edge = True
                
                # Case 2: only one of the nodes connects to MST.
                if (None in [node_i_in, node_j_in]):
                    # Add the other (new) node to same subtree.
                    if (node_i_in != None): subTrees[node_i_in].add(node_j)
                    if (node_j_in != None): subTrees[node_j_in].add(node_i)
                    
                else: # Case 3: both nodes in different subtrees.
                    # Union of both subtree sets.
                    subTrees[node_i_in].update(subTrees[node_j_in])
                    # Remove one of both.
                    subTrees.remove(subTrees[node_j_in])
                    
            # Append edge to tree if applicable.
            if add_edge:
                Edges = Edges.append(Weights.loc[row],
                                     ignore_index = True)
        return Edges
    
    # End class declaration


if (__name__ == "__main__"):
    '''
    To be executed when this file is run as standalone.
    Might serve as one basic unit test.
    '''
    
    # Fast test: gaming businesses' stocks.
    quotes = ["NVDA", "ATVI", "NTDOY", "SNE", "MSFT", "UBI.PA", "EA"]
    
    # Market data retrieved from last year to date.
    tf = dt.date.today()
    td = dt.timedelta(days = 16)
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
    MST.plot()