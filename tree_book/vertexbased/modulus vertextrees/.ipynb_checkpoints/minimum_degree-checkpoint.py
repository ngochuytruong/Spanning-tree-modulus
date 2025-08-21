import networkx as nx
import numpy as np
def minimum_degree(G,S):
    H = G
    for (u,v) in H.edges():
        if u in S:
            if v in S:
                H[u][v]['weight']= 2
            else:
                H[u][v]['weight']= 1
        else:
            if v in S:
                H[u][v]['weight']= 1
            else:
                H[u][v]['weight']= 0
    T = list(nx.minimum_spanning_edges(H, weight='weight', data=True)) 
    m = sum( H[u][v]['weight'] for (u,v,w) in T)
    return m

def minimum_degree_tree(G,S):
    H = G
    for (u,v) in H.edges():
        if u in S:
            if v in S:
                H[u][v]['weight']= 2
            else:
                H[u][v]['weight']= 1
        else:
            if v in S:
                H[u][v]['weight']= 1
            else:
                H[u][v]['weight']= 0
    T = list(nx.minimum_spanning_edges(H, weight='weight', data=False)) 
    return  T
class VertexbasedMinimumSpanningTree:
    """
    Functor class for finding the minimum rho-length spanning tree.
    """
    
    def __init__(self, G):
        
        # remember the graph
        self.G = G
        
        # enumerate the edges so we can keep track of them when
        # processing a spanning tree
        for i, (u,v) in enumerate(G.edges()):
            G[u][v]['enum'] = i
                
    def __call__(self, rho, tol):
        
        # assign weight to the graph edges
        for i, (u,v) in enumerate(self.G.edges()):
            self.G[u][v]['weight'] = rho[u]+rho[v]
            
        # find a minimum spanning tree
        T = list(nx.minimum_spanning_edges(self.G, weight='weight', data=False))
        H = nx.Graph(T)
        # form the row vector
        n = np.zeros(rho.shape)
        for u in H.nodes():
            n[u] = H.degree[u]
            
        return T, n