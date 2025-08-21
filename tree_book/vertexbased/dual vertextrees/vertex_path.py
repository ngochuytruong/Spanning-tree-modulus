import networkx as nx
import numpy as np
class ShortestVertexPath:
    """
    Functor class for finding the shortest rho-length path between two sets of nodes.
    """

    # the dummy source and target node names
    src = '__source__'
    tgt = '__target__'
    
    def __init__(self, G, S, T):
                
        # remember the graph, source and target sets
        self.G = G
        self.S = S
        self.T = T
        
        # enumerate the edges so we can keep track of them
        # when processing a path
        for i, (u,v) in enumerate(G.edges()):
            G[u][v]['enum'] = i
        for i,u in enumerate(G.nodes()):
            G.nodes[u]['enum']=i
        
        # make a copy of G to work on
        self.H = G.copy()
        self.H.add_node(self.src)
        self.H.add_node(self.tgt)
        # link the dummy nodes
        for v in S:
            self.H.add_edge(self.src, v)
        for v in T:
            self.H.add_edge(v, self.tgt)
            
    def __call__(self, rho, tol):
        
        # assign rho to the graph edges
        for i, (u,v) in enumerate(self.G.edges()):
       
                self.H[u][v]['weight'] = rho[self.G.nodes[u]['enum']] + rho[self.G.nodes[u]['enum']]
        for i, (u,v) in enumerate(self.H.edges()):
            if u == self.src:
                self.H[u][v]['weight'] = rho[self.G.nodes[u]['enum']]   
        for i, (u,v) in enumerate(self.H.edges()):
            if v == self.tgt:
                self.H[u][v]['weight'] = rho[self.G.nodes[u]['enum']] 
        
            
        p = nx.shortest_path(self.H, self.src, self.tgt, weight ='weight')
        
        # the actual path omits the source and target dummy nodes
        p = p[1:-1]
        
        # form the row vector
        n = np.zeros(rho.shape)
        for u in p:
            n[self.G.nodes[u]['enum']]=1
            
        return p, n

