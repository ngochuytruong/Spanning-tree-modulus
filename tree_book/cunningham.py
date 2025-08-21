import networkx as nx
import numpy as np
"""
Functions for computing spanning tree modulus using Cunningham's algorithm.
"""

import networkx as nx
import numpy as np
from fractions import Fraction
from collections import UserDict
from warnings import warn

def rank(G,J):
    """
    Computes the rank of J in G.
    """

    # create a new graph with the same vertices as G
    H = nx.Graph()
    H.add_nodes_from(G.nodes)

    # connect edges from J
    H.add_edges_from(J)

    # compute the rank 
    return len(G.nodes) - nx.number_connected_components(H)

class EdgeFunction(UserDict):
    """
    Defines an edge function on a graph with symmetry x(u,v)=x(v,u) built in.
    This is accomplished by treating edge tuples as frozen sets.

    WARNING: I haven't built in any safety checks yet.
    """

    def __init__(self, G):
        """
        Initializes the EdgeFunction as the zero function on edges of G.
        """
        super().__init__()
        self.update({frozenset(e):0 for e in G.edges})
    
    def __getitem__(self, e):
        """
        Evaluates the EdgeFunction on edge j.
        """
        return super().__getitem__(frozenset(e))
    
    def __setitem__(self, e, v):
        """
        Set's the value at j to v.
        """
        super().__setitem__(frozenset(e), v)

    def __str__(self):
        keys = self.keys()
        l = ["({},{}): {}".format(u,v,self[(u,v)]) for u,v in keys]
        return ", ".join(l)
    

def create_flow_graph(G):
    """
    Creates a flow graph for applying Cunningham's algorithm.
    """

    # copy G
    F = nx.Graph(G)

    # add source and target
    F.add_node("__source")
    F.add_node("__target")

    # connect source and target
    F.add_edges_from([("__source",v) for v in G.nodes])
    F.add_edges_from([("__target",v) for v in G.nodes])

    return F

def solve_subproblem(G, F, x, e, q):
    """
    Uses max flow to complete a step of Cunningham's algorithm on
    edge e.
    """

    # set up capacities between "regular" nodes
    for u,v in G.edges:
        F[u][v]["capacity"] = x[(u,v)]

        # set up fixed weights to the target
        for v in G.nodes:
            F[v]["__target"]["capacity"] = 2*q

    # set up capacities from source
    for v in G.nodes:
        if v in e:
            F["__source"][v]["capacity"] = np.Inf
            continue

        F["__source"][v]["capacity"] = sum([x[(u,v)] for u in G[v]])

    # perform the minimum cut
    cut_value, partition = nx.minimum_cut(F, "__source", "__target")

    # find epsilon from the cut value
    eps = cut_value // 2 - q - sum([x[e] for e in G.edges])

    # the critical set is the partition containing the source
    if "__source" in partition[0]:
        crit_set = partition[0]
    else:
        crit_set = partition[1]
    
    # find all edges contained in the critical set
    crit_set.remove("__source")
    crit_edges = set((u,v) for u,v in G.edges if u in crit_set and v in crit_set)

    return eps, crit_edges

def cunningham_min(G, F, p, q):
    """
    Finds a P(qf)-basis for p along with a tight set.
    """

    # initialize x
    x = EdgeFunction(G)

    # edges already found
    A = set()

    # loop over the edges
    for e in G.edges:

        # skip the edge if it's already critical
        if e in A:
            continue

        # solve the subproblem
        eps, crit_edges = solve_subproblem(G, F, x, e, q)

        # see if we can take the full step to y
        if eps < p - x[e]:
            A = A.union(crit_edges)
        else:
            eps = p - x[e]
        x[e] += eps

    # return x and the tight set    
    return x, A

def graph_vulnerability(G, F):
    """
    Uses a binary search to find the vulnerability of a graph.  Also returns
    the optimal tight set.
    """

    # count the number of edges and vertices
    m, n = len(G.edges), len(G.nodes)

    # create the set of all possible theta values
    Theta = set(Fraction(p,q) for q in range(1,m+1) for p in range(1,min(n-1,q)+1))

    # turn it into a sorted list
    Theta = sorted(list(Theta))

    # initialize the edge set
    crit_set = None
    
    # initialize the left and right limits of the search
    lb, ub = 0, len(Theta)

    # perform the search
    while(lb < ub):

        # find the midpoint value
        mid = (ub+lb)//2
        p,q = Theta[mid].numerator, Theta[mid].denominator

        # ask the oracle
        x,J = cunningham_min(G, F, p, q)
        xE = sum(x.values())

        if(xE >= q*(n-1)):
            ub = mid
            crit_set = J
        else:
            lb = mid+1
    
    return Theta[lb], J

def modulus(G):
    """
    Uses Cunningham's algorithm to compute the modulus of G.
    """

    # initialize eta^*
    eta_star = EdgeFunction(G)

    # initialize edge counter
    remain = len(G.edges)

    # list of graphs to process
    # (in case G is disconnected, we break it into connected components)
    process = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    # print table header
    print("| {:^12} | {:^12} | {:^12} | {:^12} |".format("eta", "num edge", "edge_remain", "comp remain"))
    print( ("+" + "-"*14)*4 + "+")

    # loop as long as there are still edges to process
    while process:

        # get a graph to process
        H = process.pop()

        # if it's trivial, go to the next
        if len(H.edges) == 0:
            continue

        # create a flow graph to work on
        F = create_flow_graph(H)

        # compute the vulnerability
        theta, J = graph_vulnerability(H, F)

        # see if we got the whole set in J
        m = len(H.edges)
        p,q = theta.numerator, theta.denominator
        if len(J) == m:
            warn("Got entire edge set as tight.  Rerunning.")
            pp = p*m**2 - q
            qq = q*m**2
            _, J = cunningham_min(G, F, pp, qq)

        # compute the complement of J
        crit_set = [(u,v) for u,v in H.edges if (u,v) not in J and (v,u) not in J]

        # update eta^* on the critical set
        for e in crit_set:
            eta_star[e] = theta
        
        # remove critical edges from H and split into connected components
        H.remove_edges_from(crit_set)
        comps = [H.subgraph(c).copy() for c in nx.connected_components(H)]
        assert(theta == Fraction(len(comps)-1, len(crit_set)))
        process.extend(comps)

        # update the table
        remain -= len(crit_set)
        print("| {:^12} | {:^12} | {:^12} | {:^12} |".format(str(theta), len(crit_set), remain, len(process)))

    return eta_star

def create_house_graph():
    """
    Creates a house graph for testing.
    """

    G = nx.cycle_graph(5)
    G.add_edge(0,2)

    pos = nx.circular_layout(G)
    return G, pos

def create_nested_graph(n=10):
    """
    Creates a nested graph for testing.
    """

    G = nx.Graph()

    # add complete set of edges
    for i in range(n):
        for j in range(i+1,n):
            G.add_edge(i,j)

    for i in range(n):
        for b in (-1,0,1):
            j = (i+b)%n + n
            G.add_edge(i,j)
        j = (i+1)%n + n
        G.add_edge(i+n, j)

    for i in range(n):
        G.add_edge(i+n,i+2*n)
        j = (i+1)%n + 2*n
        G.add_edge(i+2*n, j)

    # position the nodes in concentric circles
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos = {i:(np.cos(th[i]),np.sin(th[i])) for i in range(n)}
    pos.update({n+i:(2*np.cos(th[i]),2*np.sin(th[i])) for i in range(n)})
    pos.update({2*n+i:(3*np.cos(th[i]),3*np.sin(th[i])) for i in range(n)})

    return G,pos
