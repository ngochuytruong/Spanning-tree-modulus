import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

""" 
Impotant function
create_nested_graph(n=10)
maximum_denseness(G) given weights
modulus(G), given weights
reinforcement(G,c), given costs c
assignw(G,sigma)
drawedge(G,pos,w)
Reff(G,u,v) given weights
weta(G)
matrixtree(G)
"""
# Example


def matrixtree(G):
    for i,(u,v) in enumerate(G.edges()):
        G[u][v]['enum'] = i
    n = len(G.nodes)
    m = len(G.edges())
    N=[]
    for T in combinations(G.edges,n-1):
            H = nx.Graph(T)
            if nx.is_tree(H):   
                v = np.zeros(m)
                for (a,b) in T:
            
                    ind = G[a][b]['enum']
                    v[ind] = 1
                N.append(v)       
    return(np.array(N))

def reffedge(G):
    reff = np.ones(len(G.edges()))
    for i,(u,v) in enumerate(G.edges()):
        reff[i] = Reff(G,u,v)
    return reff

def edgeprob(G):
    N = matrixtree(G)
    t = np.shape(N)[0]
    tree = np.ones(t)
    for i in range(t):
        for j ,(u,v) in enumerate(G.edges()):
            tree[i] = tree[i]*((G[u][v]['weight'])**(N[i,j]))
    edgepro = np.matmul(np.transpose(N),tree)/(sum(tree))
    return edgepro



def Reff(G,u,v):
    eff = nx.resistance_distance(G, u,v , weight='weight', invert_weight=False)
    return eff

def assignw(G,sigma):
    for i,(u,v) in enumerate(G.edges()):
        G[u][v]['weight']  = sigma[i]

    
def drawedge(G,pos,w):
    nx.draw(G, pos, node_color="black", node_size=50, 
            edge_color=[w[G[u][v]['enum']] for (u,v) in G.edges], edge_cmap=plt.cm.Set2)
    
    
    
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
def subproblem_inf(G, x, a):
    """
    Uses max flow to complete a step of Cunningham's algorithm on
    edge e.
    """
    F = create_flow_graph(G)
    # set up capacities between "regular" nodes
    for u,v in G.edges:
        F[u][v]["capacity"] = (x[G[u][v]['enum']])/2

        # set up fixed weights to the target
        for v in G.nodes:
            F[v]["__target"]["capacity"] = 1

    # set up capacities from source
    for v in G.nodes:
        if v == a:
            F["__source"][v]["capacity"] = np.Inf
            continue

        F["__source"][v]["capacity"] = sum([(x[G[u][v]['enum']]) for u in G[v]])/2

    # perform the minimum cut
    cut_value, partition = nx.minimum_cut(F, "__source", "__target")

    # find epsilon from the cut value
    minimum_value = cut_value  - 1 - sum([x[G[u][v]['enum']] for (u,v) in G.edges])

    # the critical set is the partition containing the source
    if "__source" in partition[0]:
        crit_set = partition[0]
    else:
        crit_set = partition[1]
    
    # find all edges contained in the critical set
    crit_set.remove("__source")
    #crit_edges = set((u,v) for u,v in G.edges if u in crit_set and v in crit_set)

    return minimum_value, crit_set
def subproblem(G, x):
    value, cut = subproblem_inf(G, x, list(G.nodes)[0])
    for a in G.nodes():
        minimum_value, crit_set = subproblem_inf(G, x, a)
        if minimum_value < value:
            value = minimum_value
            cut = crit_set
    return value, cut
def maximum_denseness(G):
    for i,(u,v) in enumerate(G.edges()):
        G[u][v]['enum']  = i
    B = G.nodes()
    b = sum([G[u][v]['weight'] for (u,v) in G.edges()])/(len(B) - 1)
    x = [G[u][v]['weight']/b for i,(u,v) in enumerate(G.edges())]
    g_b, cut = subproblem(G, x)
    while round(g_b,4) < 0:
        B = cut
        edges_cut = list((u,v) for (u,v) in G.edges if u in cut and v in cut)
        b = sum([G[u][v]['weight'] for (u,v) in edges_cut])/(len(B) - 1)
        x = [G[u][v]['weight']/b for i,(u,v) in enumerate(G.edges())]
        g_b, cut = subproblem(G, x)
    return b,B
def shrink(G,subgraph):
    H = nx.Graph()
    H.add_node('new_node')
    for (u,v) in G.edges():
        if u not in subgraph and v not in subgraph:
            H.add_edge(u,v)
            H[u][v]['weight'] = G[u][v]['weight']
            H[u][v]['ori_enum'] = G[u][v]['ori_enum']
    for u in G.nodes():
        if u not in subgraph:
            original = []
            weight = 0
            for v in subgraph:
                if (u,v) in G.edges():
                    H.add_edge(u,'new_node') 
                    H[u]['new_node']['weight'] = G[u][v]['weight'] + weight
                    weight = H[u]['new_node']['weight']
                    original.extend(G[u][v]['ori_enum'])
                    H[u]['new_node']['ori_enum'] = original   
    mapping = {u:i for i,u in enumerate(H.nodes()) }
    H = nx.relabel_nodes(H, mapping)
    return H
# Compute modulus
def modulus(G):
    """
    Uses maximum denseness to compute the modulus of G.
    """
    weight_1eta_star = np.zeros(len(G.edges()))
    for i,(u,v) in enumerate(G.edges()):
        G[u][v]['ori_enum'] = [i]
        G[u][v]['enum']  = i
    weight = [G[u][v]['weight'] for i,(u,v) in enumerate(G.edges())]
    # initialize node counter
    remain = len(G.nodes)
    H = G
    # print table header
    print("| {:^20} | {:^20} | {:^20} |".format("weight_1eta", "shrink_node", "node_remain"))
    print( ("+" + "-"*22)*3 + "+")

    # loop as long as there are still nodes to process
    while len(H.nodes()) > 1:
        # compute the 
        max_denseness, subgraph = maximum_denseness(H)
        # compute the complement of J
        crit_edges = [(u,v) for u,v in H.edges if (u,v) not in subgraph and (v,u) not in subgraph]

        # update eta^* on the critical set
        for (u,v) in crit_edges:
            for i in H[u][v]['ori_enum']:
                weight_1eta_star[i] = 1/max_denseness
        theta = 1/max_denseness
                
        # shrink graph
        H = shrink(H,subgraph)
        # update the table
        remain -= len(subgraph)-1
        print("| {:^20} | {:^20} | {:^20} | ".format(str(theta), len(subgraph), remain))

    return weight_1eta_star
def weta(G):
    """
    Uses maximum denseness to compute the modulus of G.
    """
    weight_1eta_star = np.zeros(len(G.edges()))
    for i,(u,v) in enumerate(G.edges()):
        G[u][v]['ori_enum'] = [i]
        G[u][v]['enum']  = i
    weight = [G[u][v]['weight'] for i,(u,v) in enumerate(G.edges())]
    # initialize node counter
    remain = len(G.nodes)
    H = G

    # loop as long as there are still nodes to process
    while len(H.nodes()) > 1:
        # compute the 
        max_denseness, subgraph = maximum_denseness(H)
        # compute the complement of J
        crit_edges = [(u,v) for u,v in H.edges if (u,v) not in subgraph and (v,u) not in subgraph]

        # update eta^* on the critical set
        for (u,v) in crit_edges:
            for i in H[u][v]['ori_enum']:
                weight_1eta_star[i] = 1/max_denseness
        theta = 1/max_denseness
                
        # shrink graph
        H = shrink(H,subgraph)
        # update the table
        remain -= len(subgraph)-1

    return weight_1eta_star


def etastar(G):
    we = weta(G)
    etastar = np.ones(len(G.edges()))
    for i,(u,v) in enumerate(G.edges()):
           etastar[i] = (we[i])/(G[u][v]['weight'])  
    return etastar

def MEO(G):
    eta = etastar(G)
    meo = sum([((eta[i])**2)/(G[u][v]['weight']) for i,(u,v) in enumerate(G.edges())])  
    return meo



def rhostar(G): 
    return weta(G)/MEO(G)



# Reinforce problem

def subproblem_2(G, x, e):
    """
    Uses max flow to complete a step of Cunningham's algorithm on
    edge e.
    """
    F = create_flow_graph(G)
    # set up capacities between "regular" nodes
    for u,v in G.edges:
        F[u][v]["capacity"] = (x[G[u][v]['enum']])/2

        # set up fixed weights to the target
        for v in G.nodes:
            F[v]["__target"]["capacity"] = 1

    # set up capacities from source
    for v in G.nodes:
        if v in e:
            F["__source"][v]["capacity"] = np.Inf
            continue

        F["__source"][v]["capacity"] = sum([(x[G[u][v]['enum']]) for u in G[v]])/2

    # perform the minimum cut
    cut_value, partition = nx.minimum_cut(F, "__source", "__target")

    # find epsilon from the cut value
    minimum_value = cut_value  - 1 - sum([x[G[u][v]['enum']] for (u,v) in G.edges])

    # the critical set is the partition containing the source
    if "__source" in partition[0]:
        crit_set = partition[0]
    else:
        crit_set = partition[1]
    
    # find all edges contained in the critical set
    crit_set.remove("__source")
    #crit_edges = set((u,v) for u,v in G.edges if u in crit_set and v in crit_set)

    return minimum_value
def reinforcement(G,c):
    D, subgraph = maximum_denseness(G)
    for i,(u,v) in enumerate(G.edges()):
        G[u][v]['enum'] = i
    inverse_enum =  [(u,v) for i,(u,v) in enumerate(G.edges())]
    new_weight = [G[u][v]['weight'] for i,(u,v) in enumerate(G.edges())]
    for j in c:
        x =[new_weight[i]/D for i in range(len(new_weight))]
        eps = subproblem_2(G, x, inverse_enum[j])*D
        new_weight[j] = new_weight[j] + eps
    return  new_weight