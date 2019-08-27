"""
Graphs
===============================================================================

Overview
-------------------------------------------------------------------------------

This module implements the following functions for analysis of techMiner:

* network_graph: generates the network graph for data matrix.



Functions in this module
-------------------------------------------------------------------------------

"""
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import itertools
from collections import OrderedDict 


def network_graph(matrix, save=True,name='network.png',corr_min=0.7,node_color='lightblue',
                  edge_color='lightgrey',edge_color2='lightcoral',node_size=None,fond_size=4):
    """
    This function generates network graph for matrix.

    Args:
        matrix (pandas.DataFrame): Matrix with variables on indexes and column titles
        save (boolean): If True, the graph will save with the name given
           name (str): Name to save the png file with the image
        corr_min (int): Minimum absolute value for  the relationships between variables 
                        to be shown in the graph. 
                        It is suggested when a correlation matrix is ​​being used
        node_color (str): Color name used to plot nodes
        edge_color (str): Color name used to plot edges with positive weights
        edge_color2 (str): Color name used to plot edges with negative weights
        node_size (int): If None value, the size of the nodes is plotted according
                         to the weights of edges that arrive and leave each one of them.
                         If numeric value, all nodes will be plotted with this given size
        fond_size (int): Node label fond size
    Returns:
        None
    #
    """

    plt.clf()
    #generate network graph
    graph=nx.Graph()
    # add nodes
    rows=matrix.index
    columns=matrix.columns
    nodes=list(set(rows.append(columns)))

    #add nodes
    graph.add_nodes_from(nodes)
    list_ = list(OrderedDict.fromkeys(itertools.product(rows, columns)))
    if len(rows)== len(columns) and (all(rows.sort_values())==all(columns.sort_values())):
        list_=list(set(tuple(sorted(t)) for t in list_))

    # add edges
    for i in range(len(list_)):
        combinations=list_[i]
        from_node, to_node = combinations[0],combinations[1] 
        if from_node != to_node:
            weight =matrix.loc[from_node,to_node]
            if weight != 0 and abs(weight)>corr_min:  
                if weight<0:
                    weight=abs(weight)
                    edge_colour =edge_color2
                else:
                    edge_colour = edge_color
                graph.add_edge(from_node, to_node, weight=weight, color = edge_colour)
                    
    #calculate distance between relationated nodes to avoid overlaping
    path_length = nx.shortest_path_length(graph)
    distances = pd.DataFrame(index=graph.nodes(), columns=graph.nodes())
    for row, data in path_length:
        for col, dist in data.items():
            distances.loc[row,col] = dist
    distances = distances.fillna(distances.max().max() )

    #layout of graph
    pos=nx.kamada_kawai_layout(graph, dist=distances.to_dict())

    #weights and colors of the relationships between nodes for edges thickness 
    weights=dict(((u, v), int(d["weight"])) for u, v, d in graph.edges(data=True))
    colors=dict(((u, v), d["color"]) for u, v, d in graph.edges(data=True))

    #Edges weights for plot
    if max([i for i in weights.values()]) <=1:
        width = ([(1+x)*2 for x in weights.values()])  
    else:
        width=list(weights.values())

    #node sizes
    if not node_size:
        node_sizes=dict(graph.degree())
        node_sizes= ([(x)*10 for key,x in node_sizes.items()])  
    else:
        node_sizes=node_size

    #visual graph configuration
    nx.draw(graph, pos,node_size=node_sizes, node_color=node_color, 
            edge_color=list(colors.values()), font_size=fond_size,
            with_labels=True, width=width)
                
    #save figure as png
    if save:
        plt.savefig(name, format="PNG", dpi=300, bbox_inches='tight')
    plt.show()
        
