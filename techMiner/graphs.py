"""
Graphs
===============================================================================

Overview
-------------------------------------------------------------------------------

This module implements the following functions for analysis of techMiner:

* network: generates the network graph  for data matrix.



Functions in this module
-------------------------------------------------------------------------------

"""
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

def network(matrix,node_color='lightblue',edge_color='lightgrey',node_size=80,fond_size=4):
    """
    This function generates the network graph for data matrix.

    Args:
        matrix (pandas.DataFrame): square matrix with the same indexes and column titles
        node_color (str): color name used to plot nodes
        edge_color (str): color name used to plot edges
        node_size (int): node size
        fond_size (int): node label fond size
    Returns:
        None
    #
    """

    #Select the node names ​​
    names = matrix.index
    #Select the values ​for edges
    matrix1 = matrix.values
   
    #generate networkx graph from the values of the matrix 
    plt.clf()
    graph = nx.from_numpy_matrix(matrix1,create_using=nx.Graph())
    graph = nx.relabel_nodes(graph, dict(enumerate(names)))
     
    #calculate distance between relationated nodes to avoid overlaping
    distances = pd.DataFrame(index=graph.nodes(), columns=graph.nodes())
    for row, matrix in nx.shortest_path_length(graph):
        for col, dist in matrix.items():
             distances.loc[row,col] = dist
    distances = distances.fillna(distances.max().max())
    
    #layout of graph
    pos=nx.kamada_kawai_layout(graph, dist=distances.to_dict())
    
    #weights of the relationships between nodes for edges thickness 
    weights=dict(((u, v), int(d["weight"])) for u, v, d in graph.edges(data=True))
    
    #visual graph configuration
    nx.draw(graph, pos,node_size=node_size, node_color=node_color, 
            edge_color=edge_color, font_size=fond_size,
            with_labels=True, width=list(weights.values()))
 
    #save fisgure as png
    # plt.savefig(name, format="PNG", dpi=300, bbox_inches='tight')

    plt.show()

    


