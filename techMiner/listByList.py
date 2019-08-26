"""
Functions for analysis 
===============================================================================

Overview
-------------------------------------------------------------------------------

The functions in this module allows the user to transform fields in a dataframe.



Functions in this module
-------------------------------------------------------------------------------

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import networkx as nx

import geopandas
import re
import itertools

from scipy.optimize import minimize
from collections import OrderedDict 



def termByTerm(df, termA, termB, sepA=None, sepB=None, minmax=None):
    """

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ... 'a':[0, 1, 2, 3, 4, 0, 1],
    ... 'b':['a', 'b', 'c', 'd', 'e', 'a', 'b']
    ... })
    >>> df # doctest: +NORMALIZE_WHITESPACE
       a  b
    0  0  a
    1  1  b
    2  2  c
    3  3  d
    4  4  e
    5  0  a
    6  1  b    

    >>> termByTerm(df, 'a', 'b')
       a  b  Num Documents
    0  0  a              2
    1  1  b              2
    2  2  c              1
    3  3  d              1
    4  4  e              1

    """
    
    df = df[[termA, termB]].dropna()


    ##
    ## Expande las dos columnas de los datos originales
    ##
    if sepA is None and sepB is None:
        df = df[[termA, termB]]
    
    if sepA is not None and sepB is None:
        
        t = [(x, y) for x, y in zip(df[termA], df[termB])]
        t = [(c, b) for a, b in t for c in a.split(sepA)]
        df = pd.DataFrame({
            termA: [a for a,b in t],
            termB: [b for a,b in t]
        })
        
    if sepA is None and sepB is not None:
    
        t = [(x, y) for x, y in zip(df[termA], df[termB])]
        t = [(a, c) for a, b in t for c in b.split(sepB)]
        df = pd.DataFrame({
            termA: [a for a,b in t],
            termB: [b for a,b in t]
        })

    if sepA is not None and sepB is not None:
    
        t = [(x, y) for x, y in zip(df[termA], df[termB])]
        t = [(c, b) for a, b in t for c in a.split(sepA)]
        t = [(a, c) for a, b in t for c in b.split(sepB)]
        df = pd.DataFrame({
            termA: [a for a,b in t],
            termB: [b for a,b in t]
        })

    x = df.groupby(by=[termA, termB]).size()
    a = [t for t,_ in x.index]
    b = [t for _,t in x.index]
    df = pd.DataFrame({
        termA: a,
        termB: b,
        'Num Documents': x.tolist()
    })

    return df




def matrix(df, ascendingA=None, ascendingB=None, minmax=None):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ... 'a':[0, 1, 2, 3, 4, 0, 1],
    ... 'b':['a', 'b', 'c', 'd', 'e', 'a', 'b']
    ... })
    >>> df # doctest: +NORMALIZE_WHITESPACE
       a  b
    0  0  a
    1  1  b
    2  2  c
    3  3  d
    4  4  e
    5  0  a
    6  1  b    

    >>> x = termByTerm(df, 'a', 'b')
    >>> x # doctest: +NORMALIZE_WHITESPACE
       a  b  Num Documents
    0  0  a              2
    1  1  b              2
    2  2  c              1
    3  3  d              1
    4  4  e              1
    >>> matrix(x)
         a    b    c    d    e
    0  2.0  0.0  0.0  0.0  0.0
    1  0.0  2.0  0.0  0.0  0.0
    2  0.0  0.0  1.0  0.0  0.0
    3  0.0  0.0  0.0  1.0  0.0
    4  0.0  0.0  0.0  0.0  1.0

    >>> matrix(x, minmax=(2, 8))
         a    b
    0  2.0  0.0
    1  0.0  2.0
    """

    if df.columns[0] == 'Year':
        termA_unique = range(min(df.Year), max(df.Year)+1)
    else:
        termA_unique = df.iloc[:,0].unique()
        
    if df.columns[1] == 'Year':
        termB_unique = range(min(df.Year), max(df.Year)+1)
    else:
        termB_unique = df.iloc[:,1].unique()
        
    if ascendingA is not None:
        termA_unique = sorted(termA_unique, reverse = not ascendingA)

    if ascendingB is not None:
        termB_unique = sorted(termB_unique, reverse = not ascendingB)


    result = pd.DataFrame(
        np.zeros((len(termA_unique), len(termB_unique)))
    )
    
    result.columns = termB_unique
    result.index = termA_unique

    for  idx in df.index:
        row = df.iloc[idx, 0]
        col = df.iloc[idx, 1]
        val = df.iloc[idx, 2]
        result.loc[row, col] = val
    
    if minmax is not None:

        minval, maxval = minmax
        r = result.copy()
       
        for a in termA_unique:
            for b in termB_unique:
                if r.loc[a, b] < minval or r.loc[a, b] > maxval:
                    r.loc[a, b] = np.nan
        
        r = r.dropna(axis='index', how='all').dropna(axis='columns', how='all')
        result = result[r.columns]
        result = result.loc[r.index,:]
    
    return result




# def stack(x, xlabel, ylabel=None, vlabel=None):
#     """
#     >>> import pandas as pd
#     >>> x = pd.DataFrame({
#     ...    'A': [0, 1, 2],
#     ...    'B': [3, 4, 5],
#     ...    'C': [6, 7, 8]},
#     ...    index=['a', 'b', 'c'])
#     >>> x
#        A  B  C
#     a  0  3  6
#     b  1  4  7
#     c  2  5  8

#     >>> stack(x, xlabel='X', ylabel='Y', vlabel='Z')
#        Z  X  Y
#     0  0  a  A
#     1  3  a  B
#     2  6  a  C
#     3  1  b  A
#     4  4  b  B
#     5  7  b  C
#     6  2  c  A
#     7  5  c  B
#     8  8  c  C

#     >>> x = pd.DataFrame({
#     ...    'A': [0, 1, 2]},
#     ...    index=['a', 'b', 'c'])

#     # >>> stack(x, xlabel='X')
#     #    A  X
#     # 0  0  a
#     # 1  1  b
#     # 2  2  c

#     # >>> stack(x.A, xlabel='X')
#     #       A  X
#     #    0  0  0
#     #    1  1  1
#     #    2  2  2

#     """
#     if isinstance(x, pd.Series):
#         x = pd.DataFrame(x)
#         x.columns = [xlabel]

#     if isinstance(x, pd.DataFrame):
#         if len(x.columns) == 1:
#             x[x.index.name] = x.index
#             x.index = range(len(x.index))
#             return x
#         else:
#             z = x.stack()
#             xindex = [i for i,_ in z.index]
#             yindex = [i for _,i in z.index]
#             z = pd.DataFrame({vlabel: z.tolist()})
#             z[xlabel] = xindex
#             z[ylabel] = yindex
#             return z







def termByTerm0(df, termA, termB, sepA=None, sepB=None, 
               ascendingA=None, ascendingB=None, minmax=None):
    
    df = df[[termA, termB]].dropna()
    
    ##
    ## Expande las dos columnas de los datos originales
    ##
    if sepA is None and sepB is None:
        df = df[[termA, termB]]
    
    if sepA is not None and sepB is None:
        
        t = [(x, y) for x, y in zip(df[termA], df[termB])]
        t = [(c, b) for a, b in t for c in a.split(sepA)]
        df = pd.DataFrame({
            termA: [a for a,b in t],
            termB: [b for a,b in t]
        })
        
    if sepA is None and sepB is not None:
    
        t = [(x, y) for x, y in zip(df[termA], df[termB])]
        t = [(a, c) for a, b in t for c in b.split(sepB)]
        df = pd.DataFrame({
            termA: [a for a,b in t],
            termB: [b for a,b in t]
        })

    if sepA is not None and sepB is not None:
    
        t = [(x, y) for x, y in zip(df[termA], df[termB])]
        t = [(c, b) for a, b in t for c in a.split(sepA)]
        t = [(a, c) for a, b in t for c in b.split(sepB)]
        df = pd.DataFrame({
            termA: [a for a,b in t],
            termB: [b for a,b in t]
        })

    if termA == 'Year':
        termA_unique = range(min(df.Year), max(df.Year)+1)
    else:
        termA_unique = df[termA].unique()
        
    if termB == 'Year':
        termB_unique = range(min(df.Year), max(df.Year)+1)
    else:
        termB_unique = df[termB].unique()
        
    if ascendingA is not None:
        termA_unique = sorted(termA_unique, reverse = not ascendingA)

    if ascendingB is not None:
        termB_unique = sorted(termB_unique, reverse = not ascendingB)

    result = pd.DataFrame(
        np.zeros((len(termA_unique), len(termB_unique)))
    )
    
    result.columns = termB_unique
    result.index = termA_unique

    for  a,b in zip(df[termA], df[termB]):
        result.loc[a, b] += 1
    
    if minmax is not None:

        minval, maxval = minmax
        r = result.copy()
       
        for a in termA_unique:
            for b in termB_unique:
                if r.loc[a, b] < minval or r.loc[a, b] > maxval:
                    r.loc[a, b] = np.nan
        
        r = r.dropna(axis='index', how='all')
        r = r.dropna(axis='columns', how='all')
        result = result[r.columns]
        result = result.loc[r.index,:]
    
    return result



def heatmap(x, figsize=(10,10)):
    """Plots a dataframe as a heatmap. 

    Arsgs:
        x (pandas.DataFrame):  Matrix of values to plot

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    plt.pcolor(x.values, cmap='Greys')
    plt.xticks(np.arange(len(x.columns))+0.5, x.columns, rotation='vertical')
    plt.yticks(np.arange(len(x.index))+0.5, x.index)
    plt.gca().set_aspect('equal', 'box')
    plt.gca().invert_yaxis()

def alt_heatmap(x):
    # Convert this grid to columnar data expected by Altair
    source = pd.DataFrame({'x': x.ravel(),
                        'y': y.ravel(),
                        'z': z.ravel()})

    alt.Chart(source).mark_rect().encode(
        x='x:O',
        y='y:O',
        color='z:Q'
    )



# def circle_chat(x, xlabel, ylabel, values):

#     alt.Chart(x).mark_circle().encode(
#         alt.X(xlabel + 'N',
#             axis=alt.Axis(labelAngle=270)),
#         alt.Y(ylabel + ':N'),
#         size='Cited by',
#         color='Cited by')


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




