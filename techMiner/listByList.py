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
import seaborn as sns
import networkx as nx

import geopandas
import re
import itertools

from scipy.optimize import minimize
from collections import OrderedDict 
from sklearn.decomposition import PCA

from techMiner import documentsByTerm


from sklearn.cluster import KMeans

def kmeans(df, n_clusters=2):
    """Apply KMeans to a pandas dataframe.
    """

    m = KMeans(n_clusters)
    m.fit(df.values)
    centers = pd.DataFrame(
        m.cluster_centers_,
        columns = df.columns,
        index = ['Cluster ' + str(i) for i in range(n_clusters)])

    clusters = pd.DataFrame(
        {'cluster': m.predict(df.values)},
        index = df.index)

    return centers, clusters


def pca2heatmap(x, figsize=(10, 10)):

    x = x.apply(lambda x: abs(x))
    plt.figure(figsize=figsize)
    plt.pcolor(x.values, cmap='Greys')
    plt.xticks(np.arange(len(x.columns))+0.5, x.columns, rotation='vertical')
    plt.yticks(np.arange(len(x.index))+0.5, x.index)
    plt.gca().set_aspect('equal', 'box')
    plt.gca().invert_yaxis()

def pca(df, term, sep=None, n_components=2, N=10):

    x = documentsByTerm(df, term, sep=sep)
    terms = x.loc[:, term].tolist()
    if N is None or len(terms) <= N:
        N = len(terms)
    terms = sorted(terms[0:N])

    x = pd.DataFrame(
        data = np.zeros((len(df), len(terms))),
        columns = terms,
        index = df.index)

    for idx in df.index:
        w = df.loc[idx, term]
        if w is not None:
            if sep is not None:
                z = w.split(sep)
            else:
                z = [w] 
            for k in z:
                if k in terms:
                    x.loc[idx, k] = 1

    s = x.sum(axis = 1)
    x = x.loc[s > 0.0]

    pca = PCA(n_components=n_components)
    
    values = np.transpose(pca.fit(X=x.values).components_)

    result = pd.DataFrame(
        values,
        columns = ['F'+str(i) for i in range(n_components)],
        index = terms )

    return result


def correlation(df, termA, termB=None, sepA=None, sepB=None, N=20):
    """Computes autocorrelation and crosscorrelation.

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ... 'A':['a;b', 'b', 'c;a', 'b;a', 'c', 'd', 'e','a;b;c', 'e;a', None]
    ... })
    >>> df # doctest: +NORMALIZE_WHITESPACE
         A
    0    a;b
    1      b
    2    c;a
    3    b;a
    4      c
    5      d
    6      e
    7  a;b;c
    8    e;a  
    9   None

    >>> correlation(df, termA='A', sepA=';') # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
       A (row) A (col)  Autocorrelation
    0        a       a         1.000000
    1        a       b         0.670820
    2        a       c         0.516398
    3        a       d         0.000000
    4        a       e         0.316228
    5        b       a         0.670820
    6        b       b         1.000000
    7        b       c         0.288675
    8        b       d         0.000000
    9        b       e         0.000000
    10       c       a         0.516398
    11       c       b         0.288675
    12       c       c         1.000000
    13       c       d         0.000000
    14       c       e         0.000000
    15       d       a         0.000000
    16       d       b         0.000000
    17       d       c         0.000000
    18       d       d         1.000000
    19       d       e         0.000000
    20       e       a         0.316228
    21       e       b         0.000000
    22       e       c         0.000000
    23       e       d         0.000000
    24       e       e         1.000000

    >>> correlation(df, termA='A', sepA=';', N=3) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
      A (row) A (col)  Autocorrelation
    0       a       a         1.000000
    1       a       b         0.670820
    2       a       c         0.516398
    3       b       a         0.670820
    4       b       b         1.000000
    5       b       c         0.288675
    6       c       a         0.516398
    7       c       b         0.288675
    8       c       c         1.000000

    >>> df = pd.DataFrame({
    ... 'c1':['a;b',   'b', 'c;a', 'b;a', 'c',   'd', 'e', 'a;b;c', 'e;a', None,  None],
    ... 'c2':['A;B;C', 'B', 'B;D', 'B;C', 'C;B', 'A', 'A', 'B;C',    None, 'B;E', None]
    ... })
    >>> df # doctest: +NORMALIZE_WHITESPACE
           c1     c2
    0     a;b  A;B;C
    1       b      B
    2     c;a    B;D
    3     b;a    B;C
    4       c    C;B
    5       d      A
    6       e      A
    7   a;b;c    B;C
    8     e;a   None
    9    None    B;E
    10   None   None

    >>> correlation(df, termA='c1', termB='c2', sepA=';', sepB=';') # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
       c1 c2  Crosscorrelation
    0   a  A          0.258199
    1   a  B          0.676123
    2   a  C          0.670820
    3   a  D          0.447214
    4   a  E          0.000000
    5   b  A          0.288675
    6   b  B          0.755929
    7   b  C          0.750000
    8   b  D          0.000000
    9   b  E          0.000000
    10  c  A          0.000000
    11  c  B          0.654654
    12  c  C          0.577350
    13  c  D          0.577350
    14  c  E          0.000000
    15  d  A          0.577350
    16  d  B          0.000000
    17  d  C          0.000000
    18  d  D          0.000000
    19  d  E          0.000000
    20  e  A          0.408248
    21  e  B          0.000000
    22  e  C          0.000000
    23  e  D          0.000000
    24  e  E          0.000000

    >>> correlation(df, termA='c1', termB='c2', sepA=';', sepB=';', N=3) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
      c1 c2  Crosscorrelation
    0  a  A          0.258199
    1  a  B          0.676123
    2  a  C          0.670820
    3  b  A          0.288675
    4  b  B          0.755929
    5  b  C          0.750000
    6  c  A          0.000000
    7  c  B          0.654654
    8  c  C          0.577350

    """ 

    if termA == termB:
        sepB = None
        termB = None


    x = documentsByTerm(df, termA, sep=sepA)
    termsA = x.loc[:, termA].tolist()
    if N is None or len(termsA) <= N:
        termsA = sorted(termsA)
    else:
        termsA = sorted(termsA[0:N])

    if termB is not None:
        x = documentsByTerm(df, termB, sep=sepB)
        termsB = x.loc[:, termB].tolist()
        if N is None or len(termsB) <= N:
            termsB = sorted(termsB)
        else:
            termsB = sorted(termsB[0:N])
    else:
        termsB = termsA
        
    x = pd.DataFrame(
        data = np.zeros((len(df), len(termsA))),
        columns = termsA,
        index = df.index)

    for idx in df.index:
        w = df.loc[idx, termA]
        if w is not None:
            if sepA is not None:
                z = w.split(sepA)
            else:
                z = [w] 
            for k in z:
                x.loc[idx, k] = 1

    if termB is not None:

        y = pd.DataFrame(
            data = np.zeros((len(df), len(termsB))),
            columns = termsB,
            index = df.index)

        for idx in df.index:
            w = df.loc[idx, termB]
            if w is not None:
                if sepB is not None:
                    z = w.split(sepB)
                else:
                    z = [w] 
                for k in z:
                    y.loc[idx, k] = 1            
    else:
        y = x

    if termB is not None:
        col0 = termA
        col1 = termB
        col2 = 'Crosscorrelation'
    else:
        col0 = termA + ' (row)'
        col1 = termA + ' (col)'
        col2 = 'Autocorrelation'

    result =  pd.DataFrame(
        data = np.zeros((len(termsA) * len(termsB), 3)),
        columns = [col0, col1, col2]
    )

    idx = 0
    for a in termsA:
        for b in termsB:
            s1 = x.loc[:, a]
            s2 = y.loc[:, b]
            num = np.sum((s1 * s2))
            den = (np.sqrt((np.sum(s1**2))*(np.sum(s2**2))))
            if den != 0.0:
                value =  num / den
            result.loc[idx, col0] = a
            result.loc[idx, col1] = b
            result.loc[idx, col2] = value

            idx += 1

    return result       



def termByTerm(df, termA, termB, sepA=None, sepB=None, minmax=None):
    """

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ... 'A':[0, 1, 2, 3, 4, 0, 1],
    ... 'B':['a', 'b', 'c', 'd', 'e', 'a', 'b']
    ... })
    >>> df # doctest: +NORMALIZE_WHITESPACE
       A  B
    0  0  a
    1  1  b
    2  2  c
    3  3  d
    4  4  e
    5  0  a
    6  1  b    

    >>> termByTerm(df, 'A', 'B')
       A  B  Num Documents
    0  0  a              2
    1  1  b              2
    2  2  c              1
    3  3  d              1
    4  4  e              1

    >>> termByTerm(df, 'A', 'B', minmax=(2,8))
       A  B  Num Documents
    0  0  a              2
    1  1  b              2
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

    if minmax is not None:

        minval, maxval = minmax
        df = df[ df[df.columns[2]] >= minval ]
        df = df[ df[df.columns[2]] <= maxval ]

    return df



def matrix(df, ascendingA=None, ascendingB=None):
    """Displays a term by term dataframe as a matrix.

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

    for index, r in df.iterrows():
        row = r[0]
        col = r[1]
        val = r[2]
        result.loc[row, col] = val
        
    return result


def heatmap(df, ascendingA=None, ascendingB=None, figsize=(10,10), transform=True):
    """Plots a dataframe as a heatmap. 

    Arsgs:
        df (pandas.DataFrame):  Dataframe.

    Returns:
        None
    
    """
    if transform is True:
        x = matrix(df, ascendingA, ascendingB)
    else:
        x = df
    plt.figure(figsize=figsize)
    plt.pcolor(x.values, cmap='Greys')
    plt.xticks(np.arange(len(x.columns))+0.5, x.columns, rotation='vertical')
    plt.yticks(np.arange(len(x.index))+0.5, x.index)
    plt.gca().set_aspect('equal', 'box')
    plt.gca().invert_yaxis()

def alt_heatmap(df):
    return alt.Chart(df).mark_rect().encode(
        x=df.columns[0] + ':O',
        y=df.columns[1] + ':O',
        color=df.columns[2] + ':Q'
    )

def alt_circleplot(df):
    return alt.Chart(df).mark_circle().encode(
        alt.X(df.columns[0] + ':N',
        axis=alt.Axis(labelAngle=270)),
        alt.Y(df.columns[1] + ':N'),
        size=df.columns[2],
        color=df.columns[2])


def sns_heatmap(df):
    return sns.heatmap(df)




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







# def termByTerm0(df, termA, termB, sepA=None, sepB=None, 
#                ascendingA=None, ascendingB=None, minmax=None):
    
#     df = df[[termA, termB]].dropna()
    
#     ##
#     ## Expande las dos columnas de los datos originales
#     ##
#     if sepA is None and sepB is None:
#         df = df[[termA, termB]]
    
#     if sepA is not None and sepB is None:
        
#         t = [(x, y) for x, y in zip(df[termA], df[termB])]
#         t = [(c, b) for a, b in t for c in a.split(sepA)]
#         df = pd.DataFrame({
#             termA: [a for a,b in t],
#             termB: [b for a,b in t]
#         })
        
#     if sepA is None and sepB is not None:
    
#         t = [(x, y) for x, y in zip(df[termA], df[termB])]
#         t = [(a, c) for a, b in t for c in b.split(sepB)]
#         df = pd.DataFrame({
#             termA: [a for a,b in t],
#             termB: [b for a,b in t]
#         })

#     if sepA is not None and sepB is not None:
    
#         t = [(x, y) for x, y in zip(df[termA], df[termB])]
#         t = [(c, b) for a, b in t for c in a.split(sepA)]
#         t = [(a, c) for a, b in t for c in b.split(sepB)]
#         df = pd.DataFrame({
#             termA: [a for a,b in t],
#             termB: [b for a,b in t]
#         })

#     if termA == 'Year':
#         termA_unique = range(min(df.Year), max(df.Year)+1)
#     else:
#         termA_unique = df[termA].unique()
        
#     if termB == 'Year':
#         termB_unique = range(min(df.Year), max(df.Year)+1)
#     else:
#         termB_unique = df[termB].unique()
        
#     if ascendingA is not None:
#         termA_unique = sorted(termA_unique, reverse = not ascendingA)

#     if ascendingB is not None:
#         termB_unique = sorted(termB_unique, reverse = not ascendingB)

#     result = pd.DataFrame(
#         np.zeros((len(termA_unique), len(termB_unique)))
#     )
    
#     result.columns = termB_unique
#     result.index = termA_unique

#     for  a,b in zip(df[termA], df[termB]):
#         result.loc[a, b] += 1
    
#     if minmax is not None:

#         minval, maxval = minmax
#         r = result.copy()
       
#         for a in termA_unique:
#             for b in termB_unique:
#                 if r.loc[a, b] < minval or r.loc[a, b] > maxval:
#                     r.loc[a, b] = np.nan
        
#         r = r.dropna(axis='index', how='all')
#         r = r.dropna(axis='columns', how='all')
#         result = result[r.columns]
#         result = result.loc[r.index,:]
    
#     return result





