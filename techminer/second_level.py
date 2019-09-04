"""
techMiner.SecondLevelResult
==================================================================================================



"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from collections import OrderedDict 
import itertools

class SecondLevelResult(pd.DataFrame):
    """Class to represent a matrix of analysis of bibliographic records.
    """
    #---------------------------------------------------------------------------------------------
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, isfactor=False,
        transform=True):
        super().__init__(data, index, columns, dtype, copy)
        self._isfactor = isfactor
        self._transform = transform
        
    #---------------------------------------------------------------------------------------------
    @property
    def _constructor_expanddim(self):
        self._constructor_expanddim()

    #---------------------------------------------------------------------------------------------
    def transpose(self, *args, **kwargs):
        result = SecondLevelResult(super().transpose())
        result._isfactor = self._isfactor
        result._transform = self._transform
        return result

    #---------------------------------------------------------------------------------------------
    def to_matrix(self, ascendingA=None, ascendingB=None):
        """Displays a term by term dataframe as a matrix.

        >>> mtx = SecondLevelResult({
        ...   'rows':['r0', 'r1', 'r2', 'r0', 'r1', 'r2'],
        ...   'cols':['c0', 'c1', 'c0', 'c1', 'c0', 'c1'],
        ...   'vals':[ 1.0,  2.0,  3.0,  4.0,  5.0,  6.0]
        ... })
        >>> mtx
          rows cols  vals
        0   r0   c0   1.0
        1   r1   c1   2.0
        2   r2   c0   3.0
        3   r0   c1   4.0
        4   r1   c0   5.0
        5   r2   c1   6.0

        >>> mtx.to_matrix() # doctest: +NORMALIZE_WHITESPACE
             c0   c1
        r0  1.0  4.0
        r1  5.0  2.0
        r2  3.0  6.0    

        """
        if self._transform is False:
            return pd.DataFrame(self)

        if self.columns[0] == 'Year':
            termA_unique = range(min(self.Year), max(self.Year)+1)
        else:
            termA_unique = self.iloc[:,0].unique()
            
        if self.columns[1] == 'Year':
            termB_unique = range(min(self.Year), max(self.Year)+1)
        else:
            termB_unique = self.iloc[:,1].unique()
            
        if ascendingA is not None:
            termA_unique = sorted(termA_unique, reverse = not ascendingA)

        if ascendingB is not None:
            termB_unique = sorted(termB_unique, reverse = not ascendingB)

        result = pd.DataFrame(
            np.zeros((len(termA_unique), len(termB_unique)))
        )
        
        result.columns = termB_unique
        result.index = termA_unique

        for index, r in self.iterrows():
            row = r[0]
            col = r[1]
            val = r[2]
            result.loc[row, col] = val
            
        return SecondLevelResult(result, transform = False)


    #---------------------------------------------------------------------------------------------
    def heatmap(self, ascendingA=None, ascendingB=None, figsize=(10, 10)):
        
        if self._isfactor is True:
            x = self.apply(lambda x: abs(x))
        else:
            if self._transform is True:
                x = self.to_matrix(ascendingA, ascendingB)
            else:
                x = self

        plt.figure(figsize=figsize)
        plt.pcolor(np.transpose(x.values), cmap='Greys')
        plt.xticks(np.arange(len(x.index))+0.5, x.index, rotation='vertical')
        plt.yticks(np.arange(len(x.columns))+0.5, x.columns)
        plt.gca().set_aspect('equal', 'box')
        plt.gca().invert_yaxis()
    
    #---------------------------------------------------------------------------------------------
    def heatmap_in_altair(self, ascendingA=None, ascendingB=None):

        if ascendingA is None or ascendingA is True:
            sort_X = 'ascending'
        else:
            sort_X = 'descending'

        if ascendingB is None or ascendingB is True:
            sort_Y = 'ascending'
        else:
            sort_Y = 'descending'

        return alt.Chart(self).mark_rect().encode(
            alt.X(self.columns[0] + ':O', sort=sort_X),
            alt.Y(self.columns[1] + ':O', sort=sort_Y),
            color=self.columns[2] + ':Q')

    #---------------------------------------------------------------------------------------------
    def heatmap_in_seaborn(self):
        return sns.heatmap(self)

    #---------------------------------------------------------------------------------------------
    def circleplot_in_altair(self, ascendingA=None, ascendingB=None):

        if ascendingA is None or ascendingA is True:
            sort_X = 'ascending'
        else:
            sort_X = 'descending'

        if ascendingB is None or ascendingB is True:
            sort_Y = 'ascending'
        else:
            sort_Y = 'descending'

        return alt.Chart(self).mark_circle().encode(
            alt.X(self.columns[0] + ':N',
                axis=alt.Axis(labelAngle=270), 
                sort=sort_X),
            alt.Y(self.columns[1] + ':N',
                sort=sort_Y),
            size=self.columns[2],
            color=self.columns[2])

    #---------------------------------------------------------------------------------------------
    def kmeans(self, n_clusters=2):
        """Apply KMeans to a pandas dataframe.
        """

        if self._isfactor is True:
            x = self.copy()
        else:
            x = self.to_matrix()

        m = KMeans(n_clusters)
        m.fit(x.values)
        # centers = pd.DataFrame(
        #     m.cluster_centers_,
        #     columns = x.columns,
        #     index = ['Cluster ' + str(i) for i in range(n_clusters)])



        centers = SecondLevelResult(
            np.transpose(m.cluster_centers_),
            columns = ['Cluster ' + str(i) for i in range(n_clusters)],
            index = x.columns,
            transform = False)



        clusters = pd.DataFrame(
            {'cluster': m.predict(x.values)},
            index = x.index)

        return centers, clusters

    #---------------------------------------------------------------------------------------------
    def network_graph(self, save=True, name='network.png', corr_min=0.7, node_color='lightblue',
                  edge_color='lightgrey', edge_color2='lightcoral', node_size=None, fond_size=4):
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
        
        """

        plt.clf()
        #generate network graph
        graph=nx.Graph()
        # add nodes
        rows = self.index
        columns = self.columns
        nodes = list(set(rows.append(columns)))

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
                weight =self.loc[from_node,to_node]
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
        pos = nx.kamada_kawai_layout(graph, dist=distances.to_dict())

        #weights and colors of the relationships between nodes for edges thickness 
        weights = dict(((u, v), int(d["weight"])) for u, v, d in graph.edges(data=True))
        colors = dict(((u, v), d["color"]) for u, v, d in graph.edges(data=True))

        #Edges weights for plot
        if max([i for i in weights.values()]) <=1:
            width = ([(1+x)*2 for x in weights.values()])  
        else:
            width = list(weights.values())

        #node sizes
        if not node_size:
            node_sizes = dict(graph.degree())
            node_sizes = ([(x)*10 for key,x in node_sizes.items()])  
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

