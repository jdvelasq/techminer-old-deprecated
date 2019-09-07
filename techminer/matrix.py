"""
TechMiner.Result
==================================================================================================

"""
import altair as alt
import geopandas
import geoplot
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from collections import OrderedDict 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
from shapely.geometry import Point, LineString
from sklearn.cluster import KMeans


class Matrix(pd.DataFrame):
    """Class implementing a dataframe with results of analysis.
    """
    #---------------------------------------------------------------------------------------------
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, rtype=None):
        super().__init__(data, index, columns, dtype, copy)
        self._rtype = rtype

    #----------------------------------------------------------------------------------------------
    @property
    def _constructor_expanddim(self):
        return self

    #----------------------------------------------------------------------------------------------
    def barhplot_in_altair(self):
        """Plots a pandas.DataFrame using Altair.
        """
        if self._rtype not in [
            'num-docs-by-terms',
            'num_docs-by-year'
            'num-citations-by-terms',
            'num-citations-by-year']:

            raise Exception('Invalid function call for type: ' + self._rtype )

        columns = self.columns.tolist()
        return alt.Chart(self).mark_bar().encode(
            alt.Y(columns[0] + ':N', sort=alt.EncodingSortField(field=columns[1] + ':Q')),
            alt.X(columns[1] + ':Q'),
            alt.Color(columns[1] + ':Q', scale=alt.Scale(scheme='greys'))
        )

    #----------------------------------------------------------------------------------------------
    def barplot_in_altair(self):
        """Vertical bar plot in Altair.
        """

        if self._rtype not in [
            'num-docs-by-terms',
            'num_docs-by-year'
            'num-citations-by-terms',
            'num-citations-by-year']:

            raise Exception('Invalid function call for type: ' + self._rtype )

        columns = self.columns.tolist()
        return alt.Chart(self).mark_bar().encode(
            alt.X(columns[0] + ':N', sort=alt.EncodingSortField(field=columns[1] + ':Q')),
            alt.Y(columns[1] + ':Q'),
            alt.Color(columns[1] + ':Q', scale=alt.Scale(scheme='greys'))
        )

    #----------------------------------------------------------------------------------------------
    def barhplot_in_seaborn(self):
        """Horizontal bar plot using Seaborn.
        """

        if self._rtype not in [
            'num-docs-by-terms',
            'num_docs-by-year'
            'num-citations-by-terms',
            'num-citations-by-year']:

            raise Exception('Invalid function call for type: ' + self._rtype )

        columns = self.columns.tolist()
        return sns.barplot(
            x="Num Documents",
            y=columns[0],
            data=self,
            label=columns[0],
            color="gray"
        )

    #----------------------------------------------------------------------------------------------
    def barplot_in_seaborn(self):
        """Vertical bar plot using Seaborn.
        """

        if self._rtype not in [
            'num-docs-by-terms',
            'num_docs-by-year'
            'num-citations-by-terms',
            'num-citations-by-year']:

            raise Exception('Invalid function call for type: ' + self._rtype )

        columns = self.columns.tolist()
        result = sns.barplot(
            y="Num Documents",
            x=columns[0],
            data=self,
            label=columns[0],
            color="gray")
        _, labels = plt.xticks()
        result.set_xticklabels(labels, rotation=90)
        return result


    #---------------------------------------------------------------------------------------------
    def circleplot_in_altair(self, ascendingA=None, ascendingB=None):

        if self._rtype not in [
            'co_ocurrence-matrix',
            'cross-matrix'
            'auto-matrix',
            'factor-matrix']:

            raise Exception('Invalid function call for type: ' + self._rtype )

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
    def heatmap(self, ascendingA=None, ascendingB=None, figsize=(10, 10)):
        
        if self._rtype not in [
            'co_ocurrence-matrix',
            'cross-matrix'
            'auto-matrix',
            'factor-matrix']:

            raise Exception('Invalid function call for type: ' + self._rtype )

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

        if self._rtype not in [
            'co_ocurrence-matrix',
            'cross-matrix'
            'auto-matrix']:

            raise Exception('Invalid function call for type: ' + self._rtype )


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
    def kmeans(self, n_clusters=2):
        """Apply KMeans to a pandas dataframe.
        """

        if self._rtype not in [
            'co_ocurrence-matrix',
            'cross-matrix'
            'auto-matrix']:

            raise Exception('Invalid function call for type: ' + self._rtype )


        if self._rtype == 'factor-matrix' is True:
            x = self.copy()
        else:
            x = self.to_matrix()

        m = KMeans(n_clusters)
        m.fit(x.values)
        # centers = pd.DataFrame(
        #     m.cluster_centers_,
        #     columns = x.columns,
        #     index = ['Cluster ' + str(i) for i in range(n_clusters)])



        centers = SecondLevelMatrix(
            np.transpose(m.cluster_centers_),
            columns = ['Cluster ' + str(i) for i in range(n_clusters)],
            index = x.columns,
            transform = False)



        clusters = pd.DataFrame(
            {'cluster': m.predict(x.values)},
            index = x.index)

        return centers, clusters



    #---------------------------------------------------------------------------------------------
    #TODO personalizar valor superior para escalar los pesos de los puentes
    #TODO map
    def network(self, save=False, name='network.png', corr_min=0.7, node_color='lightblue',
                  edge_color='lightgrey', edge_color2='lightcoral', node_size=None, fond_size=4,
                  figsize = (10,10)):
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
            figsize (float, float): size of figure drawn

        Returns:
            None
        
        """

        if self._rtype not in [
            'co_ocurrence-matrix',
            'cross-matrix'
            'auto-matrix',
            'factor-matrix']:

            raise Exception('Invalid function call for type: ' + self._rtype )


        if self._rtype == 'factor-matrix':
            x = self.copy()
        else:
            x = self.to_matrix()

        plt.clf()
        plt.figure(figsize=figsize)
        
        #generate network graph
        graph = nx.Graph()
        # add nodes
        rows = x.index
        columns = x.columns
        nodes = list(set(rows.append(columns)))

        #add nodes
        graph.add_nodes_from(nodes)
        list_ = list(OrderedDict.fromkeys(itertools.product(rows, columns)))
        if len(rows) == len(columns) and (all(rows.sort_values())==all(columns.sort_values())):
            list_ = list(set(tuple(sorted(t)) for t in list_))

        # add edges
        for i in range(len(list_)):
            combinations=list_[i]
            from_node, to_node = combinations[0], combinations[1] 
            if from_node != to_node:
                weight = x.loc[from_node, to_node]
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
        max_=max([i for i in weights.values()])
        min_=min([i for i in weights.values()])
        min_range=1
        max_range=5
        if max_<=1:
            width = ([(1+x)*2 for x in weights.values()])  
        else:
            width = ([((((x-min_)/(max_-min_))*(max_range-min_range))+min_range) for x in weights.values()]) 
            # width=list(weights.values())
    
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
        return None


    #---------------------------------------------------------------------------------------------
   #TODO networkmap validar como pasar lonlat,
    #que pasa si valores negativos???
    #personalizar tamaño de la figura, 
    #guardar archivo 
    #quitar ejes

    def networkmap(matrix, color_edges ='grey', color_node='red',color_map = 'white', edge_map = 'lightgrey', node_size =None, edge_weight = None):

        """
        This function generates network graph over map, for matrix with country relations.

        Args:
            matrix (pandas.DataFrame): Matrix with variables on indexes and column titles
            color_edges (str): Color name used to plot edges
            color_node (str): Color name used to plot nodes
            color_map (str): Color name used to plot map countries
            edge_map (str): Color name used to plot contries border
            node_size (int): If None value, the size of the nodes is plotted according
                            to the weights of edges that arrive and leave each one of them.
                            If numeric value, all nodes will be plotted with this given size
            edge_weight (int): If None value, the weigth of the edges is plotted according
                            to matrix values
                            If numeric value, all edges will be plotted with this given size
        Returns:
            None
        #
        """

        #Get longitudes and latituds
        lonlat=pd.read_csv('LonLat.csv',sep=';')

        #node's names
        rows=matrix.index
        columns=matrix.columns
        nodes=list(set(rows.append(columns)))
        nodes = [row.replace(' ', '') for row in rows ]


        #nodes_combinations
        list_ = list(OrderedDict.fromkeys(itertools.product(rows, columns)))
        if len(rows)== len(columns) and (all(rows.sort_values())==all(columns.sort_values())):
            list_=list(set(tuple(sorted(t)) for t in list_))
        

        pos=lonlat[lonlat.country.isin(nodes)]

        geometry = [Point(xy) for xy in zip(pos['lon'], pos['lat'])]

        # Coordinate reference system : WGS84
        crs = {'init': 'epsg:4326'}

        # Creating a Geographic data frame from nodes
        gdf = geopandas.GeoDataFrame(pos, crs=crs, geometry=geometry)

        #edges
        df=pd.DataFrame({'initial':[],'final':[],'initial_lon': [], 'initial_lat': [],'final_lon': [],'final_lat': [], 'weight': []})
        for i in range(len(list_)):
            combinations=list_[i]
            from_node, to_node = combinations[0],combinations[1] 
            if from_node != to_node:
                weight =matrix.loc[from_node,to_node]
                if weight != 0: 
                    df = df.append({'initial':from_node.replace(' ', ''),'final':to_node.replace(' ', ''),'initial_lon': pos[pos.country==from_node.replace(' ', '')]['lon'].values, 'initial_lat': pos[pos.country==from_node.replace(' ', '')]['lat'].values,'final_lon': pos[pos.country==to_node.replace(' ', '')]['lon'].values,'final_lat': pos[pos.country==to_node.replace(' ', '')]['lat'].values, 'weight': weight}, ignore_index='True')

        # Creating a Geographic data frame from edges  
        df['orig_coord'] = [Point(xy) for xy in zip(df['initial_lon'], df['initial_lat'])]
        df['dest_coord'] = [Point(xy) for xy in zip(df['final_lon'], df['final_lat'])]

        geometry_lines=[LineString(xy) for xy in zip(df.orig_coord,df.dest_coord)]
        gdf_lines=geopandas.GeoDataFrame(df, crs=crs, geometry=geometry_lines)

        #base map
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

        #nodes size
        if not node_size:
            nodes_freq=list(gdf_lines.initial) + list(gdf_lines.final)
            nodes_freq.sort()
            nodes_size= {x:nodes_freq.count(x) for x in nodes_freq}
            size=nodes_size.values()
            size=[x*5 for x in size]
        else:
            size=node_size
    
        #edges weigth
        if not node_size:
            edges_=list(gdf_lines.weight)
        else:
            edges_= node_size
        #plot graph
        gdf.plot(ax=world.plot(ax=gdf_lines.plot(color=color_edges, markersize=edges_,alpha=0.5),color=color_map, edgecolor= edge_map), color=color_node,  markersize=size)
        
        plt.show()

        return None




    #---------------------------------------------------------------------------------------------
    def to_matrix(self, ascendingA=None, ascendingB=None):
        """Displays a term by term dataframe as a matrix.

        >>> mtx = Results({
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

        if self._rtype not in [
            'co_ocurrence-matrix',
            'cross-matrix'
            'auto-matrix']:

            raise Exception('Invalid function call for type: ' + self._rtype )


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
            
        return Matrix(result, rtype=False)

    #---------------------------------------------------------------------------------------------
    def transpose(self, *args, **kwargs):
        result = Matrix(super().transpose())
        result._rtype = self._rtype
        return result    

    #----------------------------------------------------------------------------------------------
    def worldmap(self, figsize=(14, 7)):
        """Worldmap plot with the number of documents per country.
        """

        if 'Country' not in list(self.columns):
            raise Exception('No country column found in data')


        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        world = world[world.name != "Antarctica"]
        world['q'] = 0
        world.index = world.name

        rdf = self.copy()
        rdf['Country'] = rdf['Country'].map(
            lambda x: x.replace('UnitedStates', 'United States of America')
        )

        #rdf['Country'] = [w if w !=  else  for w in rdf['Country']]
        rdf.index = rdf['Country']
        for country in rdf['Country']:
            if country in world.index:
                world.at[country, 'q'] = rdf.loc[country, 'Num Documents']
        _, axx = plt.subplots(1, 1, figsize=figsize)
        divider = make_axes_locatable(axx)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        world.plot(column='q', legend=True, ax=axx, cax=cax, cmap='Pastel2')

    #----------------------------------------------------------------------------------------------


    








