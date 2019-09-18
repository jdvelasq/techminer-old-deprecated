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


#---------------------------------------------------------------------------------------------
def chord_diagram(labels, edges, figsize=(12, 12), minval=None, R=3, n_bezier=100, dist=0.2):
    
    def bezier(p0, p1, p2, linewidth, linestyle, n_bezier=100):

        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2
        
        xb = [(1 - t)**2 * x0 + 2 * t * (1-t)*x1 + t**2 * x2 for t in np.linspace(0.0, 1.0, n_bezier)]
        yb = [(1 - t)**2 * y0 + 2 * t * (1-t)*y1 + t**2 * y2 for t in np.linspace(0.0, 1.0, n_bezier)]
    
        plt.plot(xb, yb, color='black', linewidth=linewidth, linestyle=linestyle)
    
    #
    # rutina ppal
    #

    plt.figure(figsize=figsize)
    n_nodes = len(labels)
    
    theta = np.linspace(0.0, 2 * np.pi, n_nodes, endpoint=False)
    points_x = [R * np.cos(t) for t in theta]
    points_y = [R * np.sin(t) for t in theta]
    
    # dibuja los puntos sobre la circunferencia
    plt.scatter(points_x, points_y, s=80, color='black')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.gca().set_aspect('equal', 'box')
    
    # arcos de las relaciones    
    data = {label:(points_x[idx], points_y[idx], theta[idx]) for idx, label in enumerate(labels)}
    
    ## labels
    lbl_x = [(R+dist) * np.cos(t) for t in theta]
    lbl_y = [(R+dist) * np.sin(t) for t in theta]
    lbl_theta = [t / (2 * np.pi) * 360 for t in theta]
    lbl_theta = [t - 180 if t > 180 else t for t in lbl_theta]
    lbl_theta = [t - 180 if t > 90 else t for t in lbl_theta]
        
    for txt, xt, yt, angletxt, angle  in zip(labels, lbl_x, lbl_y, lbl_theta, theta):
            
        if xt >= 0:
            ha = 'left'
        else:
            ha = 'right'
    
        plt.text(
            xt, 
            yt, 
            txt, 
            fontsize=10,
            rotation=angletxt,
            va = 'center',
            ha = ha, # 'center'
            rotation_mode = 'anchor',
            backgroundcolor='white')
                
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    for txt in ['bottom', 'top', 'left', 'right']:
        plt.gca().spines[txt].set_color('white')

    for index, r in edges.iterrows():

        row = r[0]
        col = r[1]
        linewidth = r[2]
        linestyle = r[3]

        if row != col:

            x0, y0, a0 = data[row]
            x2, y2, a2 = data[col]            
            
            angle = a0 + (a2 - a0) / 2
            
            if angle > np.pi:
                angle_corr = angle - np.pi
            else:
                angle_corr = angle
                
            distance = np.abs(a2 - a0)
            if distance > np.pi:
                distance = distance - np.pi
            distance = (1.0 - 1.0 * distance / np.pi) * R / 2.5
            x1 = distance * np.cos(angle)
            y1 = distance * np.sin(angle)

            ## dibuja los arcos            
            bezier( [x0, y0], [x1, y1], [x2, y2], linewidth=linewidth, linestyle=linestyle)
                 

#---------------------------------------------------------------------------------------------
class Matrix(pd.DataFrame):
    """Class implementing a dataframe with results of analysis.
    """
    #---------------------------------------------------------------------------------------------
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, rtype=None, 
            cluster_data=None):
        
        super().__init__(data, index, columns, dtype, copy)
        self._rtype = rtype
        self._cluster_data = None
        self._cluster_data = cluster_data

    #----------------------------------------------------------------------------------------------
    @property
    def _constructor_expanddim(self):
        return self

    #----------------------------------------------------------------------------------------------
    def print_IDs(self):

        if self._rtype in ['coo-matrix', 'cross-matrix', 'auto-matrix']:

            for idx, row in self.iterrows():
                if row[-1] is not None:
                    print(row[0], ', ', row[1], ' (', len(row[-1]), ')', ' : ', sep='', end='')
                    for i in row[-1]:
                        print(i, sep='', end='')
                    print()

        elif self._rtype == 'coo-matrix-year':

            for idx, row in self.iterrows():
                if row[-1] is not None:
                    print(row[0], ', ', row[1], ', ', row[2], ' (', len(row[-1]), ')', ' : ', sep='', end='')
                    for i in row[-1]:
                        print(i, sep='', end='')
                    print()

        elif self._rtype == 'factor-matrix':
            pass
        else:
            pass

    #---------------------------------------------------------------------------------------------
    def chord_diagram(self, figsize=(12, 12), minval=None, R=3):

        
        if self._rtype not in ['auto-matrix']:
            raise Exception('Invalid function call for type: ' + self._rtype )
            
        x = self
        labels = list(set(x[x.columns[0]]))

        from_list = []
        to_list = []
        linewidth_list = []
        linestyle_list = []

        for index, r in x.iterrows():

            row = r[0]
            col = r[1]
            val = r[2]

            if row != col:
                
                if val >= 0.75:
                    linewidth = 4
                    linestyle = '-' 
                elif val >= 0.50:
                    linewidth = 2
                    linestyle = '-' 
                elif val >= 0.25:
                    linewidth = 2
                    linestyle = '--' 
                elif val < 0.25:
                    linewidth = 1
                    linestyle = ':'
                else: 
                    linewidth = 0
                    linestyle = '-'

                if minval is None or abs(val) >= minval:
                    from_list.append(row)
                    to_list.append(col)
                    linewidth_list.append(linewidth)
                    linestyle_list.append(linestyle)

        edges = pd.DataFrame({
            'from' : from_list,
            'to' : to_list,
            'linewidth' : linewidth_list,
            'linestyle' : linestyle_list
        })
            
        chord_diagram(labels, edges) 
            
    #---------------------------------------------------------------------------------------------
    def circlerel(self, ascending_r=None, ascending_c=None, library=None):


        if library is None or library == 'altair':
            if ascending_r is None or ascending_r is True:
                sort_X = 'ascending'
            else:
                sort_X = 'descending'

            if ascending_c is None or ascending_c is True:
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

        if library == 'seaborn':
            sns.relplot(
                x = self.columns[0],
                y = self.columns[1],
                size = self.columns[2],
                #sizes = (10, 500),
                alpha = 0.8,
                palette = 'viridis',
                data = df)
            plt.xticks(rotation=90)
            return

    #---------------------------------------------------------------------------------------------
    def heatmap(self, ascending_r=None, ascending_c=None, figsize=(10, 10), library=None, 
        cmap='Blues'):
        """
        Available cmaps:

        https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
        """

        def cut_text(w):
            return w if len(w) < 35 else w[:31] + '... ' + w[w.find('['):]

        if library is None:

            x = self.tomatrix(ascending_r, ascending_c)

            ## rename columns and row index
            x.columns = [cut_text(w) for w in x.columns]
            x.index = [cut_text(w) for w in x.index]

            if self._rtype == 'factor-matrix':
                x = self.tomatrix(ascending_r, ascending_c)
                x = x.transpose()
                x = x.apply(lambda w: abs(w))
            plt.figure(figsize=figsize)
            plt.pcolor(np.transpose(x.values), cmap=cmap)
            plt.xticks(np.arange(len(x.index))+0.5, x.index, rotation='vertical')
            plt.yticks(np.arange(len(x.columns))+0.5, x.columns)
            #plt.gca().set_aspect('equal', 'box')
            plt.gca().invert_yaxis()

            ## annotation
            max_value = x.values.max() / 2.0
            for idx_row, row in enumerate(x.index):
                for idx_col, col in enumerate(x.columns):

                    if self._rtype == 'coo-matrix' and x.at[row, col] > 0:
                        
                        if x.at[row, col] > max_value:
                            color = 'white'
                        else:
                            color = 'black'

                        plt.text(
                            idx_row + 0.5, 
                            idx_col + 0.5, 
                            x.at[row, col],
                            ha="center", 
                            va="center", 
                            color=color)

                    elif self._rtype in ['auto-matrix', 'cross-matrix']:

                        if x.at[row, col] > 0.5:
                            color = 'white'
                        else:
                            color = 'black'

                        plt.text(
                            idx_row + 0.5, 
                            idx_col + 0.5, 
                            "{:3.2f}".format(x.at[row, col]),
                            ha="center", 
                            va="center", 
                            color=color)

                    elif self._rtype in ['factor-matrix']:

                        max_value = x.values.max() / 2.0
                        if x.at[row, col] > max_value:
                            color = 'white'
                        else:
                            color = 'black'

                        plt.text(
                            idx_row + 0.5, 
                            idx_col + 0.5, 
                            "{:3.2f}".format(x.at[row, col]),
                            ha="center", 
                            va="center", 
                            color=color)

            ## ends annotation

            plt.show()
            
            
        if library == 'altair':

            _self = self.copy()

            _self[_self.columns[0]] = _self[_self.columns[0]].map(lambda w: cut_text(w))
            _self[_self.columns[1]] = _self[_self.columns[1]].map(lambda w: cut_text(w))

            if ascending_r is None or ascending_r is True:
                sort_X = 'ascending'
            else:
                sort_X = 'descending'

            if ascending_c is None or ascending_c is True:
                sort_Y = 'ascending'
            else:
                sort_Y = 'descending'

            graph = alt.Chart(_self).mark_rect().encode(
                alt.X(_self.columns[0] + ':O', sort=sort_X),
                alt.Y(_self.columns[1] + ':O', sort=sort_Y),
                color=_self.columns[2] + ':Q')

            if self._rtype == 'coo-matrix':
                text = graph.mark_text(
                    align='center',
                    baseline='middle',
                    dx=5
                ).encode(
                    text=_self.columns[2] + ':Q'
                )
            else:
                text = None

            if text is not None:
                return graph 

            return graph

        if library == 'seaborn':

            sns.set()
            _self = self.tomatrix(ascending_r, ascending_c)
            _self = _self.transpose()
            _self.columns = [cut_text(w) for w in _self.columns]
            _self.index = [cut_text(w) for w in _self.index]

            return sns.heatmap(_self)

    #---------------------------------------------------------------------------------------------
    def kmeans(self, n_clusters=2):
        """Apply KMeans to a pandas dataframe.
        """

        if self._rtype not in [
            'coo-matrix',
            'cross-matrix',
            'auto-matrix']:

            raise Exception('Invalid function call for type: ' + self._rtype )

        if self._rtype == 'factor-matrix' is True:
            x = self.copy()
        else:
            x = self.tomatrix()

        m = KMeans(n_clusters)
        m.fit(x.values)

        centers = pd.DataFrame(
            np.transpose(m.cluster_centers_),
            columns = ['Cluster ' + str(i) for i in range(n_clusters)],
            index = x.columns)

        clusters = pd.DataFrame(
            {'cluster': m.predict(x.values)},
            index = x.index)

        return centers, clusters



    #---------------------------------------------------------------------------------------------
    def cluster_map(self, min_value=None, top_links=None, figsize = (10,10), 
            font_size=12, factor=None, size=(25,300)):

        ## cluster dataset
        cluster_data = self._cluster_data.copy()

        ## figure properties
        plt.figure(figsize=figsize)

        ## graph
        graph = nx.Graph()

        ## adds nodes to graph
        clusters = list(set(cluster_data.cluster))

        nodes = list(set(self.tomatrix().index))
        graph.add_nodes_from(clusters)
        graph.add_nodes_from(nodes)


        ## adds edges and properties
        weigth = []
        style = []
        value = []
        for _, row in cluster_data.iterrows():
            graph.add_edge(row[1], row[2])
            if row[3] >= 0.75:
                weigth += [4]
                style += ['solid']
                value += [row[3]]
            elif row[3] >= 0.50:
                weigth += [2]
                style += ['solid']
                value += [row[3]]
            elif row[3] >= 0.25:
                weigth += [1]
                style += ['dashed']
                value += [row[3]]
            else:
                weigth += [1]
                style += ['dotted']
                value += [row[3]]


        edges = pd.DataFrame({
            'edges' : graph.edges(),
            'weight' : weigth,
            'style' : style,
            'value' : value
        })

        
        ## edges from center of cluster to nodes.
        for _, row in cluster_data.iterrows():
            graph.add_edge(row[0], row[1]) 
            graph.add_edge(row[0], row[2])
        

        ## graph layout
        path_length = nx.shortest_path_length(graph)
        distances = pd.DataFrame(index=graph.nodes(), columns=graph.nodes())
        for row, data in path_length:
            for col, dist in data.items():
                distances.loc[row,col] = dist
        distances = distances.fillna(distances.max().max())
        layout = nx.kamada_kawai_layout(graph, dist=distances.to_dict())

        ## nodes drawing
        node_size = [x[(x.find('[')+1):-1] for x in nodes]
        node_size = [float(x) for x in node_size]
        max_node_size = max(node_size)
        min_node_size = min(node_size)
        node_size = [size[0] + x / (max_node_size - min_node_size) * size[1] for x in node_size]

        nx.draw_networkx_nodes(
            graph, 
            layout, 
            nodelist=nodes, 
            node_size=node_size,
            node_color='red')

        ## edges drawing
        for style in list(set(edges['style'].tolist())):

            edges_set = edges[edges['style'] == style]

            if len(edges_set) == 0:
                continue

            nx.draw_networkx_edges(
                graph, 
                layout,
                edgelist=edges_set['edges'].tolist(), 
                style=style,
                width=edges_set['weight'].tolist(),
                edge_color='black')


        ## node labels
        x_left, x_right = plt.xlim()
        y_left, y_right = plt.ylim()
        delta_x = (x_right - x_left) * 0.01
        delta_y = (y_right - y_left) * 0.01
        for node in nodes:
            x_pos, y_pos = layout[node]
            plt.text(
                x_pos + delta_x, 
                y_pos + delta_y, 
                node, 
                size=font_size,
                ha='left',
                va='bottom',
                bbox=dict(
                    boxstyle="square",
                    ec='lightgray',
                    fc='white',
                    ))

        # node_labels = dict(zip(nodes, nodes))
        # nx.draw_networkx_labels(
        #     graph,
        #     layout,
        #     labels=node_labels,
        #     font_size=font_size,
        #     ha='left',
        #     va='bottom',
        #     bbox=dict(boxstyle="square",
        #            ec='gray',
        #            fc='white',
        #            ))

        if factor is not None:
            left, right = plt.xlim()
            width = (right - left) * factor / 2.0
            plt.xlim(left - width, right + width)

        plt.axis('off')





    #---------------------------------------------------------------------------------------------
    def map(self, min_value=None, top_links=None, figsize = (10,10)):

        def add_group(group, cluster_number, weight, style):
            """
            """
            for row in group[col0].tolist():                    
                graph.add_edge(row, cluster_number,  weight=weight, style=style)

        ## main routine
        x = self.copy()

        ## column names in dataframe
        col0 = x.columns[0]
        col1 = x.columns[1]
        col2 = x.columns[2]
    
        ## node names
        nodes = sorted(list(set(x[col0]))) 
        clusters = sorted(list(set(x[col1])))
        
        ## generate a network graph
        plt.figure(figsize=figsize)
        graph = nx.Graph()
        graph.add_nodes_from(nodes, color='lightgray', size=12)
        graph.add_nodes_from(range(len(clusters)), color='red', size=3)

        ## add nodes to each cluster
        for idx_cluster, cluster in enumerate(clusters):
            
            ## obtains the data for cluster
            values = x[x[col1] == cluster]
            values = values.sort_values(col2, ascending=False)
            values.index = values[col0].tolist()     
            
            if top_links is not None and top_links < len(values):
                values = values.head(top_links)
            
            if min_value is None:
                group0 = values[values[col2] >= 0.75]
                group1 = values[(values[col2] < 0.75) & (values[col2] >= 0.50)]
                group2 = values[(values[col2] < 0.50) & (values[col2] >= 0.25)]
                group3 = values[values[col2] <= -0.25]
            else:
                group0 = values[ 
                    (values[col2] >= 0.75) & (values[col2].map(abs) >= min_value)
                    ]
                group1 = values[ 
                    (values[col2] < 0.75) & (values[col2] >= 0.50) & (values[col2].map(abs) >= min_value)
                    ]
                group2 = values[ 
                    (values[col2] < 0.50) & (values[col2] >= 0.25) & (values[col2].map(abs) >= min_value)
                    ]
                group3 = values[
                    (values[col2] <= -0.25) & (values[col2].map(abs) >= min_value)
                    ]

            if len(group0) > 0:
                add_group(group0, idx_cluster, weight=3, style='solid')
            if len(group1) > 0:
                add_group(group1, idx_cluster, weight=2, style='solid')
            if len(group2) > 0:
                add_group(group2, idx_cluster, weight=2, style='dashed')
            if len(group3) > 0:
                add_group(group3, idx_cluster, weight=1, style='dotted')
        
        
        
        ## calculate distance between relationated nodes to avoid overlaping
        path_length = nx.shortest_path_length(graph)
        distances = pd.DataFrame(index=graph.nodes(), columns=graph.nodes())
        for row, data in path_length:
            for col, dist in data.items():
                distances.loc[row,col] = dist
        distances = distances.fillna(distances.max().max())

        ## layout of graph
        layout = nx.kamada_kawai_layout(graph, dist=distances.to_dict())

        ## ------- no se modifican las propiedades de la grafica
        ## visual graph configuration
        ## nx.draw(graph, layout, with_labels=True)
        ## -------


        ## visualization
        nx.draw_networkx_nodes(
            graph, 
            layout, 
            nodelist=nodes, 
            node_size=11, 
            node_color='white')

        nx.draw_networkx_nodes(
            graph, 
            layout, 
            nodelist=list(range(len(clusters))), 
            node_size=80, 
            node_color='white')

        nx.draw_networkx_edges(
            graph, 
            layout,
            edge_color='gray')

        node_labels = dict(zip(nodes + list(range(len(clusters))), nodes + list(range(len(clusters)))))
        nx.draw_networkx_labels(
            graph,
            layout,
            labels=node_labels)

        plt.axis('off')


        # node_labels = dict(zip(nodes, nodes))
        # pos_labels = {}
        # y_off = 0.05  # offset on the y axis
    

        # for k, v in pos.items():
        #     plt.text(v[0], v[1], node_labels[k], backgroundcolor='white')

        # for k, v in pos.items():
        #     pos_labels[k] = (v[0], v[1]+y_off)

        # nx.draw_networkx_labels(
        #     graph,
        #     pos_labels,
        #     labels=node_labels,
        #     bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

        # nx.draw_networkx_nodes(
        #     graph, 
        #     pos, 
        #     nodelist=clusters, 
        #     node_size=3, 
        #     node_color='white')

        # nx.draw_networkx_edges(
        #     graph, 
        #     pos, 
        #     width=2, 
        #     edge_color="gray")


        # plt.figure(figsize=figsize)
        #nx.draw(graph, pos, with_labels=True)
        
        #print(graph.nodes())
        #print()
        #print(graph.edges())
        #nx.draw_spectral(graph)
        # plt.axis('off')





    #---------------------------------------------------------------------------------------------
    # def xxmap(self, min_value=None, top_links=None, figsize = (10,10)):
    #     """
    #     """

    #     def add_group(group, cluster, weight, style):
            
    #         for row in group[col0].tolist():                    
    #             graph.add_edge(row, cluster,  weight=weight, style=style)

    #     x = self
    #     ## nombres de las columnas
    #     col0 = x.columns[0]
    #     col1 = x.columns[1]
    #     col2 = x.columns[2]
    
    #     ## node names
    #     nodes = sorted(list(set(x[col0]))) 
    #     clusters = sorted(list(set(x[col1]))) 
        
    #     ## generate a network graph
    #     plt.figure(figsize=figsize)
    #     graph = nx.Graph()
    #     graph.add_nodes_from(nodes)
    #     graph.add_nodes_from(clusters) 


    #     for factor in list(set(x[col1])):
            
    #         values = x[x[col1] == factor]
    #         values = values.sort_values(col2, ascending=False)
    #         values.index = values[col0].tolist()     
    #         if top_links is not None and top_links < len(values):
    #             values = values.head(top_links)
                

            
    #         if min_value is None:
    #             group0 = values[values[col2] >= 0.75]
    #             group1 = values[(values[col2] < 0.75) & (values[col2] >= 0.50)]
    #             group2 = values[(values[col2] < 0.50) & (values[col2] >= 0.25)]
    #             group3 = values[values[col2] <= -0.25]
    #         else:
    #             group0 = values[ 
    #                 (values[col2] >= 0.75) & (values[col2].map(abs) >= min_value)
    #                 ]
    #             group1 = values[ 
    #                 (values[col2] < 0.75) & (values[col2] >= 0.50) & (values[col2].map(abs) >= min_value)
    #                 ]
    #             group2 = values[ 
    #                 (values[col2] < 0.50) & (values[col2] >= 0.25) & (values[col2].map(abs) >= min_value)
    #                 ]
    #             group3 = values[
    #                 (values[col2] <= -0.25) & (values[col2].map(abs) >= min_value)
    #                 ]

            
    #         if len(group0) > 0:
    #             add_group(group0, factor, weight=3, style='solid')
    #         if len(group1) > 0:
    #             add_group(group1, factor, weight=2, style='solid')
    #         if len(group2) > 0:
    #             add_group(group2, factor, weight=2, style='dashed')
    #         if len(group3) > 0:
    #             add_group(group3, factor, weight=1, style='dotted')
        
        
        
    #     # #calculate distance between relationated nodes to avoid overlaping
    #     path_length = nx.shortest_path_length(graph)
    #     distances = pd.DataFrame(index=graph.nodes(), columns=graph.nodes())
    #     for row, data in path_length:
    #         for col, dist in data.items():
    #             distances.loc[row,col] = dist
    #     distances = distances.fillna(distances.max().max())

    #     #layout of graph
    #     pos = nx.kamada_kawai_layout(graph, dist=distances.to_dict())

    #     #visual graph configuration
    #     nx.draw_networkx_nodes(
    #         graph, 
    #         pos, 
    #         nodelist=nodes, 
    #         node_size=11, 
    #         node_color='black')

    #     node_labels = dict(zip(nodes, nodes))
    #     pos_labels = {}
    #     y_off = 0.05  # offset on the y axis
    

    #     for k, v in pos.items():
    #         plt.text(v[0], v[1], node_labels[k], backgroundcolor='white')

    #     # for k, v in pos.items():
    #     #     pos_labels[k] = (v[0], v[1]+y_off)

    #     # nx.draw_networkx_labels(
    #     #     graph,
    #     #     pos_labels,
    #     #     labels=node_labels,
    #     #     bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

    #     nx.draw_networkx_nodes(
    #         graph, 
    #         pos, 
    #         nodelist=clusters, 
    #         node_size=3, 
    #         node_color='white')

    #     nx.draw_networkx_edges(
    #         graph, 
    #         pos, 
    #         width=2, 
    #         edge_color="gray")

    #     #nx.draw(graph, pos, with_labels=True)
        
    #     #print(graph.nodes())
    #     #print()
    #     #print(graph.edges())
    #     #nx.draw_spectral(graph)
    #     plt.axis('off')


    #---------------------------------------------------------------------------------------------
    # def _map(self, min_value=None, top_links=None, figsize = (10,10)):
    #     """
    #     """

    #     def add_group(group, weight, style):
            
    #         for row in group[col0].tolist():
    #             for col in group[col0].tolist():
    #                 if row != col:
    #                     #graph.add_edge(row, col, weight=weight, style=style)
    #                     graph.add_edge(row, col)


    #     x = self
    #     ## nombres de las columnas
    #     col0 = x.columns[0]
    #     col1 = x.columns[1]
    #     col2 = x.columns[2]
    
    #     ## node names
    #     nodes = sorted(list(set(x[col0])))
        
    #     ## generate a network graph
    #     plt.figure(figsize=figsize)
    #     graph = nx.Graph()
    #     graph.add_nodes_from(nodes)

    #     for factor in list(set(x[col1])):
            
    #         values = x[x[col1] == factor]
    #         values = values.sort_values(col2, ascending=False)
    #         values.index = values[col0].tolist()     
    #         if top_links is not None and top_links < len(values):
    #             values = values.head(top_links)
            
    #         if min_value is None:
    #             group0 = values[values[col2] >= 0.75]
    #             group1 = values[(values[col2] < 0.75) & (values[col2] >= 0.50)]
    #             group2 = values[(values[col2] < 0.50) & (values[col2] >= 0.25)]
    #             group3 = values[values[col2] <= -0.25]
    #         else:
    #             group0 = values[ 
    #                 (values[col2] >= 0.75) & (values[col2].map(abs) >= min_value)
    #                 ]
    #             group1 = values[ 
    #                 (values[col2] < 0.75) & (values[col2] >= 0.50) & (values[col2].map(abs) >= min_value)
    #                 ]
    #             group2 = values[ 
    #                 (values[col2] < 0.50) & (values[col2] >= 0.25) & (values[col2].map(abs) >= min_value)
    #                 ]
    #             group3 = values[
    #                 (values[col2] <= -0.25) & (values[col2].map(abs) >= min_value)
    #                 ]

            
    #         if len(group0) > 0:
    #             add_group(group0, weight=3, style='solid')
    #         if len(group1) > 0:
    #             add_group(group1, weight=2, style='solid')
    #         if len(group2) > 0:
    #             add_group(group2, weight=2, style='dashed')
    #         if len(group3) > 0:
    #             add_group(group3, weight=1, style='dotted')
        
        
        
    #     # #calculate distance between relationated nodes to avoid overlaping
    #     path_length = nx.shortest_path_length(graph)
    #     distances = pd.DataFrame(index=graph.nodes(), columns=graph.nodes())
    #     for row, data in path_length:
    #         for col, dist in data.items():
    #             distances.loc[row,col] = dist
    #     distances = distances.fillna(distances.max().max())

    #     #layout of graph
    #     pos = nx.kamada_kawai_layout(graph, dist=distances.to_dict())

    #     #visual graph configuration
    #     nx.draw(graph, pos, with_labels=True)
        
    #     #print(graph.nodes())
    #     #print()
    #     #print(graph.edges())
    #     #nx.draw_spectral(graph)


    #---------------------------------------------------------------------------------------------
    def map_to_chord(self, figsize = (10,10), minval=None, R=3):

        def add_group(group, linewidth, linestyle):
            
            if len(group) == 0:
                return

            for row in group[col0].tolist():
                for col in group[col0].tolist():

                    if row != col:    
                    
                        from_list.append(row)
                        to_list.append(col)
                        linewidth_list.append(linewidth)
                        linestyle_list.append(linestyle)


        x = self

        ## nombres de las columnas
        col0 = x.columns[0]
        col1 = x.columns[1]
        col2 = x.columns[2]

        labels = list(set(x[col0]))
        n_labels = len(labels)

        from_list = []
        to_list = []
        linewidth_list = []
        linestyle_list = []

        for factor in list(set(x[col1])):
            
            values = x[x[col1] == factor]
            values.index = values[col0].tolist()     
            
            if minval is None:
                group0 = values[values[col2] >= 0.75]
                group1 = values[(values[col2] < 0.75) & (values[col2] >= 0.50)]
                group2 = values[(values[col2] < 0.50) & (values[col2] >= 0.25)]
                group3 = values[values[col2] <= -0.25]
            else:
                group0 = values[(values[col2] >= 0.75) & (values[col2] >= minval)]
                group1 = values[(values[col2] < 0.75) & (values[col2] >= 0.50) & (values[col2] >= minval)]
                group2 = values[(values[col2] < 0.50) & (values[col2] >= 0.25) & (values[col2] >= minval)]
                group3 = values[(values[col2] <= -0.25) & (values[col2] >= minval)]

            
            if len(group0) > 0:
                add_group(group0, linewidth=3, linestyle='solid')
            if len(group1) > 0:
                add_group(group1, linewidth=2, linestyle='solid')
            if len(group2) > 0:
                add_group(group2, linewidth=2, linestyle='dashed')
            if len(group3) > 0:
                add_group(group3, linewidth=1, linestyle='dotted')
        
        edges = pd.DataFrame({
            'from' : from_list,
            'to' : to_list,
            'linewidth' : linewidth_list,
            'linestyle' : linestyle_list
        })
            
        chord_diagram(labels, edges) 


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
            'coo-matrix',
            'cross-matrix',
            'auto-matrix',
            'factor-matrix']:

            raise Exception('Invalid function call for type: ' + self._rtype )


        if self._rtype == 'factor-matrix':
            x = self.copy()
        else:
            x = self.tomatrix()

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
    def sankey_plot(self, figsize=(7,10), minval=None):
        """Cross-relation plot
        """
        if self._rtype != 'cross-matrix':
            Exception('Invalid matrix type:' + self._rtype)

        x = self
        
        llabels = sorted(list(set(x[x.columns[0]])))
        rlabels = sorted(list(set(x[x.columns[1]])))

        factorL = max(len(llabels)-1, len(rlabels)-1) / (len(llabels) - 1)
        factorR = max(len(llabels)-1, len(rlabels)-1) / (len(rlabels) - 1)

        lpos = {k:v*factorL for v, k in enumerate(llabels)}
        rpos = {k:v*factorR for v, k in enumerate(rlabels)}
        
        fig, ax1 = plt.subplots(figsize=(7, 10))
        ax1.scatter([0] * len(llabels), llabels, color='black', s=50)

        for index, r in x.iterrows():

            row = r[0]
            col = r[1]
            val = r[2]

            if val >= 0.75:
                linewidth = 4
                linestyle = '-' 
            elif val >= 0.50:
                linewidth = 2
                linstyle = '-' 
            elif val >= 0.25:
                linewidth = 2
                linestyle = '--' 
            elif val < 0.25:
                linewidth = 1
                linestyle = ':'
            else: 
                linewidth = 0
                linestyle = '-'

            if minval is  None:
                plt.plot(
                    [0, 1], 
                    [lpos[row], rpos[col]], 
                    linewidth=linewidth, 
                    linestyle=linestyle, 
                    color='black')
            elif abs(val) >= minval :
                plt.plot(
                    [0, 1], 
                    [lpos[row], rpos[col]], 
                    linewidth=linewidth, 
                    linestyle=linestyle, 
                    color='black')

        ax2 = ax1.twinx()
        ax2.scatter([1] * len(rlabels), rlabels, color='black', s=50)
        #ax2.set_ylim(0, len(rlabels)-1)
        
                
        for txt in ['bottom', 'top', 'left', 'right']:
            ax1.spines[txt].set_color('white')
            ax2.spines[txt].set_color('white')
        
        ax2.set_xticks([])
    
    #---------------------------------------------------------------------------------------------
    def tomatrix(self, ascending_r=None, ascending_c=None):
        """Displays a term by term dataframe as a matrix.

        >>> mtx = Matrix({
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

        >>> mtx.tomatrix() # doctest: +NORMALIZE_WHITESPACE
             c0   c1
        r0  1.0  4.0
        r1  5.0  2.0
        r2  3.0  6.0    

        """

        # if self._rtype not in [
        #     'coo-matrix',
        #     'cross-matrix',
        #     'auto-matrix']:

        #     raise Exception('Invalid function call for type: ' + self._rtype )


        if self.columns[0] == 'Year':
            termA_unique = range(min(self.Year), max(self.Year)+1)
        else:
            termA_unique = self.iloc[:,0].unique()
            
        if self.columns[1] == 'Year':
            termB_unique = range(min(self.Year), max(self.Year)+1)
        else:
            termB_unique = self.iloc[:,1].unique()
            
        if ascending_r is not None:
            termA_unique = sorted(termA_unique, reverse = not ascending_r)

        if ascending_c is not None:
            termB_unique = sorted(termB_unique, reverse = not ascending_c)

        if self._rtype == 'coo-matrix':
            result = pd.DataFrame(
                np.full((len(termA_unique), len(termB_unique)), 0)
            )

        else:
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
            
        return Matrix(result, rtype='matrix')

    #---------------------------------------------------------------------------------------------
    def transpose(self, *args, **kwargs):
        result = Matrix(super().transpose())
        result._rtype = self._rtype
        return result    

    #---------------------------------------------------------------------------------------------


    








