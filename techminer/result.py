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

class FirstLevelResult(pd.DataFrame):
    """First level results for analysis (list)
    """
    #----------------------------------------------------------------------------------------------
    @property
    def _constructor_expanddim(self):
        return self

    #----------------------------------------------------------------------------------------------
    def barhplot_in_altair(self):
        """Plots a pandas.DataFrame using Altair.
        """
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

    #----------------------------------------------------------------------------------------------
    def worldmap(self, figsize=(14, 7)):
        """Worldmap plot with the number of documents per country.
        """

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
