"""
TechMiner.List
==================================================================================================

"""
import altair as alt
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from wordcloud import WordCloud, ImageColorGenerator

class List(pd.DataFrame):
    """Class implementing a dataframe with results of first level analysis.
    """

    #----------------------------------------------------------------------------------------------
    @property
    def _constructor_expanddim(self):
        return self

    #----------------------------------------------------------------------------------------------
    def print_IDs(self):
        result = {}
        for idx, ids in enumerate(self['ID']):
            print(self.loc[idx, self.columns[0]], ' : ', sep='', end='')
            for i in ids:
                print(i, sep='', end='')
            print()

    #----------------------------------------------------------------------------------------------
    def barhplot(self, library=None):
        """Plots a pandas.DataFrame using Altair.
        """

        columns = self.columns.tolist()

        if library is None:
            return self.plot.barh(columns[0], columns[1], color='gray')

        if library == 'altair':
            columns = self.columns.tolist()
            return alt.Chart(self).mark_bar().encode(
                alt.Y(columns[0] + ':N', sort=alt.EncodingSortField(field=columns[1] + ':Q')),
                alt.X(columns[1] + ':Q'),
                alt.Color(columns[1] + ':Q', scale=alt.Scale(scheme='greys')))

        if library == 'seaborn':
            return sns.barplot(
                x=columns[1],
                y=columns[0],
                data=self,
                label=columns[0],
                color="gray")


    #----------------------------------------------------------------------------------------------
    def barplot(self, library=None):
        """Vertical bar plot in Altair.
        """

        columns = self.columns.tolist()

        if library is None:
            return self.plot.bar(columns[0], columns[1], color='gray');

        if library == 'altair':
        
            return alt.Chart(self).mark_bar().encode(
                alt.X(columns[0] + ':N', sort=alt.EncodingSortField(field=columns[1] + ':Q')),
                alt.Y(columns[1] + ':Q'),
                alt.Color(columns[1] + ':Q', scale=alt.Scale(scheme='greys')))

        if library == 'seaborn':
            columns = self.columns.tolist()
            result = sns.barplot(
                y=columns[1],
                x=columns[0],
                data=self,
                label=columns[0],
                color="gray")
            _, labels = plt.xticks()
            result.set_xticklabels(labels, rotation=90)
            return result

    #----------------------------------------------------------------------------------------------
    def wordcloud(self, figsize=(14, 7), max_font_size=50, max_words=100, 
        background_color="white"):
        

        columns = self.columns.tolist()

        words = [row[0]  for _, row in self.iterrows() for i in range(row[1])]

        wordcloud = WordCloud(
            max_font_size=max_font_size, 
            max_words=max_words, 
            background_color=background_color).generate(' '.join(words))

        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

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
            lambda x: x.replace('United States', 'United States of America')
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
    def title_view(self, column_values=None):
        """
        """






