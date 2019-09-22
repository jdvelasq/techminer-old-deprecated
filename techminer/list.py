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
from techminer.aux import cut_text

class List(pd.DataFrame):
    """Class implementing a dataframe with results of first level analysis.
    """

    #----------------------------------------------------------------------------------------------
    @property
    def _constructor_expanddim(self):
        return self

    #----------------------------------------------------------------------------------------------
    def print_IDs(self):
        for idx, row in self.iterrows():
            print(row[0], ' (', len(row[-1]), ')',' : ', sep='', end='')
            for i in row[-1]:
                print(i, sep='', end='')
            print()

    #----------------------------------------------------------------------------------------------
    def barhplot(self, library=None, color=None):
        """Plots a pandas.DataFrame using Altair.
        """

        columns = self.columns.tolist()

        data = List(self.copy())
        if data.columns[1] != 'Cited by':
            data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
            data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))

        if library is None:
            if color is None:
                color = 'gray'
            if columns[0] == 'Year':
                data =  data.sort_values(by=columns[0], ascending=True)
            else:
                data =  data.sort_values(by=columns[1], ascending=True)
            data.plot.barh(columns[0], columns[1], color=color)
            plt.gca().xaxis.grid(True)
            return
            

        if library == 'altair':
            if color is None:
                color = 'Greys'
            columns = self.columns.tolist()
            if columns[0] == 'Year':
                data = data.sort_values(by=columns[0], ascending=False)
            return alt.Chart(data).mark_bar().encode(
                alt.Y(columns[0] + ':N', sort=alt.EncodingSortField(
                    field=columns[1] + ':Q')),
                alt.X(columns[1] + ':Q'),
                alt.Color(columns[1] + ':Q', scale=alt.Scale(scheme=color)))

        if library == 'seaborn':
            if color is None:
                color = 'gray'
            if columns[0] == 'Year':
                data = data.sort_values(by=columns[0], ascending=False)
            else:
                data = data.sort_values(by=columns[1], ascending=False)
            sns.barplot(
                x=columns[1],
                y=columns[0],
                data=data,
                label=columns[0],
                color=color)
            plt.gca().xaxis.grid(True)
            return

    #----------------------------------------------------------------------------------------------
    def barplot(self, library=None):
        """Vertical bar plot in Altair.
        """

        columns = self.columns.tolist()

        data = List(self.copy())
        data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
        data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))

        if library is None:
            data.plot.bar(columns[0], columns[1], color='gray')
            plt.gca().yaxis.grid(True)
            return

        if library == 'altair':
        
            return alt.Chart(data).mark_bar().encode(
                alt.X(columns[0] + ':N', sort=alt.EncodingSortField(field=columns[1] + ':Q')),
                alt.Y(columns[1] + ':Q'),
                alt.Color(columns[1] + ':Q', scale=alt.Scale(scheme='greys')))

        if library == 'seaborn':
            columns = data.columns.tolist()
            result = sns.barplot(
                y=columns[1],
                x=columns[0],
                data=data,
                label=columns[0],
                color="gray")
            _, labels = plt.xticks()
            result.set_xticklabels(labels, rotation=90)
            plt.gca().yaxis.grid(True)
            return

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






