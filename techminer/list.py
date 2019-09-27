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
from techminer.common import cut_text

class List(pd.DataFrame):
    """Class implementing a dataframe with results of first level analysis.
    """

    #----------------------------------------------------------------------------------------------
    @property
    def _constructor_expanddim(self):
        return self

    #----------------------------------------------------------------------------------------------
    def altair_barhplot(self, color='Greys'):
        """


        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> from techminer.dataframe import  *
        >>> rdf = RecordsDataFrame(
        ...     pd.read_json('./data/cleaned.json', orient='records', lines=True)
        ... )
        >>> rdf.documents_by_year().altair_barhplot()
        alt.Chart(...)

        .. image:: ../figs/altair_barhplot.jpg
            :width: 800px
            :align: center        
        
        """
        columns = self.columns.tolist()
        data = List(self.copy())
        if data.columns[1] != 'Cited by':
            data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
            data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))
        if columns[0] == 'Year':
            data = data.sort_values(by=columns[0], ascending=False)
        return alt.Chart(data).mark_bar().encode(
            alt.Y(columns[0] + ':N', sort=alt.EncodingSortField(
                field=columns[1] + ':Q')),
            alt.X(columns[1] + ':Q'),
            alt.Color(columns[1] + ':Q', scale=alt.Scale(scheme=color)))

    #----------------------------------------------------------------------------------------------
    def altair_barplot(self):
        """Vertical bar plot in Altair.

        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> from techminer.dataframe import  *
        >>> rdf = RecordsDataFrame(
        ...     pd.read_json('./data/cleaned.json', orient='records', lines=True)
        ... )
        >>> rdf.documents_by_year().altair_barplot()

        .. image:: ../figs/altair_barplot.jpg
            :width: 500px
            :align: center                
        """

        columns = self.columns.tolist()
        data = List(self.copy())
        data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
        data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))

        alt.Chart(data).mark_bar().encode(
            alt.X(columns[0] + ':N', sort=alt.EncodingSortField(field=columns[1] + ':Q')),
            alt.Y(columns[1] + ':Q'),
            alt.Color(columns[1] + ':Q', scale=alt.Scale(scheme='greys')))


    #----------------------------------------------------------------------------------------------
    def barhplot(self, color=None, figsize=(12,8)):
        """Plots a pandas.DataFrame using Altair.

        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> from techminer.dataframe import  *
        >>> rdf = RecordsDataFrame(
        ...     pd.read_json('./data/cleaned.json', orient='records', lines=True)
        ... )
        >>> rdf.documents_by_year().barhplot()

        .. image:: ../figs/barhplot.jpg
            :width: 600px
            :align: center   
        """

        plt.figure(figsize=figsize)

        columns = self.columns.tolist()
        data = List(self.copy())
        if data.columns[1] != 'Cited by':
            data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
            data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))

        if color is None:
            color = 'gray'
        if columns[0] == 'Year':
            data =  data.sort_values(by=columns[0], ascending=True)
        else:
            data =  data.sort_values(by=columns[1], ascending=True)
        data.plot.barh(columns[0], columns[1], color=color)
        plt.gca().xaxis.grid(True)
        



    #----------------------------------------------------------------------------------------------
    def barplot(self, color='gray', figsize=(8,12)):
        """Vertical bar plot in matplotlib.

        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> from techminer.dataframe import  *
        >>> rdf = RecordsDataFrame(
        ...     pd.read_json('./data/cleaned.json', orient='records', lines=True)
        ... )
        >>> rdf.documents_by_year().barplot()

        .. image:: ../figs/barplot.jpg
            :width: 600px
            :align: center                
        
        """

        columns = self.columns.tolist()

        plt.figure(figsize=figsize)
        data = List(self.copy())
        data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
        data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))
        data.plot.bar(columns[0], columns[1], color=color)
        plt.gca().yaxis.grid(True)

    #----------------------------------------------------------------------------------------------
    def print_IDs(self):
        for idx, row in self.iterrows():
            print(row[0], ' (', len(row[-1]), ')',' : ', sep='', end='')
            for i in row[-1]:
                print(i, sep='', end='')
            print()

    #----------------------------------------------------------------------------------------------
    def seaborn_barhplot(self, color='gray'):
        """

        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> from techminer.dataframe import  *
        >>> rdf = RecordsDataFrame(
        ...     pd.read_json('./data/cleaned.json', orient='records', lines=True)
        ... )
        >>> rdf.documents_by_year().seaborn_barhplot()
        

        .. image:: ../figs/seaborn_barhplot.jpg
            :width: 600px
            :align: center        

        """
        columns = self.columns.tolist()
        data = List(self.copy())
        if data.columns[1] != 'Cited by':
            data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
            data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))

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

    #----------------------------------------------------------------------------------------------
    def seaborn_barplot(self, color='gray'):
        """Vertical bar plot in Seaborn.

        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> from techminer.dataframe import  *
        >>> rdf = RecordsDataFrame(
        ...     pd.read_json('./data/cleaned.json', orient='records', lines=True)
        ... )
        >>> rdf.documents_by_year().altair_barplot()

        .. image:: ../figs/seaborn_barhplot.jpg
            :width: 800px
            :align: center                
        """

        columns = self.columns.tolist()
        data = List(self.copy())
        data[columns[0]] = data[columns[0]].map(str) + ' [' + data[columns[1]].map(str) + ']'
        data[data.columns[0]] = data[data.columns[0]].map(lambda x: cut_text(x))

        columns = data.columns.tolist()
        result = sns.barplot(
            y=columns[1],
            x=columns[0],
            data=data,
            label=columns[0],
            color=color)
        _, labels = plt.xticks()
        result.set_xticklabels(labels, rotation=90)
        plt.gca().yaxis.grid(True)


    #----------------------------------------------------------------------------------------------
    def title_view(self, column_values=None):
        """
        """


    #----------------------------------------------------------------------------------------------
    def wordcloud(self, figsize=(14, 7), max_font_size=50, max_words=100, 
            background_color="white"):
        """


        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> from techminer.dataframe import  *
        >>> rdf = RecordsDataFrame(
        ...     pd.read_json('./data/cleaned.json', orient='records', lines=True)
        ... )
        >>> rdf.documents_by_terms('Source title').wordcloud()

        .. image:: ../figs/wordcloud.jpg
            :width: 800px
            :align: center                
        """
        

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

        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> from techminer.dataframe import  *
        >>> from techminer.strings import  *
        >>> rdf = RecordsDataFrame(
        ...     pd.read_json('./data/cleaned.json', orient='records', lines=True)
        ... )
        >>> rdf['Country'] = rdf['Affiliations'].map(lambda x: extract_country(x, sep=';'))
        >>> rdf.documents_by_terms('Country', sep=';').head()
                  Country  Num Documents                                                 ID
        0           China             83  [[*3*], [*4*], [*6*], [*6*], [*7*], [*10*], [*...
        1          Taiwan             20  [[*14*], [*14*], [*17*], [*17*], [*17*], [*17*...
        2   United States             17  [[*3*], [*22*], [*23*], [*23*], [*26*], [*26*]...
        3  United Kingdom             15  [[*5*], [*7*], [*11*], [*11*], [*11*], [*28*],...
        4           India             15  [[*9*], [*50*], [*51*], [*56*], [*56*], [*57*]...

        >>> rdf.documents_by_terms('Country', sep=';').worldmap()

        .. image:: ../figs/worldmap.jpg
            :width: 800px
            :align: center
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







