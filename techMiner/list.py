# """
# Functions for list 
# ===============================================================================

# Overview
# -------------------------------------------------------------------------------

# The functions in this module allows the user to transform fields in a dataframe.



# Functions in this module
# -------------------------------------------------------------------------------

# """

# import pandas as pd
# import numpy as np
# import string
# import re
# import json
# import altair as alt
# import matplotlib.pyplot as plt
# import seaborn as sns

# import geopandas
# import geoplot
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# from techMiner.strings import asciify, fingerprint

# def documents_by_term(df, term, sep=None):
#     """Computes the number of documents per term. 
#     """
    
#     if sep is not None:
#         terms = [x  for x in df[term] if x is not None]
#         df = pd.DataFrame({
#             term: [y.strip() for x in terms for y in x.split(sep) if x is not None]
#         })
#     result = df.groupby(term, as_index=False).size()
#     result = pd.DataFrame({
#         term : result.index,
#         'Num Documents': result.tolist()
#     })
#     return result.sort_values(by='Num Documents', ascending=False)



# def documents_by_year(df, cumulative=False):
#     """Computes the number of documents per year.
#     """
#     docs_per_year = pd.Series(0, index=range(min(df.Year), max(df.Year)+1))
    
#     df0 = df.groupby('Year')[['Year']].count()
#     for idx, x in zip(df0.index, df0.Year):
#         docs_per_year[idx] = x
#     docs_per_year = docs_per_year.to_frame()
#     docs_per_year['Year'] = docs_per_year.index
#     docs_per_year.columns = ['Num Documents', 'Year']
#     docs_per_year = docs_per_year[['Year', 'Num Documents']]
    
#     if cumulative is True:
#         docs_per_year['Num Documents'] = docs_per_year['Num Documents'].cumsum()

#     return docs_per_year


# def citations_by_year(df, cumulative=False):
#     """Computes the number of citations to docuement per year.
#     """
#     citations_per_year = pd.Series(0, index=range(min(df.Year), max(df.Year)+1))
#     df0 = df.groupby(['Year'], as_index=False).agg({
#         'Cited by': np.sum
#     })
#     for idx, x in zip(df0['Year'], df0['Cited by']):
#         citations_per_year[idx] = x
#     citations_per_year = citations_per_year.to_frame()
#     citations_per_year['Year'] = citations_per_year.index
#     citations_per_year.columns = ['Cited by', 'Year']
#     citations_per_year = citations_per_year[['Year', 'Cited by']]
    
#     if cumulative is True:
#         citations_per_year['Cited by'] = citations_per_year['Cited by'].cumsum()
        
#     return citations_per_year


# def alt_barh_graph(x):
#     """Plots a pandas.DataFrame using Altair.

#     Args:
#         x (pandas.DataFrame): dataframe returned by documents_by_term function.

#     Returns:
#         Altair object.

#     """
#     return alt.Chart(x).mark_bar().encode(
#             alt.Y(x.columns[0] + ':N', sort=alt.EncodingSortField(field=x.columns[1] + ':Q')),
#             alt.X(x.columns[1] + ':Q'),
#             alt.Color(x.columns[1] + ':Q', scale=alt.Scale(scheme='greys'))
#             )


# def alt_bar_graph(x):
#     """Plots a pandas.DataFrame using Altair.

#     Args:
#         x (pandas.DataFrame): dataframe returned by documents_by_term function.

#     Returns:
#         Altair object.

#     """
#     return alt.Chart(x).mark_bar().encode(
#             alt.X(x.columns[0] + ':N', sort=alt.EncodingSortField(field=x.columns[1] + ':Q')),
#             alt.Y(x.columns[1] + ':Q'),
#             alt.Color(x.columns[1] + ':Q', scale=alt.Scale(scheme='greys'))
#             )


# def sns_barh_plot(x):
#     """Plots a pandas.DataFrame using Seaborn.

#     Args:
#         x (pandas.DataFrame): dataframe returned by documents_by_term function.

#     Returns:
#         Searborn object.

#     """
#     return sns.barplot(
#         x="Num Documents", 
#         y=x.columns[0], 
#         data=x,
#         label=x.columns[0], 
#         color="gray")

# def sns_bar_plot(x):
#     """Plots a pandas.DataFrame using Seaborn.

#     Args:
#         x (pandas.DataFrame): dataframe returned by documents_by_term function.

#     Returns:
#         Searborn object.

#     """
#     result = sns.barplot(
#         y="Num Documents", 
#         x=x.columns[0], 
#         data=x,
#         label=x.columns[0], 
#         color="gray")
#     _, labels = plt.xticks()
#     result.set_xticklabels(labels, rotation=90)
#     return result

# def worldmap(x, figsize=(14,7)):
#     world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#     world = world[world.name!="Antarctica"]
#     world['q'] = 0
#     world.index = world.name

#     x.Country = [w if w != 'United States' else 'United States of America' for w in x.Country]
#     x.index = x.Country
#     for country in x.Country:
#         if country in world.index:
#             world.at[country, 'q'] = x.loc[country, 'Num Documents']    
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.1)
#     world.plot(column='q',  legend=True, ax=ax, cax=cax, cmap='Pastel2');


