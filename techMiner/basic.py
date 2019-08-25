"""
Functions for basic analysis 
===============================================================================

Overview
-------------------------------------------------------------------------------

The functions in this module allows the user to transform fields in a dataframe.



Functions in this module
-------------------------------------------------------------------------------

"""

import pandas as pd
import numpy as np
import geopandas
import re
import altair as alt
import matplotlib.pyplot as plt


def documentsByTerm(df, term, sep=None):
    
    if sep is not None:
        terms = [x  for x in df[term] if x is not None]
        df = pd.DataFrame({
            term: [y.strip() for x in terms for y in x.split(sep) if x is not None]
        })
        
    return df.groupby(term).size()


def extractCountries(x, sep=';'):
    """

    >>> import pandas as pd
    >>> x = pd.DataFrame({
    ...     'Affiliations': [
    ...         'University, Cuba; University, Venezuela',
    ...         'Univesity, United States; Univesity, Singapore',
    ...         'University;',
    ...         'University; Univesity',
    ...         'University,',
    ...         'University',
    ...         None]    
    ... })
    >>> x['Affiliations'].map(lambda x: extractCountries(x))
    0             Cuba;Venezuela
    1    United States;Singapore
    2                       None
    3                       None
    4                       None
    5                       None
    6                       None
    Name: Affiliations, dtype: object
    """

    if x is None:
        return None

    ##
    ## lista generica de nombres de paises
    ##
    country_names = sorted(
        geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres')).name.tolist()
    )
        
    ## paises faltantes / nombres incompletos
    country_names.append('United States')           # United States of America
    country_names.append('Singapore')               #
    country_names.append('Russian Federation')      # Russia
    country_names.append('Czech Republic')          # Czechia
    country_names.append('Bosnia and Herzegovina')  # Bosnia and Herz.
    country_names.append('Malta')                   #

    ##
    ## Reemplazo de nombres de regiones administrativas
    ## por nombres de paises
    ##
    x = re.sub('Bosnia and Herzegovina', 'Bosnia and Herz.', x) 
    x = re.sub('Czech Republic', 'Czechia', x)
    x = re.sub('Russian Federation', 'Russia', x) 
    x = re.sub('Hong Kong', 'China', x)
    x = re.sub('Macau', 'China', x)
    x = re.sub('Macao', 'China', x)
    

    countries = [affiliation.split(',')[-1].strip() for affiliation in x.split(sep)]

    countries =  ';'.join(
        [country if country in country_names else '' for country in countries]
    )
    
    if countries == '' or countries == ';':
        return None
    else:
        return countries


def documentsByYear(df, plot=False, cumulative=False):
    
    docs_per_year = pd.Series(0, index=range(min(df.Year), max(df.Year)+1))
    
    df0 = df.groupby('Year')[['Year']].count()
    for idx, x in zip(df0.index, df0.Year):
        docs_per_year[idx] = x
    docs_per_year = docs_per_year.to_frame()
    docs_per_year['Year'] = docs_per_year.index
    docs_per_year.columns = ['Documents', 'Year']
    
    if cumulative is True:
        docs_per_year.Documents = docs_per_year.Documents.cumsum()

    if plot is True:
        return alt.Chart(docs_per_year).mark_bar().encode(
            alt.X('Year:Q',
                  axis=alt.Axis(labelAngle=270)),
            y='Documents:Q'
        )
    else:
        return docs_per_year


def citationsByYear(df, plot=False, cumulative=False):
    citations_per_year = pd.Series(0, index=range(min(df.Year), max(df.Year)+1))
    df0 = df.groupby(['Year'], as_index=False).agg({
        'Cited by': np.sum
    })
    for idx, x in zip(df0['Year'], df0['Cited by']):
        citations_per_year[idx] = x
    citations_per_year = citations_per_year.to_frame()
    citations_per_year['Year'] = citations_per_year.index
    citations_per_year.columns = ['Citations', 'Year']
    
    if cumulative is True:
        citations_per_year['Citations'] = citations_per_year['Citations'].cumsum()
        
    if plot is True:
        return alt.Chart(citations_per_year).mark_bar().encode(
            alt.X('Year:Q',
                  axis=alt.Axis(labelAngle=270)),
            y='Citations:Q'
        )
    else:
        return citations_per_year


def termByTerm(df, termA, termB, sepA=None, sepB=None, 
               ascendingA=None, ascendingB=None, minmax=None,
               plot=False, figsize=(10,10)):
    
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
    
    if plot is True:
        plt.figure(figsize=figsize)
        plt.pcolor(result.values, cmap='Greys')
        plt.xticks(np.arange(len(result.columns))+0.5, result.columns, rotation='vertical')
        plt.yticks(np.arange(len(result.index))+0.5, result.index)
        plt.gca().set_aspect('equal', 'box')
        plt.gca().invert_yaxis()
        
    else:
        return result







