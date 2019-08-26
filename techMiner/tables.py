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
import geopandas
import re
import altair as alt
import matplotlib.pyplot as plt


def stack(x, xlabel, ylabel=None, vlabel=None):
    """
    >>> import pandas as pd
    >>> x = pd.DataFrame({
    ...    'A': [0, 1, 2],
    ...    'B': [3, 4, 5],
    ...    'C': [6, 7, 8]},
    ...    index=['a', 'b', 'c'])
    >>> x
       A  B  C
    a  0  3  6
    b  1  4  7
    c  2  5  8

    >>> stack(x, xlabel='X', ylabel='Y', vlabel='Z')
       Z  X  Y
    0  0  a  A
    1  3  a  B
    2  6  a  C
    3  1  b  A
    4  4  b  B
    5  7  b  C
    6  2  c  A
    7  5  c  B
    8  8  c  C

    >>> x = pd.DataFrame({
    ...    'A': [0, 1, 2]},
    ...    index=['a', 'b', 'c'])
    >>> stack(x, xlabel='X')
       A  X
    0  0  a
    1  1  b
    2  2  c

    >>> stack(x.A, xlabel='X')
          A  X
       0  0  0
       1  1  1
       2  2  2

    """
    if isinstance(x, pd.Series):
        x = pd.DataFrame(x)
        x.columns = [xlabel]

    if isinstance(x, pd.DataFrame):
        if len(x.columns) == 1:
            x[x.index.name] = x.index
            x.index = range(len(x.index))
            return x
        else:
            z = x.stack()
            xindex = [i for i,_ in z.index]
            yindex = [i for _,i in z.index]
            z = pd.DataFrame({vlabel: z.tolist()})
            z[xlabel] = xindex
            z[ylabel] = yindex
            return z



def documentsByTerm(df, term, sep=None):
    
    if sep is not None:
        terms = [x  for x in df[term] if x is not None]
        df = pd.DataFrame({
            term: [y.strip() for x in terms for y in x.split(sep) if x is not None]
        })
        
    return df.groupby(term).size()



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
               ascendingA=None, ascendingB=None, minmax=None):
    
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
    
    return result







