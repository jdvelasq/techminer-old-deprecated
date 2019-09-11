"""
TechMiner.RecordsDataFrame
==================================================================================================



"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from techminer.list import List
from techminer.matrix import Matrix


class RecordsDataFrame(pd.DataFrame):
    """Class to represent a dataframe of bibliographic records.
    """
    #----------------------------------------------------------------------------------------------
    @property
    def _constructor_expanddim(self):
        return self


    #----------------------------------------------------------------------------------------------
    def years_list(self):

        df = self[['Year']].copy()
        df['Year'] = df['Year'].map(lambda x: None if np.isnan(x) else x)
        df = df.dropna()
        df['Year'] = df['Year'].map(int)
        minyear = min(df.Year)
        maxyear = max(df.Year)
        return pd.Series(0, index=range(minyear, maxyear+1), name='Year')

    #----------------------------------------------------------------------------------------------
    #
    #  Analytical functions
    #
    #----------------------------------------------------------------------------------------------
    def documents_by_year(self, cumulative=False):
        """Computes the number of documents per year.

        >>> rdf = RecordsDataFrame({'Year': [2014, 2014, 2016, 2017, None, 2019]})
        >>> rdf.documents_by_year()
           Year  Num Documents
        0  2014              2
        1  2015              0
        2  2016              1
        3  2017              1
        4  2018              0
        5  2019              1
        >>> rdf.documents_by_year(cumulative=True)
           Year  Num Documents
        0  2014              2
        1  2015              2
        2  2016              3
        3  2017              4
        4  2018              4
        5  2019              5

        """

        count = self.groupby('Year')[['Year']].count()

        result = self.years_list()
        result = result.to_frame()
        result['Year'] = result.index
        result['Num Documents'] = 0
        result.at[count.index.tolist(), 'Num Documents'] = count['Year'].tolist()
        result.index = range(len(result))

        if cumulative is True:
            result['Num Documents'] = result['Num Documents'].cumsum()

        return List(result)

    #----------------------------------------------------------------------------------------------
    def terms_by_year(self, column, sep=None, top_n=None, minmax=None):
        """

        >>> rdf = RecordsDataFrame({
        ...   'Year':[2014,  2014, 2015, 2015, 2018, 2018, 2018 ],
        ...   'term':[   'a;b;a', 'b;c', 'c;b', 'd;a',  'e;b',  'a;a',  'b']
        ... })
        >>> rdf # doctest: +NORMALIZE_WHITESPACE
           Year   term
        0  2014  a;b;a
        1  2014    b;c
        2  2015    c;b
        3  2015    d;a
        4  2018    e;b
        5  2018    a;a
        6  2018      b

        >>> rdf.terms_by_year('term', sep=';')
          term  Year  Num Documents
        0    a  2014              2
        1    a  2015              1
        2    a  2018              2
        3    b  2014              2
        4    b  2015              1
        5    b  2018              2
        6    c  2014              1
        7    c  2015              1
        8    d  2015              1
        9    e  2018              1

        >>> rdf.terms_by_year('term',  minmax=(2,8), sep=';')
          term  Year  Num Documents
        0    a  2014              2
        2    a  2018              2
        3    b  2014              2
        5    b  2018              2

        >>> rdf.terms_by_year('term',  top_n=3, minmax=(1,8), sep=';')
          term  Year  Num Documents
        0    a  2014              2
        1    a  2015              1
        2    a  2018              2
        3    b  2014              2
        4    b  2015              1
        5    b  2018              2
        6    c  2014              1
        7    c  2015              1
        """

        df = self[[column, 'Year']].dropna()
        if sep is not None:
            df[column] = df[column].map(lambda x: x.split(sep) if x is not None else None)
            df = df.explode(column)
            
        df = df.groupby(by=[column, 'Year'], as_index=False).size()
        idx_term = [t for t,_ in df.index]
        idx_year = [t for _,t in df.index]
        df = pd.DataFrame({
            column: idx_term,
            'Year': idx_year,
            'Num Documents': df.tolist()
        })

        if top_n is not None:
            top = self.documents_by_terms(column, sep)
            if len(top) > top_n:
                top = top[0:top_n][column].tolist()
                selected = [True if row[0] in top else False for idx, row in df.iterrows()] 
                df = df[selected]

        if minmax is not None:
            minval, maxval = minmax
            df = df[ df[df.columns[2]] >= minval ]
            df = df[ df[df.columns[2]] <= maxval ]

        return Matrix(df, rtype='coo-matrix')







    #----------------------------------------------------------------------------------------------
    def autocorr(self, column, sep=None, N=20):
        """

            
        >>> rdf = RecordsDataFrame({
        ... 'A':['a;b', 'b', 'c;a', 'b;a', 'c', 'd', 'e','a;b;c', 'e;a', None]
        ... })
        >>> rdf # doctest: +NORMALIZE_WHITESPACE
               A
        0    a;b
        1      b
        2    c;a
        3    b;a
        4      c
        5      d
        6      e
        7  a;b;c
        8    e;a  
        9   None

        >>> rdf.autocorr(column='A', sep=';') # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
           A (row) A (col)  Autocorrelation
        0        a       a         1.000000
        1        b       b         1.000000
        2        e       e         1.000000
        3        c       c         1.000000
        4        d       d         1.000000
        5        a       b         0.670820
        6        b       a         0.670820
        7        a       c         0.516398
        8        c       a         0.516398
        9        a       e         0.316228
        10       e       a         0.316228
        11       b       c         0.288675
        12       c       b         0.288675
        13       d       e         0.000000
        14       d       c         0.000000
        15       d       b         0.000000
        16       d       a         0.000000
        17       e       d         0.000000
        18       a       d         0.000000
        19       e       c         0.000000
        20       e       b         0.000000
        21       c       e         0.000000
        22       b       d         0.000000
        23       b       e         0.000000
        24       c       d         0.000000

        >>> rdf.autocorr(column='A', sep=';', N=3) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
          A (row) A (col)  Autocorrelation
        0       a       a         1.000000
        1       b       b         1.000000
        2       c       c         1.000000
        3       a       b         0.670820
        4       b       a         0.670820
        5       a       c         0.516398
        6       c       a         0.516398
        7       b       c         0.288675
        8       c       b         0.288675

        """
        result = self.crosscorr(column_r=column, column_c=column, sep_r=sep, sep_c=sep, N=N)
        result._rtype = 'auto-matrix'
        return result

    #----------------------------------------------------------------------------------------------
    def citations_by_terms(self, column, sep=None):
        """Computes the number of citations to docuement per year.

        >>> rdf = RecordsDataFrame({
        ...   'term':     ['a;b', 'a', 'b', 'c', None, 'b'],
        ...   'Cited by': [   1,   2,   3,   4,     3,  7]
        ... })
        >>> rdf.citations_by_terms('term', sep=';')
          term  Cited by
        0    b        11
        1    c         4
        2    a         3
        """
        terms = self[column].dropna()
        if sep is not None:
            terms = [y.strip() for x in terms for y in x.split(sep) if x is not None]
        else:
            terms = terms.tolist()

        terms = list(set(terms))

        ## crea el dataframe de resultados
        result = pd.DataFrame({
            column : terms},
            index = terms)

        result['Cited by'] = 0

        ## suma de citaciones
        for index, row in self[[column, 'Cited by']].iterrows():
            if row[0] is not None:
                citations = row[1] if not np.isnan(row[1]) else 0
                if sep is not None:
                    for term in row[0].split(sep):
                        term = term.strip()
                        result.at[term, 'Cited by'] =  result.loc[term, 'Cited by'] + citations
                else:
                    result.at[row[0], 'Cited by'] =  result.loc[row[0], 'Cited by'] + citations

        result = result.sort_values(by='Cited by', ascending=False)
        result.index = range(len(result))

        return List(result)


    #----------------------------------------------------------------------------------------------
    def citations_by_terms_by_year(self, column, sep=None):
        """Computes the number of citations to docuement per year.

        >>> rdf = RecordsDataFrame({
        ...   'term':     ['a;b', 'a', 'b', 'c', None, 'b'],
        ...   'Cited by': [   1,   2,   3,   4,     3,  7]
        ... })
        >>> rdf.citations_by_terms_by_year('term', sep=';')
          term  Cited by
        0    b        11
        1    c         4
        2    a         3
        """






        terms = self[column].dropna()
        if sep is not None:
            terms = [y.strip() for x in terms for y in x.split(sep) if x is not None]
        else:
            terms = terms.tolist()

        terms = list(set(terms))

        ## crea el dataframe de resultados
        result = pd.DataFrame({
            column : terms},
            index = terms)
        result['Cited by'] = 0

        ## suma de citaciones
        for index, row in self[[column, 'Cited by']].iterrows():
            if row[0] is not None:
                citations = row[1] if not np.isnan(row[1]) else 0
                if sep is not None:
                    for term in row[0].split(sep):
                        term = term.strip()
                        result.at[term, 'Cited by'] =  result.loc[term, 'Cited by'] + citations
                else:
                    result.at[row[0], 'Cited by'] =  result.loc[row[0], 'Cited by'] + citations

        result = result.sort_values(by='Cited by', ascending=False)
        result.index = range(len(result))

        return List(result)

    #----------------------------------------------------------------------------------------------
    def citations_by_year(self, cumulative=False, yearcol='Year', citedcol='Cited by'):
        """Computes the number of citations to docuement per year.

        >>> rdf = RecordsDataFrame({
        ...   'Year': [2014, 2014, 2016, 2017, None, 2019],
        ...   'Cited by': [1, 2, 3, 4, 3, 7]
        ... })
        >>> rdf.citations_by_year()
           Year  Cited by
        0  2014         3
        1  2015         0
        2  2016         3
        3  2017         4
        4  2018         0
        5  2019         7

        """
        yearsdf = self[[yearcol]].copy()
        yearsdf[yearcol] = yearsdf[yearcol].map(lambda x: None if np.isnan(x) else x)
        yearsdf = yearsdf.dropna()
        yearsdf[yearcol] = yearsdf[yearcol].map(int)
        
        minyear = min(yearsdf[yearcol])
        maxyear = max(yearsdf[yearcol]) + 1
        
        citations_per_year = pd.Series(0, index=range(minyear, maxyear))
        df0 = self.groupby(['Year'], as_index=False).agg({
            'Cited by': np.sum
        })
        for idx, x in zip(df0['Year'], df0['Cited by']):
            citations_per_year[idx] = x
        citations_per_year = citations_per_year.to_frame()
        citations_per_year['Year'] = citations_per_year.index
        citations_per_year.columns = ['Cited by', 'Year']
        citations_per_year = citations_per_year[['Year', 'Cited by']]
        
        if cumulative is True:
            citations_per_year['Cited by'] = citations_per_year['Cited by'].cumsum()
            
        citations_per_year.index = range(len(citations_per_year))
        return List(citations_per_year)

    #----------------------------------------------------------------------------------------------
    def tdf(self, column, sep, N=20):
        """

        >>> rdf = RecordsDataFrame({
        ...   'col0': ['a', 'a;b', 'b', 'b;c', None, 'c']
        ... })
        >>> rdf.tdf('col0', sep=';') # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
             b    a    c
        0  0.0  1.0  0.0
        1  1.0  1.0  0.0
        2  1.0  0.0  0.0
        3  1.0  0.0  1.0
        4  0.0  0.0  0.0
        5  0.0  0.0  1.0        
        
        """

        ## computa los N terminos mas frecuentes
        x = self.documents_by_terms(column, sep=sep)
        terms = x[x.columns[0]].tolist()
        if N is not None and len(terms) > N:
            terms = terms[0:N]
        
        tdf = pd.DataFrame(
            data = np.zeros((len(self), len(terms))),
            columns = terms,
            index = self.index)

        for idx in self.index:
            txt = self.loc[idx, column]
            if txt is not None:
                if sep is not None:
                    txt = [t.strip() for t in txt.split(sep)]
                else:
                    txt = [txt.strip()] 
                for t in txt:
                    if t in terms:
                        tdf.at[idx, t] = 1

        return tdf

    #----------------------------------------------------------------------------------------------
    def crosscorr(self, column_r, column_c=None, sep_r=None, sep_c=None, N=20):
        """Computes autocorrelation and crosscorrelation.


        >>> rdf = RecordsDataFrame({
        ... 'c1':['a;b',   'b', 'c;a', 'b;a', 'c',   'd', 'e', 'a;b;c', 'e;a', None,  None],
        ... 'c2':['A;B;C', 'B', 'B;D', 'B;C', 'C;B', 'A', 'A', 'B;C',    None, 'B;E', None]
        ... })
        >>> rdf # doctest: +NORMALIZE_WHITESPACE
               c1     c2
        0     a;b  A;B;C
        1       b      B
        2     c;a    B;D
        3     b;a    B;C
        4       c    C;B
        5       d      A
        6       e      A
        7   a;b;c    B;C
        8     e;a   None
        9    None    B;E
        10   None   None

        >>> rdf.crosscorr(column_r='c1', column_c='c2', sep_r=';', sep_c=';') # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
           c1 c2  Crosscorrelation
        0   b  B          0.755929
        1   b  C          0.750000
        2   a  B          0.676123
        3   a  C          0.670820
        4   c  B          0.654654
        5   c  D          0.577350
        6   c  C          0.577350
        7   d  A          0.577350
        8   a  D          0.447214
        9   e  A          0.408248
        10  b  A          0.288675
        11  a  A          0.258199
        12  d  C          0.000000
        13  d  D          0.000000
        14  d  B          0.000000
        15  e  E          0.000000
        16  e  D          0.000000
        17  c  A          0.000000
        18  e  C          0.000000
        19  e  B          0.000000
        20  c  E          0.000000
        21  b  E          0.000000
        22  b  D          0.000000
        23  a  E          0.000000
        24  d  E          0.000000
        >>> rdf.crosscorr(column_r='c1', column_c='c2', sep_r=';', sep_c=';', N=3) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
          c1 c2  Crosscorrelation
        0  b  B          0.755929
        1  b  C          0.750000
        2  a  B          0.676123
        3  a  C          0.670820
        4  c  B          0.654654
        5  c  C          0.577350
        6  b  A          0.288675
        7  a  A          0.258199
        8  c  A          0.000000

        """ 
 
        if column_r == column_c:
            sep_c = None
            column_c = None

        tdf_rows = self.tdf(column_r, sep_r, N)
        if column_c is not None:
            tdf_cols = self.tdf(column_c, sep_c, N)
        else:
            tdf_cols = tdf_rows.copy()
            
        if column_c is not None:
            col0 = column_r
            col1 = column_c
            col2 = 'Crosscorrelation'
        else:
            col0 = column_r + ' (row)'
            col1 = column_r + ' (col)'
            col2 = 'Autocorrelation'

        termsA = tdf_rows.columns.tolist()
        termsB = tdf_cols.columns.tolist()

        result =  pd.DataFrame({
            col0 : [None] * (len(termsA) * len(termsB)),
            col1 : [None] * (len(termsA) * len(termsB)),
            col2 : [0.0] * (len(termsA) * len(termsB))   
        })

        idx = 0
        for a in termsA:
            for b in termsB:

                s1 = tdf_rows[a]
                s2 = tdf_cols[b]

                num = np.sum((s1 * s2))
                den = np.sqrt(np.sum(s1**2) * np.sum(s2**2))
                value =  num / den
                result.at[idx, col0] = a
                result.at[idx, col1] = b
                result.at[idx, col2] = value
                idx += 1

        result = result.sort_values(col2, ascending=False)
        result.index = range(len(result))
        return Matrix(result, rtype='cross-matrix')

    #----------------------------------------------------------------------------------------------
    def documents_by_terms(self, column, sep=None):
        """Computes the number of documents per term.


        >>> rdf = RecordsDataFrame({'letters': ['a', 'b', 'c', 'a', None, 'c']})
        >>> rdf.documents_by_terms('letters')
          letters  Num Documents
        0       a              2
        2       c              2
        1       b              1
        >>> rdf = RecordsDataFrame({'letters': ['a|b', 'b|d|a', 'c', 'a', None, 'c']})
        >>> rdf.documents_by_terms('letters', sep='|')
          letters  Num Documents
        0       a              3
        1       b              2
        2       c              2
        3       d              1

        """
        terms = self[column].dropna()
        if sep is not None:
            pdf = pd.DataFrame({
                column: [y.strip() for x in terms for y in x.split(sep) if x is not None]
            })
        else:
            pdf = pd.DataFrame({
                column: terms
            })

        result = pdf.groupby(column, as_index=False).size()
        result = pd.DataFrame({
            column : result.index,
            'Num Documents': result.tolist()
        })
        result = result.sort_values(by='Num Documents', ascending=False)

        return List(result)



    #----------------------------------------------------------------------------------------------
    def factor_analysis(self, column, sep=None, n_components=2, N=10):

        x = self.documents_by_terms(column, sep=sep)
        terms = x.loc[:, column].tolist()
        if N is None or len(terms) <= N:
            N = len(terms)
        terms = sorted(terms[0:N])

        x = pd.DataFrame(
            data = np.zeros((len(self), len(terms))),
            columns = terms,
            index = self.index)

        for idx in self.index:
            w = self.loc[idx, column]
            if w is not None:
                if sep is not None:
                    z = w.split(sep)
                else:
                    z = [w] 
                for k in z:
                    if k in terms:
                        x.loc[idx, k] = 1

        s = x.sum(axis = 1)
        x = x.loc[s > 0.0]

        pca = PCA(n_components=n_components)
        
        values = np.transpose(pca.fit(X=x.values).components_)

        cols = [['F'+str(i) for i in range(n_components)] for k in range(len(terms))]
        rows = [[t for n in range(n_components) ] for t in terms]
        values = [values[i,j] for i in range(len(terms)) for j in range(n_components)]

        cols = [e for row in cols for e in row]
        rows = [e for row in rows for e in row]
        #values = [e for row in values for e in row]

        result = pd.DataFrame({
            column : rows,
            'Factor' : cols,
            'value' : values})

        return Matrix(result, rtype = 'factor-matrix')


    #----------------------------------------------------------------------------------------------
    def most_cited_documents(self, top_n=10, min_value=None):
        """ Returns the top N most cited documents and citations > min_value .
        Args:
            top_n (int) : number of documents to be returned.
            min_value (int) : minimal number of citations

        Results:
            pandas.DataFrame
        """
        result = self.sort_values(by='Cited by', ascending=False)
        if min_value is not None:
            result = result[result['Cited by'] >= min_value]
        if top_n is not None and len(result) > top_n:
            result = result[0:top_n]
        return result[['Title', 'Authors', 'Year', 'Cited by']]


    #----------------------------------------------------------------------------------------------
    @property
    def num_of_sources(self):
        return len(self['Source title'].unique())

    #----------------------------------------------------------------------------------------------
    def terms_by_terms(self, column_r, column_c, sep_r=None, sep_c=None, minmax=None):
        """

    
        >>> rdf = RecordsDataFrame({
        ...   'A':[0, 1, 2, 3, 4, 0, 1],
        ...   'B':['a', 'b', 'c', 'd', 'e', 'a', 'b']
        ... })
        >>> rdf # doctest: +NORMALIZE_WHITESPACE
           A  B
        0  0  a
        1  1  b
        2  2  c
        3  3  d
        4  4  e
        5  0  a
        6  1  b    

        >>> rdf.terms_by_terms('A', 'B')
           A  B  Num Documents
        0  0  a              2
        1  1  b              2
        2  2  c              1
        3  3  d              1
        4  4  e              1

        >>> rdf.terms_by_terms('A', 'B', minmax=(2,8))
           A  B  Num Documents
        0  0  a              2
        1  1  b              2
        """
        
        df = self[[column_r, column_c]].dropna()


        ##
        ## Expande las dos columnas de los datos originales
        ##
        if sep_r is None and sep_c is None:
            df = df[[column_r, column_c]]
        
        if sep_r is not None and sep_c is None:
            
            t = [(x, y) for x, y in zip(df[column_r], df[column_c])]
            t = [(c, b) for a, b in t for c in a.split(sep_r)]
            df = pd.DataFrame({
                column_r: [a.strip() if isinstance(a, str) else a for a,b in t],
                column_c: [b.strip() if isinstance(b, str) else b for a,b in t]
            })
            
        if sep_r is None and sep_c is not None:
        
            t = [(x, y) for x, y in zip(df[column_r], df[column_c])]
            t = [(a, c.strip()) for a, b in t for c in b.split(sep_c)]
            df = pd.DataFrame({
                column_r: [a.strip() if isinstance(a, str) else a for a,b in t],
                column_c: [b.strip() if isinstance(b, str) else b for a,b in t]
            })

        if sep_r is not None and sep_c is not None:
        
            t = [(x, y) for x, y in zip(df[column_r], df[column_c])]
            t = [(c, b) for a, b in t for c in a.split(sep_r)]
            t = [(a, c) for a, b in t for c in b.split(sep_c)]
            df = pd.DataFrame({
                column_r: [a.strip() if isinstance(a, str) else a for a,b in t],
                column_c: [b.strip() if isinstance(b, str) else b for a,b in t]
            })

        x = df.groupby(by=[column_r, column_c]).size()
        a = [t for t,_ in x.index]
        b = [t for _,t in x.index]
        df = pd.DataFrame({
            column_r: a,
            column_c: b,
            'Num Documents': x.tolist()
        })

        if minmax is not None:

            minval, maxval = minmax
            df = df[ df[df.columns[2]] >= minval ]
            df = df[ df[df.columns[2]] <= maxval ]

        return Matrix(df, rtype='coo-matrix')

