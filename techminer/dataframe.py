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
    def autocorrelation(self, column, sep=None, N=20):
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

        >>> rdf.autocorrelation(column='A', sep=';') # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
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

        >>> rdf.autocorrelation(column='A', sep=';', N=3) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
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
        result = self.crosscorrelation(termA=column, termB=column, sepA=sep, sepB=sep, N=N)
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
    def crosscorrelation(self, termA, termB=None, sepA=None, sepB=None, N=20):
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

        >>> rdf.crosscorrelation(termA='c1', termB='c2', sepA=';', sepB=';') # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
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
        >>> rdf.crosscorrelation(termA='c1', termB='c2', sepA=';', sepB=';', N=3) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
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
 
        if termA == termB:
            sepB = None
            termB = None

        tdf_rows = self.tdf(termA, sepA, N)
        if termB is not None:
            tdf_cols = self.tdf(termB, sepB, N)
        else:
            tdf_cols = tdf_rows.copy()
            
        if termB is not None:
            col0 = termA
            col1 = termB
            col2 = 'Crosscorrelation'
        else:
            col0 = termA + ' (row)'
            col1 = termA + ' (col)'
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
    def documents_by_terms(self, term, sep=None):
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
        terms = self[term].dropna()
        if sep is not None:
            pdf = pd.DataFrame({
                term: [y.strip() for x in terms for y in x.split(sep) if x is not None]
            })
        else:
            pdf = pd.DataFrame({
                term: terms
            })

        result = pdf.groupby(term, as_index=False).size()
        result = pd.DataFrame({
            term : result.index,
            'Num Documents': result.tolist()
        })
        result = result.sort_values(by='Num Documents', ascending=False)

        return List(result)


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
        yearsdf = self[['Year']].copy()
        yearsdf['Year'] = yearsdf['Year'].map(lambda x: None if np.isnan(x) else x)
        yearsdf = yearsdf.dropna()
        yearsdf['Year'] = yearsdf['Year'].map(int)
        
        minyear = min(yearsdf.Year)
        maxyear = max(yearsdf.Year) + 1
        docs_per_year = pd.Series(0, index=range(minyear, maxyear))
        
        count_per_year = yearsdf.groupby('Year')[['Year']].count()
        docs_per_year[count_per_year.index] = count_per_year['Year']
        docs_per_year = docs_per_year.to_frame()
        docs_per_year['Year'] = docs_per_year.index
        docs_per_year.columns = ['Num Documents', 'Year']
        docs_per_year = docs_per_year[['Year', 'Num Documents']]
        docs_per_year.index = range(len(docs_per_year))
        
        if cumulative is True:
            docs_per_year['Num Documents'] = docs_per_year['Num Documents'].cumsum()

        return List(docs_per_year)

    #----------------------------------------------------------------------------------------------
    def factor_analysis(self, term, sep=None, n_components=2, N=10):

        x = self.documents_by_terms(term, sep=sep)
        terms = x.loc[:, term].tolist()
        if N is None or len(terms) <= N:
            N = len(terms)
        terms = sorted(terms[0:N])

        x = pd.DataFrame(
            data = np.zeros((len(self), len(terms))),
            columns = terms,
            index = self.index)

        for idx in self.index:
            w = self.loc[idx, term]
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
            term : rows,
            'Factor' : cols,
            'value' : values})

        return Matrix(result, rtype = 'factor-matrix')


    #----------------------------------------------------------------------------------------------
    def most_cited_documents(self, N=10):
        """ Returns the top N most cited documents.
        Args:
            N (int) : number of documents to be returned.

        Results:
            pandas.DataFrame
        """
        result = self.sort_values(by='Cited by', ascending=False)
        return result[['Title', 'Authors', 'Year', 'Cited by']][0:N]


    #----------------------------------------------------------------------------------------------
    @property
    def num_of_sources(self):
        return len(self['Source title'].unique())

    #----------------------------------------------------------------------------------------------
    def term_by_term(self, termA, termB, sepA=None, sepB=None, minmax=None):
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

        >>> rdf.term_by_term('A', 'B')
           A  B  Num Documents
        0  0  a              2
        1  1  b              2
        2  2  c              1
        3  3  d              1
        4  4  e              1

        >>> rdf.term_by_term('A', 'B', minmax=(2,8))
           A  B  Num Documents
        0  0  a              2
        1  1  b              2
        """
        
        df = self[[termA, termB]].dropna()


        ##
        ## Expande las dos columnas de los datos originales
        ##
        if sepA is None and sepB is None:
            df = df[[termA, termB]]
        
        if sepA is not None and sepB is None:
            
            t = [(x, y) for x, y in zip(df[termA], df[termB])]
            t = [(c, b) for a, b in t for c in a.split(sepA)]
            df = pd.DataFrame({
                termA: [a.strip() if isinstance(a, str) else a for a,b in t],
                termB: [b.strip() if isinstance(b, str) else b for a,b in t]
            })
            
        if sepA is None and sepB is not None:
        
            t = [(x, y) for x, y in zip(df[termA], df[termB])]
            t = [(a, c.strip()) for a, b in t for c in b.split(sepB)]
            df = pd.DataFrame({
                termA: [a.strip() if isinstance(a, str) else a for a,b in t],
                termB: [b.strip() if isinstance(b, str) else b for a,b in t]
            })

        if sepA is not None and sepB is not None:
        
            t = [(x, y) for x, y in zip(df[termA], df[termB])]
            t = [(c, b) for a, b in t for c in a.split(sepA)]
            t = [(a, c) for a, b in t for c in b.split(sepB)]
            df = pd.DataFrame({
                termA: [a.strip() if isinstance(a, str) else a for a,b in t],
                termB: [b.strip() if isinstance(b, str) else b for a,b in t]
            })

        x = df.groupby(by=[termA, termB]).size()
        a = [t for t,_ in x.index]
        b = [t for _,t in x.index]
        df = pd.DataFrame({
            termA: a,
            termB: b,
            'Num Documents': x.tolist()
        })

        if minmax is not None:

            minval, maxval = minmax
            df = df[ df[df.columns[2]] >= minval ]
            df = df[ df[df.columns[2]] <= maxval ]

        return Matrix(df, rtype='coo-matrix')

    #----------------------------------------------------------------------------------------------

