"""
TechMiner.RecordsDataFrame
==================================================================================================



"""
import pandas as pd
import math
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
    def generate_ID(self):
        """Generates a unique ID for each document.
        """
        self['ID'] = [ '[*'+str(x)+ '*]' for x in range(len(self))]
        self.index = self['ID']
        return self



    #----------------------------------------------------------------------------------------------
    def get_records_by_IDs(self, IDs):
        """
        """
        selected = self[['ID']]
        selected['Selected'] = False
        for ID in IDs:
            selected['Selected'] = selected['Selected'] | (self['ID'] == ID)
        return self[selected]

    #----------------------------------------------------------------------------------------------
    #
    #  Analytical functions --- Num documents
    #
    #----------------------------------------------------------------------------------------------
    def documents_by_terms(self, column, sep=None, top_n=None, minmax=None):
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

        # computes the number of documents by term
        data = self[[column, 'ID']].dropna()
        if sep is not None:
            data[column] = data[column].map(lambda x: x.split(sep) if x is not None else None)
            data[column] = data[column].map(
                lambda x: [z.strip() for z in x] if isinstance(x, list) else x
            )
            data = data.explode(column)
            data.index = range(len(data))
        numdocs = data.groupby(column, as_index=False).size()
        

        ## dataframe with results
        result = pd.DataFrame({
            column : numdocs.index,
            'Num Documents' : numdocs.tolist()
        })
        result = result.sort_values(by='Num Documents', ascending=False)
        result.index = result[column]

        ## compute top_n terms
        if top_n is not None and len(result) > top_n:
            result = result.head(top_n)

        if minmax is not None:
            minval, maxval = minmax
            result = result[ result[result.columns[1]] >= minval ]
            result = result[ result[result.columns[1]] <= maxval ]


        result['ID'] = None
        for current_term in result[result.columns[0]]:
            selected_IDs = data[data[column] == current_term]['ID']
            if len(selected_IDs):
                result.at[current_term,'ID'] = selected_IDs.tolist()

        result.index = range(len(result))

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

        ## number of documents by year
        numdocs = self.groupby('Year')[['Year']].count()

        ## dataframe with results
        result = self.years_list()
        result = result.to_frame()
        result['Year'] = result.index
        result['Num Documents'] = 0
        result.at[numdocs.index.tolist(), 'Num Documents'] = numdocs['Year'].tolist()
        result.index = result['Year']

        if cumulative is True:
            result['Num Documents'] = result['Num Documents'].cumsum()

        result['ID'] = None
        for current_term in result['Year']:
            selected_IDs = self[self['Year'] == current_term]['ID']            
            if len(selected_IDs):
                result.at[current_term,'ID'] = selected_IDs.tolist()

        result.index = range(len(result))

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

        ## computes the number of documents by year
        data = self[[column, 'Year', 'ID']].dropna()
        if sep is not None:
            data[column] = data[column].map(lambda x: x.split(sep) if x is not None else None)
            data[column] = data[column].map(
                lambda x: [z.strip() for z in x] if isinstance(x, list) else x
            )
            data = data.explode(column)    
            data.index = range(len(data))
        numdocs = data.groupby(by=[column, 'Year'], as_index=False).size()

        ## dataframe with results
        idx_term = [t for t,_ in numdocs.index]
        idx_year = [t for _,t in numdocs.index]
        result = pd.DataFrame({
            column : idx_term,
            'Year' : idx_year,
            'Num Documents' : numdocs.tolist()
        })

        ## compute top_n terms
        if top_n is not None:
            top = self.documents_by_terms(column, sep)
            if len(top) > top_n:
                top = top[0:top_n][column].tolist()
                selected = [True if row[0] in top else False for idx, row in result.iterrows()] 
                result = result[selected]

        ## range of values
        if minmax is not None:
            minval, maxval = minmax
            result = result[ result[result.columns[2]] >= minval ]
            result = result[ result[result.columns[2]] <= maxval ]
        
        result['ID'] = None
        for idx, row in result.iterrows():
            current_term = row[0]
            year = row[1]
            selected_IDs = data[(data[column] == current_term) & (data['Year'] == year)]['ID']
            if len(selected_IDs):
                result.at[idx, 'ID'] = selected_IDs.tolist()

        result.index = range(len(result))

        return Matrix(result, rtype='coo-matrix')

    #----------------------------------------------------------------------------------------------
    def terms_by_terms(self, column_r, column_c, sep_r=None, sep_c=None, top_n=None, minmax=None):
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
        
        ## computes the number of documents by term by term
        data = self[[column_r, column_c, 'ID']].dropna()
        if sep_r is not None:
            data[column_r] = data[column_r].map(lambda x: x.split(sep_r) if x is not None else None)
            data[column_r] = data[column_r].map(
                lambda x: [z.strip() for z in x] if isinstance(x, list) else x
            )
            data = data.explode(column_r)
            data.index = range(len(data))
        if sep_c is not None:
            data[column_c] = data[column_c].map(lambda x: x.split(sep_c) if x is not None else None)
            data[column_c] = data[column_c].map(
                lambda x: [z.strip() for z in x] if isinstance(x, list) else x
            )
            data = data.explode(column_c)   
            data.index = range(len(data))
        numdocs = data.groupby(by=[column_r, column_c]).size()

        ## results dataframe
        a = [t for t,_ in numdocs.index]
        b = [t for _,t in numdocs.index]
        result = pd.DataFrame({
            column_r : a,
            column_c : b,
            'Num Documents' : numdocs.tolist()
        })

        ## compute top_n terms
        if top_n is not None:
            ## rows
            top = self.documents_by_terms(column_r, sep_r)
            if len(top) > top_n:
                top = top[0:top_n][column_r].tolist()
                selected = [True if row[0] in top else False for idx, row in result.iterrows()] 
                result = result[selected]

            ## cols
            top = self.documents_by_terms(column_c, sep_c)
            if len(top) > top_n:
                top = top[0:top_n][column_c].tolist()
                selected = [True if row[1] in top else False for idx, row in result.iterrows()] 
                result = result[selected]
            
        if minmax is not None:
            minval, maxval = minmax
            result = result[ result[result.columns[2]] >= minval ]
            result = result[ result[result.columns[2]] <= maxval ]


        ## collects the references
        result['ID'] = None
        for idx, row in result.iterrows():
            term0 = row[0]
            term1 = row[1]
            selected_IDs = data[(data[column_r] == term0) & (data[column_c] == term1)]['ID']
            if len(selected_IDs):
                result.at[idx, 'ID'] = selected_IDs.tolist()

        result.index = range(len(result))

        ## adds number of records to columns
        num = self.documents_by_terms(column_r, sep_r)
        new_names = {}
        for idx, row in num.iterrows():
            old_name = row[0]
            new_name = old_name + ' [' + str(row[1]) + ']'
            new_names[old_name] = new_name

        result[column_r] = result[column_r].map(lambda x: new_names[x])

        num = self.documents_by_terms(column_c, sep_c)
        new_names = {}
        for idx, row in num.iterrows():
            old_name = row[0]
            new_name = old_name + ' [' + str(row[1]) + ']'
            new_names[old_name] = new_name

        result[column_c] = result[column_c].map(lambda x: new_names[x])

        return Matrix(result, rtype='coo-matrix')

    #----------------------------------------------------------------------------------------------
    def terms_by_terms_by_year(self, column_r, column_c, sep_r=None, sep_c=None, top_n=None, minmax=None):
        """

    
        >>> rdf = RecordsDataFrame({
        ...   'Year'  : [   2013,   2013,  2014,    2014,  2015,  2016, 2016],
        ...   'term0' : [ '0;1;2', '1;2', '2;1',     '3',   '4',   '0',  '1'],
        ...   'term1' : [     'a',   'a',   'c', 'd,a,b', 'e,b', 'a,b',  'b']
        ... })
        >>> rdf # doctest: +NORMALIZE_WHITESPACE
           Year  term0  term1
        0  2013  0;1;2      a
        1  2013    1;2      a
        2  2014    2;1      c
        3  2014      3  d,a,b
        4  2015      4    e,b
        5  2016      0    a,b
        6  2016      1      b          

        >>> rdf.terms_by_terms_by_year('term0', 'term1', sep_r=';', sep_c=',')
           term0 term1  Year  Num Documents
        0      0     a  2013              1
        1      0     a  2016              1
        2      0     b  2016              1
        3      1     a  2013              2
        4      1     b  2016              1
        5      1     c  2014              1
        6      2     a  2013              2
        7      2     c  2014              1
        8      3     a  2014              1
        9      3     b  2014              1
        10     3     d  2014              1
        11     4     b  2015              1
        12     4     e  2015              1

        >>> rdf.terms_by_terms_by_year('term0', 'term1', sep_r=';', sep_c=',', top_n=3)
          term0 term1  Year  Num Documents
        0     0     a  2013              1
        1     0     a  2016              1
        2     0     b  2016              1
        3     1     a  2013              2
        4     1     b  2016              1
        5     1     c  2014              1
        6     2     a  2013              2
        7     2     c  2014              1

        >>> rdf.terms_by_terms_by_year('term0', 'term1', minmax=(2,8), sep_r=';', sep_c=',')
          term0 term1  Year  Num Documents
        3     1     a  2013              2
        6     2     a  2013              2


        """
        
        ## computes the number of documents by term by term
        data = self[[column_r, column_c, 'Year', 'ID']].dropna()
        if sep_r is not None:
            data[column_r] = data[column_r].map(lambda x: x.split(sep_r))
            data[column_r] = data[column_r].map(
                lambda x: [z.strip() for z in x] if isinstance(x, list) else x
            )
            data = data.explode(column_r)
            data.index = range(len(data))
        if sep_c is not None:
            #data = data[[column_c, column_r, 'Year']]
            data[column_c] = data[column_c].map(lambda x: x.split(sep_c))
            data[column_c] = data[column_c].map(
                lambda x: [z.strip() for z in x] if isinstance(x, list) else x
            )
            data = data.explode(column_c)   
            data.index = range(len(data))
        
        numdocs = data.groupby(by=[column_r, column_c, 'Year']).size()

        ## results dataframe
        a = [t for t,_,_ in numdocs.index]
        b = [t for _,t,_ in numdocs.index]
        y = [t for _,_,t in numdocs.index]
        result = pd.DataFrame({
            column_r : a,
            column_c : b,
            'Year' : y,
            'Num Documents' : numdocs.tolist()
        })

        ## compute top_n terms
        if top_n is not None:
            ## rows
            top = self.documents_by_terms(column_r, sep_r)
            if len(top) > top_n:
                top = top[0:top_n][column_r].tolist()
                selected = [True if row[0] in top else False for idx, row in result.iterrows()] 
                result = result[selected]

            ## cols
            top = self.documents_by_terms(column_c, sep_c)
            if len(top) > top_n:
                top = top[0:top_n][column_c].tolist()
                selected = [True if row[1] in top else False for idx, row in result.iterrows()] 
                result = result[selected]
            
        if minmax is not None:
            minval, maxval = minmax
            result = result[ result[result.columns[3]] >= minval ]
            result = result[ result[result.columns[3]] <= maxval ]

        result['ID'] = None
        for idx, row in result.iterrows():
            term0 = row[0]
            term1 = row[1]
            term2 = row[2]
            selected_IDs = data[
                (data[column_r] == term0) & (data[column_c] == term1) & (data['Year'] == term2)
            ]['ID']
            if len(selected_IDs):
                result.at[idx, 'ID'] = selected_IDs.tolist()

        return Matrix(result, rtype='coo-matrix-year')


    #----------------------------------------------------------------------------------------------
    #
    #  Analytical functions --- Citations
    #
    #----------------------------------------------------------------------------------------------
    def citations_by_terms(self, column, sep=None, top_n=None, minmax=None):
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
        citations = self[[column, 'Cited by']]
        citations = citations.dropna()
        if sep is not None:
            citations[column] = citations[column].map(lambda x: x.split(sep) if x is not None else None)
            citations[column] = citations[column].map(
                lambda x: [z.strip() for z in x] if isinstance(x, list) else x
            )
            citations = citations.explode(column)
            citations.index = range(len(citations))
        citations = citations.groupby([column], as_index=True).agg({
            'Cited by': np.sum
        })

        result = pd.DataFrame({
            column : citations.index,
            'Cited by' : citations['Cited by'].tolist()
        })
        result = result.sort_values(by='Cited by', ascending=False)
        result.index = range(len(result))

        if top_n is not None and len(result) > top_n:
            result = result.head(top_n)

        if minmax is not None:
            minval, maxval = minmax
            result = result[ result[result.columns[1]] >= minval ]
            result = result[ result[result.columns[1]] <= maxval ]

        return List(result)

    #----------------------------------------------------------------------------------------------
    def citations_by_year(self, cumulative=False):
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

        ## computes number of citations
        citations = self[['Year', 'Cited by']]
        citations = citations.dropna()
        citations['Year'] = citations['Year'].map(int)
        citations = citations.groupby(['Year'], as_index=True).agg({
            'Cited by': np.sum
        })

        result = self.years_list()
        result = result.to_frame()
        result['Year'] = result.index
        result['Cited by'] = 0
        result.at[citations.index, 'Cited by'] = citations['Cited by'].tolist()
        result.index = range(len(result))

        if cumulative is True:
            result['Cited by'] = result['Cited by'].cumsum()

        return List(result)

    #----------------------------------------------------------------------------------------------
    def citations_by_terms_by_year(self, column, sep=None, top_n=None, minmax=None):
        """Computes the number of citations to docuement per year.

        >>> rdf = RecordsDataFrame({
        ...   'Year' :     [  2014, 2014, 2016,  2017, None, 2019 ],
        ...   'term' :     [ 'a;b', 'a;c',  'b',  'c', None,   'b'],
        ...   'Cited by' : [     1,    2,     3,    4,    3,     7]
        ... })

        >>> rdf.citations_by_terms_by_year('term', sep=';')
          term  Year  Cited by
        0    a  2014         3
        1    b  2014         1
        2    b  2016         3
        3    b  2019         7
        4    c  2014         2
        5    c  2017         4

        """
        citations = self[[column, 'Cited by', 'Year']]
        citations = citations.dropna()
        citations['Year'] = citations['Year'].map(int)
        if sep is not None:
            citations[column] = citations[column].map(lambda x: x.split(sep) if x is not None else None)
            citations[column] = citations[column].map(
                lambda x: [z.strip() for z in x] if isinstance(x, list) else x
            )
            citations = citations.explode(column)
            citations.index = range(len(citations))
        citations = citations.groupby(by=[column, 'Year'], as_index=True).agg({
            'Cited by': np.sum
        })

        ## results dataframe
        a = [t for t,_ in citations.index]
        b = [t for _,t in citations.index]
        result = pd.DataFrame({
            column : a,
            'Year' : b,
            'Cited by' : citations['Cited by'].tolist()
        })

        ## rows
        top = self.documents_by_terms(column, sep)
        if top_n is not None and len(top) > top_n:
            top = top[0:top_n][column].tolist()
            selected = [True if row[0] in top else False for idx, row in result.iterrows()] 
            result = result[selected]

        if minmax is not None:
            minval, maxval = minmax
            result = result[ result[result.columns[2]] >= minval ]
            result = result[ result[result.columns[2]] <= maxval ]

        return Matrix(result, rtype='coo-matrix')



    #----------------------------------------------------------------------------------------------
    #
    #  Analytical functions --- Analysis
    #
    #----------------------------------------------------------------------------------------------
    def tdf(self, column, sep, top_n=20):
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
        if top_n is not None and len(terms) > top_n:
            terms = terms[0:top_n]
        
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
    def autocorr(self, column, sep=None, top_n=20):
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

        >>> rdf.autocorr(column='A', sep=';', top_n=3) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
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
        result = self.crosscorr(column_r=column, column_c=column, sep_r=sep, sep_c=sep, top_n=top_n)
        result._rtype = 'auto-matrix'
        return result

    #----------------------------------------------------------------------------------------------
    def crosscorr(self, column_r, column_c=None, sep_r=None, sep_c=None, top_n=20):
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
        >>> rdf.crosscorr(column_r='c1', column_c='c2', sep_r=';', sep_c=';', top_n=3) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
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

        tdf_rows = self.tdf(column_r, sep_r, top_n)
        if column_c is not None:
            tdf_cols = self.tdf(column_c, sep_c, top_n)
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

        result['ID'] = None

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

                selected_IDs = self[(s1 > 0) & (s2 > 0)]['ID']
                if len(selected_IDs):
                    result.at[idx, 'ID'] = selected_IDs.tolist()

                idx += 1

        ## adds number of records to columns
        num = self.documents_by_terms(column_r, sep_r)
        new_names = {}
        for idx, row in num.iterrows():
            old_name = row[0]
            new_name = old_name + ' [' + str(row[1]) + ']'
            new_names[old_name] = new_name

        result[col0] = result[col0].map(lambda x: new_names[x])

        if column_c is not None:
            num = self.documents_by_terms(column_c, sep_c)
            new_names = {}
            for idx, row in num.iterrows():
                old_name = row[0]
                new_name = old_name + ' [' + str(row[1]) + ']'
                new_names[old_name] = new_name

        result[col1] = result[col1].map(lambda x: new_names[x])



        ## end adds numbers of records to columns

        result = result.sort_values(col2, ascending=False)
        result.index = range(len(result))
        return Matrix(result, rtype='cross-matrix')

    #----------------------------------------------------------------------------------------------
    def factor_analysis(self, column, sep=None, n_components=None, top_n=10):


        tdf = self.tdf(column, sep, top_n)
        terms = tdf.columns.tolist()

        if n_components is None:
            n_components = int(math.sqrt(len(set(terms))))

        pca = PCA(n_components=n_components)
        
        values = np.transpose(pca.fit(X=tdf.values).components_)

        cols = [['F'+str(i) for i in range(n_components)] for k in range(len(terms))]
        rows = [[t for n in range(n_components) ] for t in terms]
        values = [values[i,j] for i in range(len(terms)) for j in range(n_components)]

        cols = [e for row in cols for e in row]
        rows = [e for row in rows for e in row]

        result = pd.DataFrame({
            column : rows,
            'Factor' : cols,
            'value' : values})


        return Matrix(result, rtype='factor-matrix')


    #----------------------------------------------------------------------------------------------
    def most_cited_documents(self, top_n=10, minmax=None):
        """ Returns the top N most cited documents and citations > min_value .
        Args:
            top_n (int) : number of documents to be returned.
            min_value (int) : minimal number of citations

        Results:
            pandas.DataFrame
        """
        result = self.sort_values(by='Cited by', ascending=False)

        if top_n is not None and len(result) > top_n:
            result = result[0:top_n]

        if minmax is not None:
            minval, maxval = minmax
            result = result[ result['Cited by'] >= minval ]
            result = result[ result['Cited by'] <= maxval ]
        
        return result[['Title', 'Authors', 'Year', 'Cited by', 'ID']]


    #----------------------------------------------------------------------------------------------
    @property
    def num_of_sources(self):
        return len(self['Source title'].unique())

