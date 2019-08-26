"""
Functions for field transformation
===============================================================================

Overview
-------------------------------------------------------------------------------

The functions in this module allows the user to transform fields in a dataframe.



Functions in this module
-------------------------------------------------------------------------------

"""

import pandas as pd
import string
import re
import json

from techMiner.mapfunc import asciify, fingerprint



def makeCleanupDict(x, match=None, sep=None):
    """Create a dictionary for list cleanup.

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'f': ['a b c a b a', 
    ...          'a b c a b',
    ...          'a b c a b', 
    ...          'A C b', 
    ...          'a b', 
    ...          'a, b, c, a', 
    ...          'a B'],
    ... })
    >>> df
                 f
    0  a b c a b a
    1    a b c a b
    2    a b c a b
    3        A C b
    4          a b
    5   a, b, c, a
    6          a B

    >>> makeCleanupDict(df.f, match=None) # doctest: +NORMALIZE_WHITESPACE
    {'a b c a b': ['A C b', 'a b c a b', 'a b c a b a', 'a, b, c, a'], 'a b': ['a B', 'a b']}

    >>> makeCleanupDict(df.f, match='shortest') # doctest: +NORMALIZE_WHITESPACE
    {'A C b': ['A C b', 'a b c a b', 'a b c a b a', 'a, b, c, a'], 'a b': ['a B', 'a b']}

    >>> makeCleanupDict(df.f, match='longest') # doctest: +NORMALIZE_WHITESPACE
    {'a b c a b a': ['A C b', 'a b c a b', 'a b c a b a', 'a, b, c, a'], 'a B': ['a B', 'a b']}

    >>> df = pd.DataFrame({
    ...    'f': ['a b, c a, b a', 
    ...          'A b, c A, b',
    ...          'a b, C A, B', 
    ...          'A C, b', 
    ...          None,
    ...          'a b', 
    ...          'a, b, c, a', 
    ...          'a B'],
    ... })
    >>> df
                   f
    0  a b, c a, b a
    1    A b, c A, b
    2    a b, C A, B
    3         A C, b
    4           None
    5            a b
    6     a, b, c, a
    7            a B

    >>> import json
    >>> print(json.dumps(makeCleanupDict(df.f, match='longest', sep=','), indent=4, sort_keys=True)) # doctest: +NORMALIZE_WHITESPACE
    {
       "A C": [
           "A C",
           "C A",
           "c A",
           "c a"
       ],
       "a B": [
           "A b",
           "a B",
           "a b",
           "b a"
       ],
       "b": [
           "B",
           "b"
       ]
    }



    """

    x = x.dropna()
    if sep is not None:
        x = pd.Series([z.strip() for y in x for z in y.split(sep)]) 
    y = x.map(lambda w: fingerprint(w))
    y = y.sort_values()
    counts = y.value_counts()
    counts = counts[counts > 1]
    result = {}
    for z in counts.index.tolist():
        w = x[y == z]

        if match is None or match == 'mostfrequent':
            m = w.value_counts().sort_values()
            m = m[m == m[-1]].sort_index()
            groupName = m.index[-1]
            
        if match == 'longest' or match == 'shortest':
            m = pd.Series([len(a) for a in w], index = w.index).sort_values()
            if match == 'longest':
                groupName = w[m.index[-1]]
            else:
                groupName = w[m.index[0]]

        z = w.sort_values().unique().tolist()
        if len(z) > 1:
            result[groupName] = z

    return result
    

def printDict(x):
    """
    >>> d = {'b':[1, 2 ,3, 4], 'a': [5, 4, 2]} 
    >>> printDict(d) # doctest: +NORMALIZE_WHITESPACE
    {
       "a": [
           5,
           4,
           2
       ],
       "b": [
           1,
           2,
           3,
           4
       ]
    }

    """
    print(json.dumps(x, indent=4, sort_keys=True) )

def applyCleanupDict(x, d, sep=None):
    """

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'f': [0, 1, 2, 3, None, 4, 5, 6, 7, 8, 9],
    ... })
    >>> df
          f
    0   0.0
    1   1.0
    2   2.0
    3   3.0
    4   NaN
    5   4.0
    6   5.0
    7   6.0
    8   7.0
    9   8.0
    10  9.0
    
    >>> d = {'a':[0, 1, 2],
    ...      'b':[4, 5, 6],
    ...      'c':[7, 8, 9]}
    >>> df.f.map(lambda x: applyCleanupDict(x, d))
    0       a
    1       a
    2       a
    3       3
    4     NaN
    5       b
    6       b
    7       b
    8       c
    9       c
    10      c
    Name: f, dtype: object

    >>> df = pd.DataFrame({
    ...    'f': ['a b, A B', 'A b, A B', None, 'b c', 'b, B A', 'b, a, c', 'A, B'],
    ... })
    >>> df
              f
    0  a b, A B
    1  A b, A B
    2      None
    3       b c
    4    b, B A
    5   b, a, c
    6      A, B
    >>> d = {'0':['a b', 'A B', 'B A'],
    ...      '1':['b c'],
    ...      '2':['a', 'b']}
    >>> df.f.map(lambda x: applyCleanupDict(x, d, sep=','))
    0      0,0
    1    A b,0
    2     None
    3        1
    4      2,0
    5    2,2,c
    6      A,B
    Name: f, dtype: object


    """

    if x is None:
        return None

    if sep is None:
        for k in d.keys():
            if x in d[k]:
                return k
        return x
    else:
        result = []
        for z in x.split(sep):
            z = z.strip()
            found = False
            for k in d.keys():
                if z in d[k]:
                    result += [k]
                    found = True
                    break
            if found is False:    
                result += [z]
        return sep.join(result)
