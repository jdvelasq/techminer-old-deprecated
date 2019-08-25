"""
Functions for manipulating records (rows) in a dataframe
===============================================================================

Overview
-------------------------------------------------------------------------------

The functions in this module allows the user to manipualte the current dataframe
in order to group, delete, modify and add new information to the current dataframe
of bibliographical information.

Functions in this module
-------------------------------------------------------------------------------


"""

import pandas as pd
import string
import re

from techMiner.common import asciify, fingerprint

def displayRecords(df):
    """Show one or more records of dataframe at a time. User can use standard
    pandas.DataFrame funcions to select and order specific records or a dataframe.

    Args:
        df (pandas.DataFrame): Generic pandas.DataFrame.

    Returns:
        None.

    **Examples**

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ... 'f0': ['a', 'b', 'a', 'c', 'a', 'e'],
    ... 'f1': [  1,   2,   2,   3,   1,   5]
    ... })
    >>> df
      f0  f1
    0  a   1
    1  b   2
    2  a   2
    3  c   3
    4  a   1
    5  e   5

    >>> displayRecords(df)   
    Record index: 0
        f0    a
        f1    1
        Name: 0, dtype: object
    <BLANKLINE>
    Record index: 1
        f0    b
        f1    2
        Name: 1, dtype: object
    <BLANKLINE>
    Record index: 2
        f0    a
        f1    2
        Name: 2, dtype: object
    <BLANKLINE>
    Record index: 3
        f0    c
        f1    3
        Name: 3, dtype: object
    <BLANKLINE>
    Record index: 4
        f0    a
        f1    1
        Name: 4, dtype: object
    <BLANKLINE>
    Record index: 5
        f0    e
        f1    5
        Name: 5, dtype: object
    <BLANKLINE>
    <BLANKLINE>

    >>> displayRecords(df.sort_values(by='f0'))
    Record index: 0
        f0    a
        f1    1
        Name: 0, dtype: object
    <BLANKLINE>
    Record index: 2
        f0    a
        f1    2
        Name: 2, dtype: object
    <BLANKLINE>
    Record index: 4
        f0    a
        f1    1
        Name: 4, dtype: object
    <BLANKLINE>
    Record index: 1
        f0    b
        f1    2
        Name: 1, dtype: object
    <BLANKLINE>
    Record index: 3
        f0    c
        f1    3
        Name: 3, dtype: object
    <BLANKLINE>
    Record index: 5
        f0    e
        f1    5
        Name: 5, dtype: object
    <BLANKLINE>
    <BLANKLINE>

    """

    result = ''
    for idx in df.index:
        result += 'Record index: ' + idx.__str__() + '\n'
        x = df.loc[idx].__str__()
        x = x.split('\n')
        for y in x:
            result += '    ' + y + '\n'
        result += '\n'
    print(result)


def coverage(df):
    """Counts the number of None.

    Args:
        df (pandas.DataFrame): Generic pandas.DataFrame.

    Returns:
        None.

    **Examples**

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ... 'f0': ['a', None, 'a', None,    'a', 'e'],
    ... 'f1': [  1,    2,   2,    3,   None,   5]
    ... })
    >>> coverage(df)
      Field  Number of items  Coverage
    0    f0                4  0.666667
    1    f1                5  0.833333


    """

    result = pd.DataFrame({
        'Field': df.columns,
        'Number of items': [len(df) - df[col].isnull().sum() for col in df.columns],
        'Coverage': [(len(df) - df[col].isnull().sum()) / len(df) for col in df.columns]
    })

    return result
        

def removeDuplicateRecords(df, fields, matchType='strict'):
    """Remove duplicate records in a dataframe based in the velue of one 
    or more fields.

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ... 'f0': ['a', 'b', 'a', 'c', 'a', 'e'],
    ... 'f1': [  1,   2,   2,   3,   1,   5]
    ... })
    >>> df
      f0  f1
    0  a   1
    1  b   2
    2  a   2
    3  c   3
    4  a   1
    5  e   5

    >>> removeDuplicateRecords(df, fields='f0')
      f0  f1
    0  a   1
    1  b   2
    3  c   3
    5  e   5

    >>> removeDuplicateRecords(df, fields=['f0'])
      f0  f1
    0  a   1
    1  b   2
    3  c   3
    5  e   5

    >>> removeDuplicateRecords(df, fields=['f0', 'f1'])
      f0  f1
    0  a   1
    1  b   2
    2  a   2
    3  c   3
    5  e   5

    >>> df = pd.DataFrame({
    ... 'f0': ['A;', 'b', 'A,', 'c', 'a', 'e'],
    ... 'f1': [  1,   2,   2,   3,   1,   5]
    ... })
    >>> df
       f0  f1
    0  A;   1
    1   b   2
    2  A,   2
    3   c   3
    4   a   1
    5   e   5

    >>> removeDuplicateRecords(df, fields='f0', matchType='fingerprint')
       f0  f1
    0  A;   1
    1   b   2
    3   c   3
    5   e   5


    """

    df0 = df.copy()
    if isinstance(fields, str):
        df0 = df0[[fields]]
    elif isinstance(fields, list):
        df0 = df0[fields]
    else:
        ## generar error
        pass

    if matchType == 'strict':

        df0 = df0.drop_duplicates()
        return df.loc[df0.index,:]

    if matchType == 'fingerprint':
        for field in df0.columns:
            df0[field] =  df0[field].map(lambda x: fingerprint(x))
        
        df0 = df0.drop_duplicates()
        return df.loc[df0.index,:]

    if matchType == 'fuzzy':
        pass






