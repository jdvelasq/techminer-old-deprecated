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
import json

from techMiner.transform import asciify, fingerprint

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
    -----------------------------------------------
    Record index: 0
    {
      "f0": "a",
      "f1": 1
    }
    -----------------------------------------------
    Record index: 1
    {
      "f0": "b",
      "f1": 2
    }
    -----------------------------------------------
    Record index: 2
    {
      "f0": "a",
      "f1": 2
    }
    -----------------------------------------------
    Record index: 3
    {
      "f0": "c",
      "f1": 3
    }
    -----------------------------------------------
    Record index: 4
    {
      "f0": "a",
      "f1": 1
    }
    -----------------------------------------------
    Record index: 5
    {
      "f0": "e",
      "f1": 5
    }

    """

    x = df.apply(lambda x: x.to_json(), axis = 1)
    x = x.sort_index()
    index = x.index   
    for idx, y in zip(index, x):

        print('-----------------------------------------------')
        print('Record index: ' + str(idx))
        parsed = json.loads(y)
        print(json.dumps(parsed, indent=2, sort_keys=True))



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


def merge_fields(fieldA, fieldB, sepA=None, sepB=None, new_sep=';'):

    if sepA is not None:
        fieldA = [x if x is None else [z.strip() for z in x.split(sepA)] for x in fieldA]
    else:
        fieldA = [x for x in fieldA]
        
    if sepB is not None:
        fieldB = [x if x is None else [z.strip() for z in x.split(sepB)] for x in fieldB]
    else:
        fieldB = [x for x in fieldB]
    
    result = []
    for a, b in zip(fieldA, fieldB):        
        if a is None and b is None:
            result.append(None)
        elif a is None:    
            if b is not None and not isinstance(b, list):
                b = [b]    
            result.append(b)
        elif b is None:
            if a is not None and not isinstance(a, list):
                a = [a]
            result.append(a)
        else:
            if a is not None and not isinstance(a, list):
                a = [a]
            if b is not None and not isinstance(b, list):
                b = [b]   
            
            a.extend(b)
            a = list(set(a))
            result.append(a)

    result = pd.Series([new_sep.join(x) if x is not None else x for x in result])
    #result = pd.Series(result, index=df.index)
    
    return result
