"""
techMiner.Thesaurus
===============================

""" 
import pandas as pd
import json
from techMiner.strings import (
    find_string, 
    replace_string,
    fingerprint)

def text_clustering(x, name_strategy='mostfrequent', search_strategy='fingerprint', sep=None):
    """Builds a thesaurus by clustering a list of strings.
    
    Args:
        x (list): list  of string to create thesaurus.

        name_strategy (string): method for assigning keys in thesaurus.

            * 'mostfrequent': Most frequent string in the cluster.

            * 'longest': Longest string in the cluster.

            * 'shortest': Shortest string in the cluster.

        search_strategy (string): cluster method.

            * 'fingerprint'.

        sep (string): separator character for elements in `x`.

    Returns:
        A Thesaurus object.

    
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

    >>> print(text_clustering(df.f).to_json()) # doctest: +NORMALIZE_WHITESPACE
    {
      "a b": [
        "a B",
        "a b"
        ],
      "a b c a b": [
        "A C b",
        "a b c a b",
        "a b c a b a",
        "a, b, c, a"
        ]
    }

    >>> print(text_clustering(df.f, name_strategy='shortest').to_json()) # doctest: +NORMALIZE_WHITESPACE
    {
      "A C b": [
        "A C b",
        "a b c a b",
        "a b c a b a",
        "a, b, c, a"
        ],
      "a b": [
        "a B",
        "a b"
        ]
    }

    >>> print(text_clustering(df.f, name_strategy='longest').to_json()) # doctest: +NORMALIZE_WHITESPACE
    {
      "a B": [
        "a B",
        "a b"
        ],
      "a b c a b a": [
        "A C b",
        "a b c a b",
        "a b c a b a",
        "a, b, c, a"
        ]
    }

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

    >>> print(text_clustering(df.f, sep=',', name_strategy='longest').to_json()) # doctest: +NORMALIZE_WHITESPACE
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

    if search_strategy == 'fingerprint':
        y = x.map(lambda w: fingerprint(w))
    y = y.sort_values()
    counts = y.value_counts()
    counts = counts[counts > 1]

    result = {}
    for z in counts.index.tolist():

        w = x[y == z]

        if name_strategy is None or name_strategy == 'mostfrequent':
            m = w.value_counts().sort_values()
            m = m[m == m[-1]].sort_index()
            groupName = m.index[-1]
            
        if name_strategy == 'longest' or name_strategy == 'shortest':
            m = pd.Series([len(a) for a in w], index = w.index).sort_values()
            if name_strategy == 'longest':
                groupName = w[m.index[-1]]
            else:
                groupName = w[m.index[0]]

        z = w.sort_values().unique().tolist()
        if len(z) > 1:
            result[groupName] = z

    return Thesaurus(result, ignore_case=False, full_match=True)


class Thesaurus:
    def __init__(self, x={}, ignore_case=True, full_match=False, use_re=False):
        self._thesaurus = x
        self._ignore_case = ignore_case
        self._full_match = full_match
        self._use_re = use_re

    def cleanup(self, x, sep=None):
        """

        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...    'f': ['0', '1', '2', '3', None, '4', '5', '6', '7', '8', '9'],
        ... })
        >>> df # doctest: +NORMALIZE_WHITESPACE
               f
        0      0
        1      1
        2      2
        3      3
        4   None
        5      4
        6      5
        7      6
        8      7
        9      8
        10     9
        
        >>> d = {'a':['0', '1', '2'],
        ...      'b':['4', '5', '6'],
        ...      'c':['7', '8', '9']}

        >>> df.f.map(lambda x: Thesaurus(d, ignore_case=False, full_match=True).apply(x)) # doctest: +NORMALIZE_WHITESPACE
        0        a
        1        a
        2        a
        3        3
        4     None
        5        b
        6        b
        7        b
        8        c
        9        c
        10       c
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
        >>> df.f.map(lambda x: Thesaurus(d, ignore_case=False, full_match=True).apply(x, sep=','))
        0      0,0
        1    A b,0
        2     None
        3        1
        4      2,0
        5    2,2,c
        6      A,B
        Name: f, dtype: object


        """

        return self.apply(x, sep=sep)


    def apply(self, x, sep=None):
        """

        >>> df = pd.DataFrame({
        ...    'f': ['- AAA -, - BBB -', '- BBBB - CCC -', None, 'DDD'],
        ... })
        >>> df
                          f
        0  - AAA -, - BBB -
        1    - BBBB - CCC -
        2              None
        3               DDD
        >>> d = {'0':['AAA'],
        ...      '1':['BBB']}
        >>> df.f.map(lambda x: Thesaurus(d).apply(x, sep=','))
        0     0,1
        1       1
        2    None
        3     DDD
        Name: f, dtype: object

    """

        if x is None:
            return None

        if sep is None:

            for key in self._thesaurus.keys():

                for pattern in self._thesaurus[key]:

                    y = find_string(
                        pattern = pattern,
                        x = x.strip(),
                        ignore_case = self._ignore_case,
                        full_match = self._full_match,
                        use_re = self._use_re
                    )

                    if y is not None and len(y):
                        return key

            return x

        else:
            result = []
            for z in x.split(sep):
                z = z.strip()
                for key in self._thesaurus.keys():
                    found = False
                    for pattern in self._thesaurus[key]:

                        y = find_string(
                            pattern = pattern,
                            x = z,
                            ignore_case = self._ignore_case,
                            full_match = self._full_match,
                            use_re = self._use_re
                        )

                        if y is not None and len(y):
                            result += [key]
                            found = True
                            break

                    if found:
                        break

                if not found:
                    result += [z]

            if len(result):
                return sep.join(result)
            
        return None

    def findAndReplace(self, x, sep=None):
        """Applies a thesaurus to a string, reemplacing the portion of string
        matching the current pattern with the key.

        >>> df = pd.DataFrame({
        ...    'f': ['- AAA -, - BBB -', '- BBBB - CCC -', None, 'DDD'],
        ... })
        >>> df
                          f
        0  - AAA -, - BBB -
        1    - BBBB - CCC -
        2              None
        3               DDD
        >>> d = {'0':['AAA'],
        ...      '1':['BBB']}
        >>> df.f.map(lambda x: Thesaurus(d).findAndReplace(x, sep=','))
        0     - 0 -,- 1 -
        1    - 1B - CCC -
        2            None
        3             DDD
        Name: f, dtype: object

        """

        if x is None:
            return None

        if sep is None:

            for key in self._thesaurus.keys():

                for pattern in self._thesaurus[key]:

                    x = replace_string(
                        pattern = pattern,
                        x = x,
                        repl = key,
                        ignore_case = self._ignore_case,
                        full_match = self._full_match,
                        use_re = self._use_re
                    )

            return x

        else:
            result = []
            for z in x.split(sep):
                z = z.strip()
                for key in self._thesaurus.keys():
                    found = False
                    for pattern in self._thesaurus[key]:

                        z = replace_string(
                            pattern = pattern,
                            x = z,
                            repl = key,
                            ignore_case = self._ignore_case,
                            full_match = self._full_match,
                            use_re = self._use_re
                        )

                result += [z]
                            
            return sep.join(result)
            
        return None




    def to_json(self):
        """Returns a json representation of the Thesaurus.
        """
        return json.dumps(self._thesaurus, indent=2, sort_keys=True)


    def merge_keys(self, key, popkey):
        """Adds the strings associated to popkey to key and delete popkey.
        """
        if isinstance(popkey, list):
            for k in popkey:
                self._thesaurus[key] = self._thesaurus[key] + self._thesaurus[k]        
                self._thesaurus.pop(k)
        else:
            self._thesaurus[key] = self._thesaurus[key] + self._thesaurus[popkey]
            self._thesaurus.pop(popkey)

    def pop_key(self, key):
        """Deletes key from thesaurus.
        """
        self._thesaurus.pop(key)
        
    def change_key(self, current_key, new_key):
        self._thesaurus[new_key] = self._thesaurus[current_key]
        self._thesaurus.popkey(current_key)
